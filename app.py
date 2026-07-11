import os
import re
import uuid
import logging
from typing import List, Dict, Any, Optional, Tuple

import requests
import yaml
from fastapi import FastAPI, HTTPException, Security
from fastapi.security.api_key import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
import chromadb
from chromadb.utils import embedding_functions
from rapidfuzz import process, fuzz

# =====================================================================
# 0. KONFIGURATION (.env statt Klartext-Credentials im Code)
# =====================================================================
load_dotenv()

OPENHAB_URL = os.getenv("OPENHAB_URL", "http://192.168.0.10:8080").rstrip("/")
OPENHAB_TOKEN = os.getenv("OPENHAB_TOKEN", "")
API_KEY = os.getenv("API_KEY", "your_local_key")
SYNONYMS_PATH = os.getenv("SYNONYMS_PATH", "./synonyms.yaml")
CHROMA_PATH = os.getenv("CHROMA_PATH", "./chroma_db")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("oh-ai-bridge")

app = FastAPI(title="openHAB Semantic Hybrid Bridge v4")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
api_key_header = APIKeyHeader(name="Authorization", auto_error=False)

chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL)
collection = chroma_client.get_or_create_collection(name="openhab_items", embedding_function=emb_fn)

HEADERS_JSON = {"Authorization": f"Bearer {OPENHAB_TOKEN}", "Accept": "application/json"}

# =====================================================================
# 1. SYNONYM-KONFIGURATION LADEN (admin-pflegbar, siehe synonyms.yaml)
# =====================================================================
def load_synonym_config() -> Dict[str, Any]:
    try:
        with open(SYNONYMS_PATH, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    except FileNotFoundError:
        log.warning(f"[CONFIG] {SYNONYMS_PATH} nicht gefunden, nutze leere Fallback-Konfiguration.")
        data = {}
    data.setdefault("equipment", {})
    data.setdefault("location", {})
    data.setdefault("actions", {})
    data.setdefault("fuzzy", {"min_score": 82})
    return data


SYN_CONFIG = load_synonym_config()
FUZZY_MIN_SCORE = SYN_CONFIG.get("fuzzy", {}).get("min_score", 82)

# Wird beim Sync mit den LIVE-Synonymen aus openHAB /rest/tags gemerged.
# Struktur nach Merge: {"Lightbulb": {"licht", "lampe", ...}, ...}
EQUIPMENT_SYNONYMS: Dict[str, set] = {}
LOCATION_SYNONYMS: Dict[str, set] = {}

# Umgekehrte Nachschlage-Tabellen fürs Fuzzy-Matching: Wort -> (kind, tag)
VOCAB_EQUIPMENT: Dict[str, str] = {}
VOCAB_LOCATION: Dict[str, str] = {}


def _merge_synonym_dict(base_cfg: Dict[str, List[str]], target: Dict[str, set]):
    for tag, words in base_cfg.items():
        target.setdefault(tag, set())
        for w in words:
            target[tag].add(normalize_text(w))


def rebuild_vocab():
    """Baut die Flat-Vokabular-Tabellen für exakte + Fuzzy-Suche neu auf."""
    VOCAB_EQUIPMENT.clear()
    VOCAB_LOCATION.clear()
    for tag, words in EQUIPMENT_SYNONYMS.items():
        for w in words:
            VOCAB_EQUIPMENT[w] = tag
    for tag, words in LOCATION_SYNONYMS.items():
        for w in words:
            VOCAB_LOCATION[w] = tag


# --- HELFER: UMLAUT-BEREINIGUNG (unverändert aus v3, bewährt) ---
def normalize_text(text: str) -> str:
    """ Konvertiert deutsche Umlaute in ASCII-Schreibweise (ä -> ae, etc.) """
    text = text.lower().strip()
    replacements = {'ä': 'ae', 'ö': 'oe', 'ü': 'ue', 'ß': 'ss'}
    for umlaut, rep in replacements.items():
        text = text.replace(umlaut, rep)
    return text


# =====================================================================
# 2. OPENHAB SEMANTIC MODEL (dasselbe Modell wie HABot)
# =====================================================================
# openHAB >= 4.0 stellt unter /rest/tags die vollständige Tag-Hierarchie
# (Location/Equipment/Point/Property) inkl. lokalisierter Synonyme bereit
# -- exakt die Quelle, die HABot selbst zur NLU-Auflösung nutzt.
# Wir laden sie live, statt (wie in v3) Räume/Gerätetypen hart im Code zu
# verdrahten. Schlägt der Request fehl (ältere openHAB-Version, Endpoint
# deaktiviert), fallen wir sauber auf die YAML-Konfiguration zurück.
#
# HINWEIS: Das exakte JSON-Schema von /rest/tags kann sich je nach
# openHAB-Version leicht unterscheiden (z.B. Feldname "uid" vs "name",
# "parentTag" vs "parent"). Der Parser unten ist daher defensiv und
# probiert mehrere gängige Feldnamen. Bitte einmal gegen /api/tags
# (Debug-Endpoint dieser Bridge) prüfen, ob die Zuordnung für dein
# System stimmt, und ggf. die Feldnamen unten anpassen.

_TAG_PARENT: Dict[str, Optional[str]] = {}
_TAG_SYNONYMS_DE: Dict[str, List[str]] = {}
_TAG_CATEGORY_CACHE: Dict[str, Optional[str]] = {}

ROOT_CATEGORIES = {"Location", "Equipment", "Point", "Property"}


def fetch_openhab_tag_registry() -> bool:
    """Lädt /rest/tags (deutsche Synonyme via Accept-Language) und baut
    die Parent-Hierarchie auf. Gibt True zurück bei Erfolg."""
    global _TAG_PARENT, _TAG_SYNONYMS_DE
    _TAG_PARENT = {}
    _TAG_SYNONYMS_DE = {}
    try:
        headers = dict(HEADERS_JSON)
        headers["Accept-Language"] = "de"
        res = requests.get(f"{OPENHAB_URL}/rest/tags", headers=headers, timeout=20)
        res.raise_for_status()
        tags = res.json()
        if not isinstance(tags, list):
            return False
        for t in tags:
            uid = t.get("uid") or t.get("name") or t.get("id")
            if not uid:
                continue
            parent = t.get("parentTag") or t.get("parent")
            synonyms_raw = t.get("synonyms") or ""
            if isinstance(synonyms_raw, str):
                syns = [s.strip() for s in synonyms_raw.split(",") if s.strip()]
            elif isinstance(synonyms_raw, list):
                syns = synonyms_raw
            else:
                syns = []
            label = t.get("label")
            if label:
                syns.append(label)
            # Der kurze Tag-Name (letztes Segment nach "_") ist selbst auch ein Wort
            short = uid.split("_")[-1]
            syns.append(short)
            _TAG_PARENT[uid] = parent
            _TAG_SYNONYMS_DE[uid] = syns
        log.info(f"[SEMANTIC] {len(tags)} Tags von {OPENHAB_URL}/rest/tags geladen.")
        return True
    except Exception as e:
        log.warning(f"[SEMANTIC] /rest/tags nicht verfügbar ({e}). Nutze YAML-Fallback für Räume/Geräte.")
        return False


def tag_category(uid: Optional[str]) -> Optional[str]:
    """Läuft die Parent-Kette hoch bis Location/Equipment/Point/Property."""
    if not uid:
        return None
    if uid in _TAG_CATEGORY_CACHE:
        return _TAG_CATEGORY_CACHE[uid]
    seen = set()
    current = uid
    while current and current not in seen:
        if current in ROOT_CATEGORIES:
            _TAG_CATEGORY_CACHE[uid] = current
            return current
        seen.add(current)
        current = _TAG_PARENT.get(current)
    _TAG_CATEGORY_CACHE[uid] = None
    return None


def build_synonym_tables_from_openhab():
    """Übersetzt die geladene Tag-Hierarchie in unsere EQUIPMENT_/LOCATION_
    SYNONYMS-Tabellen und merged sie mit der YAML-Fallback-Konfiguration."""
    global EQUIPMENT_SYNONYMS, LOCATION_SYNONYMS
    EQUIPMENT_SYNONYMS = {}
    LOCATION_SYNONYMS = {}

    for uid, syns in _TAG_SYNONYMS_DE.items():
        cat = tag_category(uid)
        short = uid.split("_")[-1]
        words = {normalize_text(s) for s in syns if s}
        if cat == "Equipment":
            EQUIPMENT_SYNONYMS.setdefault(short, set()).update(words)
        elif cat == "Location":
            LOCATION_SYNONYMS.setdefault(short, set()).update(words)

    # Merge mit YAML (eigene Ergänzungen / Fallback wenn openHAB nichts lieferte)
    _merge_synonym_dict(SYN_CONFIG.get("equipment", {}), EQUIPMENT_SYNONYMS)
    _merge_synonym_dict(SYN_CONFIG.get("location", {}), LOCATION_SYNONYMS)

    rebuild_vocab()
    log.info(f"[SEMANTIC] {len(EQUIPMENT_SYNONYMS)} Equipment-Klassen, "
              f"{len(LOCATION_SYNONYMS)} Location-Klassen im Vokabular.")


# --- ACTION-SYNONYME (rein YAML, kein Teil des openHAB Semantic Models) ---
ACTION_WORDS: Dict[str, set] = {}


def build_action_table():
    ACTION_WORDS.clear()
    for action, words in SYN_CONFIG.get("actions", {}).items():
        ACTION_WORDS[action] = {normalize_text(w) for w in words}


build_action_table()


# =====================================================================
# 3. FUZZY-MATCHING (Tippfehler-Toleranz, s. "Fuzzy Matching"-Kapitel)
# =====================================================================
def fuzzy_lookup(word: str, vocab: Dict[str, str]) -> Optional[str]:
    """Exakter Treffer zuerst, sonst RapidFuzz-Korrektur (z.B. 'lich' -> 'licht')."""
    if word in vocab:
        return vocab[word]
    if not vocab:
        return None
    match = process.extractOne(word, vocab.keys(), scorer=fuzz.WRatio, score_cutoff=FUZZY_MIN_SCORE)
    if match:
        return vocab[match[0]]
    return None


def resolve_equipment(normalized_query: str) -> Optional[str]:
    # 1) Mehrwort-Synonyme zuerst (Substring-Check, längste zuerst = spezifischer)
    for word, tag in sorted(VOCAB_EQUIPMENT.items(), key=lambda x: -len(x[0])):
        if word and word in normalized_query:
            return tag
    # 2) Fuzzy je Token (fängt Tippfehler wie "lich" statt "licht" ab)
    for token in normalized_query.split():
        tag = fuzzy_lookup(token, VOCAB_EQUIPMENT)
        if tag:
            return tag
    return None


def resolve_location(normalized_query: str) -> Optional[str]:
    for word, tag in sorted(VOCAB_LOCATION.items(), key=lambda x: -len(x[0])):
        if word and word in normalized_query:
            return tag
    for token in normalized_query.split():
        tag = fuzzy_lookup(token, VOCAB_LOCATION)
        if tag:
            return tag
    return None


def resolve_action(query_lower: str) -> Optional[str]:
    """Reihenfolge wichtig: Fragen ('wie', 'ist') schlagen reine An/Aus-Worte,
    damit 'ist das Licht an?' als STATUS statt als ON erkannt wird."""
    if query_lower.strip().startswith(("sind ", "ist ", "hat ", "läuft ", "laeuft ", "gibt ")):
        return "STATUS"
    for action in ["STATUS", "OFF", "ON", "UP", "DOWN"]:
        words = ACTION_WORDS.get(action, set())
        if any(w in query_lower for w in words if w):
            return action
    return None


# =====================================================================
# 4. REQUEST-MODELLE (OpenAI-kompatibel)
# =====================================================================
class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.7


# =====================================================================
# 5. SYNC: Items von openHAB holen, semantisch klassifizieren, in Chroma
# =====================================================================
def _collect_item_tags(item: dict, items_by_name: Dict[str, dict]) -> List[str]:
    """Sammelt Tags des Items selbst + aller Ancestor-Gruppen (wie HABot:
    'Checks ancestor groups one level at a time'), ohne zusätzliche
    REST-Calls, da /rest/items bereits tags + groupNames aller Items liefert."""
    collected = list(item.get("tags", []))
    visited = set()
    frontier = list(item.get("groupNames", []))
    while frontier:
        gname = frontier.pop()
        if gname in visited:
            continue
        visited.add(gname)
        group = items_by_name.get(gname)
        if not group:
            continue
        collected.extend(group.get("tags", []))
        frontier.extend(group.get("groupNames", []))
    return collected


def classify_item_semantics(item: dict, items_by_name: Dict[str, dict]) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """Gibt (location_tag, equipment_tag, point_tag) zurück, aufgelöst über
    die echte openHAB-Tag-Hierarchie (Ancestor-Walk wie bei HABot)."""
    tags = _collect_item_tags(item, items_by_name)
    location_tag = equipment_tag = point_tag = None
    for t in tags:
        cat = tag_category(t)
        short = t.split("_")[-1] if "_" in t else t
        if cat == "Location" and not location_tag:
            location_tag = short
        elif cat == "Equipment" and not equipment_tag:
            equipment_tag = short
        elif cat == "Point" and not point_tag:
            point_tag = short
        elif cat is None:
            # Keine bekannte Kategorie (altes openHAB ohne Tag-Registry,
            # oder Tag-Name entspricht direkt einem unserer YAML-Keys) ->
            # gegen YAML-Synonymtabellen matchen als Fallback.
            norm = normalize_text(t)
            if not equipment_tag and norm in EQUIPMENT_SYNONYMS:
                equipment_tag = t
            if not location_tag and norm in LOCATION_SYNONYMS:
                location_tag = t
    return location_tag, equipment_tag, point_tag


@app.post("/api/sync")
def sync_items_to_vector_db():
    """ Holt alle Items von openHAB, klassifiziert sie über das Semantic
    Model (Location/Equipment, wie HABot) und führt ein Upsert in ChromaDB durch. """
    tag_registry_ok = fetch_openhab_tag_registry()
    build_synonym_tables_from_openhab()

    try:
        log.info("[SYNC] Rufe Items von openHAB ab...")
        response = requests.get(f"{OPENHAB_URL}/rest/items", headers=HEADERS_JSON, timeout=60)
        response.raise_for_status()
        items = response.json()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Fehler beim openHAB-Verbindungsaufbau: {str(e)}")

    items_by_name = {i["name"]: i for i in items}

    ids, documents, metadatas = [], [], []
    log.info(f"[SYNC] Analysiere {len(items)} Items (Semantic Model: {'live' if tag_registry_ok else 'YAML-Fallback'})...")

    for item in items:
        name = item.get("name", "")
        label = item.get("label", "") or name
        type_ = item.get("type", "")
        tags = item.get("tags", [])
        group_names = item.get("groupNames", [])

        location_tag, equipment_tag, point_tag = classify_item_semantics(item, items_by_name)

        # openbrain-Metadaten weiterhin unterstützen (Altbestand aus v3)
        openbrain_hint = openbrain_role = ""
        if "metadata" in item and "openbrain" in item["metadata"]:
            ob = item["metadata"]["openbrain"].get("value", "")
            hint_match = re.search(r'hint=([^,]+)', ob)
            role_match = re.search(r'role=([^,]+)', ob)
            if hint_match:
                openbrain_hint = hint_match.group(1)
            if role_match:
                openbrain_role = role_match.group(1)

        tags_str = ", ".join(tags)
        groups_str = ", ".join(group_names)
        semantic_text = (
            f"item: {label.lower()}. name: {name.lower()}. typ: {type_.lower()}. "
            f"tags: {tags_str.lower()}. gruppen: {groups_str.lower()}."
        )
        if location_tag:
            semantic_text += f" raum: {location_tag.lower()}."
        if equipment_tag:
            semantic_text += f" geraetetyp: {equipment_tag.lower()}."
        if openbrain_hint:
            semantic_text += f" info: {openbrain_hint.lower()}."
        if openbrain_role:
            semantic_text += f" rolle: {openbrain_role.lower()}."

        ids.append(name)
        documents.append(semantic_text)
        metadatas.append({
            "name": name,
            "type": type_,
            "label": label,
            "location_tag": location_tag or "",
            "equipment_tag": equipment_tag or "",
            "point_tag": point_tag or "",
        })

    batch_size = 5000
    log.info("[SYNC] Schreibe Daten in die lokale Vektordatenbank...")
    for i in range(0, len(ids), batch_size):
        collection.upsert(
            ids=ids[i:i + batch_size],
            documents=documents[i:i + batch_size],
            metadatas=metadatas[i:i + batch_size],
        )

    classified = sum(1 for m in metadatas if m["location_tag"] or m["equipment_tag"])
    log.info("[SYNC] Synchronisierung abgeschlossen.")
    return {
        "status": "success",
        "message": f"{len(ids)} Items synchronisiert.",
        "semantic_model_source": "openhab_live" if tag_registry_ok else "yaml_fallback",
        "items_with_semantic_classification": classified,
    }


@app.get("/api/tags")
def debug_tag_registry():
    """Debug-Endpoint: zeigt, wie die Bridge deine openHAB-Tags interpretiert
    hat, damit du Feldnamen-Abweichungen (siehe Kommentar oben) erkennen kannst."""
    return {
        "equipment_classes": {k: sorted(v) for k, v in EQUIPMENT_SYNONYMS.items()},
        "location_classes": {k: sorted(v) for k, v in LOCATION_SYNONYMS.items()},
        "raw_tag_count": len(_TAG_PARENT),
    }


@app.get("/health")
def health():
    return {"status": "ok"}


# =====================================================================
# 6. REST-INTERAKTION MIT OPENHAB
# =====================================================================
def execute_openhab_command(item_name: str, command: str) -> str:
    headers = {"Authorization": f"Bearer {OPENHAB_TOKEN}", "Content-Type": "text/plain"}
    res = requests.post(f"{OPENHAB_URL}/rest/items/{item_name}", data=command, headers=headers, timeout=15)
    return "Erfolgreich geschaltet" if res.status_code == 200 else f"Fehler (Status {res.status_code})"


def get_openhab_state(item_name: str) -> str:
    headers = {"Authorization": f"Bearer {OPENHAB_TOKEN}"}
    res = requests.get(f"{OPENHAB_URL}/rest/items/{item_name}/state", headers=headers, timeout=15)
    return res.text if res.status_code == 200 else "Unbekannt"


# =====================================================================
# 7. KONTEXT-GEDÄCHTNIS AUS DER CONVERSATION (statt globaler Variable)
# =====================================================================
# v3 hielt das "zuletzt verwendete Item" in einer globalen Python-Variable.
# Das bricht bei mehreren gleichzeitigen Nutzern/Chats (Open WebUI kann
# mehrere Unterhaltungen parallel bedienen -> ein globaler Zustand würde
# zwischen fremden Chats "durchsickern"). Open WebUI schickt im OpenAI-
# Format aber ohnehin die komplette bisherige Unterhaltung mit -> wir lesen
# das zuletzt verwendete Item direkt aus request.messages, statt es global
# zu speichern. Damit ist die Bridge zustandslos und pro Chat korrekt.
ITEM_REF_PATTERN = re.compile(r"`([A-Za-z0-9_]+)`")


def find_last_used_item(messages: List[ChatMessage]) -> Optional[str]:
    for msg in reversed(messages[:-1]):  # letzte Nachricht ist die aktuelle Anfrage
        if msg.role != "assistant":
            continue
        matches = ITEM_REF_PATTERN.findall(msg.content)
        for candidate in matches:
            # Wir haben Item-Namen in Backticks immer als *zweiten* Treffer
            # ausgegeben (Label zuerst) -> einfach das letzte Backtick-Wort
            # nehmen, das tatsächlich ein bekanntes Item ist.
            try:
                existing = collection.get(ids=[candidate])
                if existing and existing.get("ids"):
                    return candidate
            except Exception:
                continue
    return None


def get_item_meta(item_name: str) -> Optional[Dict[str, Any]]:
    try:
        result = collection.get(ids=[item_name])
        if result and result.get("ids"):
            return result["metadatas"][0]
    except Exception:
        return None
    return None


# =====================================================================
# 8. NUMERISCHE WERTE (Prozent / Grad) SAUBER PARSEN
# =====================================================================
NUMBER_PATTERN = re.compile(r"(\d+(?:[.,]\d+)?)\s*(%|prozent|grad|°c?|°)?")


def extract_numeric_command(user_query: str) -> Optional[str]:
    match = NUMBER_PATTERN.search(user_query)
    if not match:
        return None
    value = match.group(1).replace(",", ".")
    # openHAB erwartet für Setpoints/Dimmer i.d.R. reine Zahlen ohne Einheit
    if value.endswith(".0"):
        value = value[:-2]
    return value


# =====================================================================
# 9. HYBRIDE SUCHE: Tag-Filter (präzise) + Vektorsuche (unscharf) kombiniert
# =====================================================================
def query_candidates(normalized_query: str, equipment_tag: Optional[str], location_tag: Optional[str], n_results: int = 5):
    """Progressive Filterung: erst beide Tags, dann einzeln, dann frei.
    Das ist der Kern der Optimierung: statt reiner Textsuche wird zuerst
    über das Semantic Model präzise vorgefiltert (viel weniger Verwechslungen
    zwischen z.B. zwei Lampen in unterschiedlichen Räumen), die Vektorsuche
    rankt danach nur noch innerhalb der (kleinen) Kandidatenmenge."""
    attempts = []
    if equipment_tag and location_tag:
        attempts.append({"$and": [{"equipment_tag": equipment_tag}, {"location_tag": location_tag}]})
    if equipment_tag:
        attempts.append({"equipment_tag": equipment_tag})
    if location_tag:
        attempts.append({"location_tag": location_tag})
    attempts.append(None)  # letzter Versuch: ungefiltert, wie in v3

    for where in attempts:
        kwargs = {"query_texts": [normalized_query], "n_results": n_results}
        if where:
            kwargs["where"] = where
        results = collection.query(**kwargs)
        if results["ids"] and len(results["ids"][0]) > 0:
            return results, where
    return None, None


# =====================================================================
# 10. HAUPT-ENDPUNKT
# =====================================================================
CLARIFY_DISTANCE_MARGIN = 0.08  # wie nah Platz 1 und 2 beieinander liegen dürfen


def resolve_single_query(user_query: str, last_item_hint: Optional[str]) -> str:
    normalized_query = normalize_text(user_query)
    query_lower = user_query.lower()

    equipment_tag = resolve_equipment(normalized_query)
    location_tag = resolve_location(normalized_query)
    action = resolve_action(query_lower)

    is_short_followup = len(query_lower.split()) <= 4 and action and not location_tag and not equipment_tag

    if is_short_followup and last_item_hint:
        best_item_name = last_item_hint
        best_item_meta = get_item_meta(best_item_name)
        if not best_item_meta:
            return "Ich konnte das zuvor verwendete Gerät nicht mehr finden. Bitte nenne Gerät oder Raum erneut."
        log.info(f"[MIDDLEWARE] Folge-Befehl erkannt. Nutze Kontext-Item: {best_item_name}")
    else:
        results, used_filter = query_candidates(normalized_query, equipment_tag, location_tag)
        if not results:
            return "Ich konnte kein passendes Gerät finden. Nenne mir gerne Gerätetyp und/oder Raum genauer."

        ids0 = results["ids"][0]
        metas0 = results["metadatas"][0]
        dists0 = results.get("distances", [[None]])[0]

        # Klärungsdialog bei Mehrdeutigkeit (Confidence-Score-Idee aus deinen
        # Recherchen: statt blind zu raten, lieber kurz nachfragen -- v.a.
        # wichtig bei ausführenden Befehlen, nicht nur bei Status-Abfragen).
        if len(ids0) > 1 and dists0[0] is not None and dists0[1] is not None:
            if (dists0[1] - dists0[0]) < CLARIFY_DISTANCE_MARGIN and used_filter is None:
                options = ", ".join(f"`{m['label']}`" for m in metas0[:3])
                return f"🤔 Das ist mehrdeutig, ich habe mehrere passende Geräte gefunden: {options}. Welches meinst du genau (Raum oder genauer Name hilft)?"

        best_item_name = ids0[0]
        best_item_meta = metas0[0]

    best_item_label = best_item_meta["label"]
    item_type = best_item_meta.get("type", "")
    log.info(f"[MIDDLEWARE] Gewähltes Item: {best_item_name} ({best_item_label}) [Typ: {item_type}]")

    # Schutz vor Audio-/Medien-Fehltriggern bleibt als zusätzliches Netz erhalten
    if best_item_meta.get("equipment_tag") in ("Speaker", "MediaPlayer", "Television") or \
       any(w in best_item_name.lower() for w in ["audio", "medialib", "play"]):
        if not any(w in query_lower for w in ["musik", "spiel", "song", "album", "interpret", "lautstaerke", "lautstärke", "sender", "kanal"]):
            return f"⚠️ Suchkonflikt: Ich habe das Medien-Item `{best_item_name}` gefunden, deine Frage bezog sich aber vermutlich nicht auf Medien. Bitte präzisiere."

    if action == "STATUS":
        state = get_openhab_state(best_item_name)
        if state.upper() == "ON" and "an" in query_lower:
            return f"💡 **Ja**, `{best_item_label}` ist aktuell **an** (ON)."
        if state.upper() == "OFF" and "aus" in query_lower:
            return f"💡 **Ja**, `{best_item_label}` ist aktuell **aus** (OFF)."
        return f"💡 **Status-Abfrage:**\n• **Item:** `{best_item_label}` (`{best_item_name}`)\n• **Aktueller Zustand:** **{state}**"

    if action == "OFF":
        cmd = "OFF" if item_type in ["Switch", "Dimmer"] else "DOWN"
        status = execute_openhab_command(best_item_name, cmd)
        return f"🛑 **Befehl gesendet:**\n• **Ziel:** `{best_item_label}` (`{best_item_name}`)\n• **Kommando:** `{cmd}`\n• **Ergebnis:** {status}"

    if action == "ON":
        cmd = "ON" if item_type in ["Switch", "Dimmer"] else "UP"
        status = execute_openhab_command(best_item_name, cmd)
        return f"🟢 **Befehl gesendet:**\n• **Ziel:** `{best_item_label}` (`{best_item_name}`)\n• **Kommando:** `{cmd}`\n• **Ergebnis:** {status}"

    numeric_value = extract_numeric_command(user_query)
    if numeric_value is not None:
        status = execute_openhab_command(best_item_name, numeric_value)
        return f"🔢 **Wert geändert:**\n• **Ziel:** `{best_item_label}` (`{best_item_name}`)\n• **Neuer Wert:** `{numeric_value}`\n• **Ergebnis:** {status}"

    return f"🔍 Ich habe das Item `{best_item_label}` (`{best_item_name}`) gefunden. Sag mir gerne, ob ich es schalten oder abfragen soll."


# Mehrfach-Befehle trennen ("Schalte Licht und Heizung an"). Bewusst nur auf
# " und " gesplittet (nicht Komma), um Dezimalwerte wie "21,5 Grad" nicht zu zerreißen.
SPLIT_PATTERN = re.compile(r"\s+und\s+", re.IGNORECASE)


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest, api_key: str = Security(api_key_header)):
    if api_key and api_key.replace("Bearer ", "") != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API Key")

    user_query = request.messages[-1].content
    last_item_hint = find_last_used_item(request.messages)

    clauses = [c.strip() for c in SPLIT_PATTERN.split(user_query) if c.strip()]
    if len(clauses) > 1:
        answers = [resolve_single_query(clause, last_item_hint) for clause in clauses]
        combined = "\n\n---\n\n".join(answers)
        return generate_openai_response(combined)

    answer = resolve_single_query(user_query, last_item_hint)
    return generate_openai_response(answer)


def generate_openai_response(text: str) -> Dict[str, Any]:
    return {
        "id": f"chatcmpl-{uuid.uuid4()}",
        "object": "chat.completion",
        "created": 1710000000,
        "model": "oh-hybrid-local",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": text},
            "finish_reason": "stop",
        }],
    }


@app.get("/v1/models")
async def list_models():
    return {"object": "list", "data": [{"id": "oh-hybrid-local", "object": "model", "owned_by": "local"}]}


@app.on_event("startup")
def on_startup():
    # Beim Start einmal versuchen, das Semantic Model zu laden, damit die
    # Bridge auch ohne manuellen /api/sync-Call sofort Synonyme kennt
    # (der Sync selbst -- also das Neu-Einlesen der Items -- bleibt separat).
    if fetch_openhab_tag_registry():
        build_synonym_tables_from_openhab()
    else:
        build_synonym_tables_from_openhab()  # baut wenigstens aus YAML auf
