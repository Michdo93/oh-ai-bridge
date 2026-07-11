import os
import re
import uuid
import logging
from typing import List, Dict, Any, Optional, Set, Tuple

import requests
import yaml
from fastapi import FastAPI, HTTPException, Security
from fastapi.security.api_key import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
import chromadb
from chromadb.utils import embedding_functions
from rapidfuzz import fuzz

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

app = FastAPI(title="openHAB Semantic Hybrid Bridge v5")
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


# --- HELFER: UMLAUT-BEREINIGUNG ---
def normalize_text(text: str) -> str:
    """ Konvertiert deutsche Umlaute in ASCII-Schreibweise (ä -> ae, etc.) """
    text = (text or "").lower().strip()
    replacements = {'ä': 'ae', 'ö': 'oe', 'ü': 'ue', 'ß': 'ss'}
    for umlaut, rep in replacements.items():
        text = text.replace(umlaut, rep)
    return text


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

# Kanonischer Tag-Name -> Set von Wörtern (z.B. "Lightbulb" -> {"licht","lampe",...})
# Enthält NACH dem Sync sowohl Equipment- als auch Property-Tags (siehe unten,
# Grund: viele reale Setups taggen nur den Point mit einer Property wie "Light",
# ohne ein separates Equipment "Lightbulb" zu modellieren -- dein
# iKueche_Hue_Lampen_Schalter ist genau so ein Fall).
DEVICE_SYNONYMS: Dict[str, Set[str]] = {}
LOCATION_SYNONYMS: Dict[str, Set[str]] = {}


def _merge_synonym_dict(base_cfg: Dict[str, List[str]], target: Dict[str, Set[str]]):
    for tag, words in base_cfg.items():
        target.setdefault(tag, set())
        for w in words:
            target[tag].add(normalize_text(w))


# --- ACTION-SYNONYME (rein YAML, kein Teil des openHAB Semantic Models) ---
ACTION_WORDS: Dict[str, Set[str]] = {}


def build_action_table():
    ACTION_WORDS.clear()
    for action, words in SYN_CONFIG.get("actions", {}).items():
        ACTION_WORDS[action] = {normalize_text(w) for w in words}


build_action_table()


# =====================================================================
# 2. OPENHAB SEMANTIC MODEL (dieselbe Quelle wie HABot)
# =====================================================================
# Zwei unabhängige Signalquellen, die wir BEIDE nutzen (nicht nur eine!),
# weil reale openHAB-Installationen selten vollständig durchmodelliert sind:
#
#   A) Die "semantics"-Metadata pro Item (value="Point_Control",
#      config.relatesTo="Property_Light", config.isPointOf="gKueche_Lampen").
#      Das ist exakt das, was openHAB selbst berechnet -- zuverlässiger als
#      reine Tag-Namen, aber nur vorhanden wenn ?metadata=... angefragt wird.
#   B) Der reine Text (Name/Label/Gruppen-Namen) des Items, z.B.
#      "iKueche_Hue_Lampen_Schalter" enthält "kueche" und "lampen" wörtlich.
#      Das ist die übliche openHAB-Namenskonvention und ein sehr robustes,
#      von der Modellierungs-Vollständigkeit unabhängiges Signal.
#
# Wenn (A) nichts liefert, greift (B). Wenn beide nichts liefern, ist das
# ein echtes "ich kenne dieses Wort nicht"-Fall -- dann (und nur dann) darf
# auf die freie Vektorsuche zurückgefallen werden.

_TAG_PARENT: Dict[str, Optional[str]] = {}
_TAG_SYNONYMS_DE: Dict[str, List[str]] = {}
_TAG_CATEGORY_CACHE: Dict[str, Optional[str]] = {}
ROOT_CATEGORIES = {"Location", "Equipment", "Point", "Property"}


def fetch_openhab_tag_registry() -> bool:
    """Lädt /rest/tags (deutsche Synonyme via Accept-Language). Nur genutzt
    um rohe Item-Tags (Fallback B ohne semantics-Metadata) zu kategorisieren."""
    global _TAG_PARENT, _TAG_SYNONYMS_DE, _TAG_CATEGORY_CACHE
    _TAG_PARENT = {}
    _TAG_SYNONYMS_DE = {}
    _TAG_CATEGORY_CACHE = {}
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
                syns = list(synonyms_raw)
            else:
                syns = []
            label = t.get("label")
            if label:
                syns.append(label)
            syns.append(uid.split("_")[-1])
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
    """Baut DEVICE_SYNONYMS (Equipment UND Property!) und LOCATION_SYNONYMS
    aus der geladenen Tag-Hierarchie, merged mit der YAML-Fallback-Konfiguration."""
    global DEVICE_SYNONYMS, LOCATION_SYNONYMS
    DEVICE_SYNONYMS = {}
    LOCATION_SYNONYMS = {}

    for uid, syns in _TAG_SYNONYMS_DE.items():
        cat = tag_category(uid)
        short = uid.split("_")[-1]
        words = {normalize_text(s) for s in syns if s}
        if cat in ("Equipment", "Property"):
            DEVICE_SYNONYMS.setdefault(short, set()).update(words)
        elif cat == "Location":
            LOCATION_SYNONYMS.setdefault(short, set()).update(words)

    _merge_synonym_dict(SYN_CONFIG.get("equipment", {}), DEVICE_SYNONYMS)
    _merge_synonym_dict(SYN_CONFIG.get("location", {}), LOCATION_SYNONYMS)

    log.info(f"[SEMANTIC] {len(DEVICE_SYNONYMS)} Geräte-/Property-Klassen, "
             f"{len(LOCATION_SYNONYMS)} Location-Klassen im Vokabular.")


def resolve_tags(normalized_query: str, synonym_table: Dict[str, Set[str]]) -> Set[str]:
    """Gibt ALLE kanonischen Tags zurück, deren Synonym-Wort in der Anfrage
    vorkommt (Substring ODER Fuzzy je Token). Bewusst eine MENGE statt eines
    einzelnen Tags: 'licht' kann z.B. sowohl als Equipment 'Lightbulb' als
    auch als Property 'Light' modelliert sein -- beide sollen zählen."""
    matched: Set[str] = set()
    tokens = normalized_query.split()
    for tag, words in synonym_table.items():
        for w in words:
            if not w:
                continue
            if w in normalized_query:
                matched.add(tag)
                break
            for token in tokens:
                if len(token) >= 3 and fuzz.WRatio(token, w) >= FUZZY_MIN_SCORE:
                    matched.add(tag)
                    break
            else:
                continue
            break
    return matched


def resolve_action(query_lower: str) -> Optional[str]:
    # 0. "Welche Geräte/Lampen/Items gibt es...": Aufzählung statt Einzel-Item.
    #    Muss VOR dem STATUS-Check laufen ("gibt es" würde sonst als STATUS-
    #    Präfix erkannt werden).
    if query_lower.strip().startswith("welche") or any(
        p in query_lower for p in ["was gibt es", "was kannst du steuern", "welche gibt es"]
    ):
        return "LIST"

    # 1. Direkte openHAB-Kommandos als eigenständige Wörter erkennen (Power-
    #    User-Syntax: "Sende ON an X", "schalten ON", "Command OFF"). Das ist
    #    KEIN Teil der deutschen Synonym-Liste, weil ON/OFF/UP/DOWN feste,
    #    sprachunabhängige openHAB-Befehlsnamen sind, keine Umgangssprache.
    tokens = re.findall(r"[a-z]+", query_lower)
    token_set = set(tokens)
    direct = {
        "on": "ON", "off": "OFF", "up": "UP", "down": "DOWN",
    }
    for word, mapped in direct.items():
        if word in token_set:
            return mapped

    # 2. Fragen ("ist ... an?") haben Vorrang vor reinen An/Aus-Wörtern.
    if query_lower.strip().startswith(("sind ", "ist ", "hat ", "läuft ", "laeuft ", "gibt ", "wie ", "was ")):
        return "STATUS"

    for action in ["STATUS", "OFF", "ON", "UP", "DOWN"]:
        words = ACTION_WORDS.get(action, set())
        if any(w in query_lower for w in words if w):
            return action
    return None


# =====================================================================
# 3. REQUEST-MODELLE (OpenAI-kompatibel)
# =====================================================================
class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.7


# =====================================================================
# 4. SEMANTISCHE KLASSIFIZIERUNG EINES ITEMS
# =====================================================================
def _classify_value(value: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    if not value or "_" not in value:
        return None, None
    cat, _, tag = value.partition("_")
    return cat, tag


def resolve_item_semantics(item: dict, items_by_name: Dict[str, dict], max_depth: int = 4) -> Dict[str, Optional[str]]:
    """Ermittelt location_tag / equipment_tag / property_tag für ein Item.

    Quelle A (bevorzugt): die 'semantics'-Metadata (value + config.relatesTo /
    isPointOf / isPartOf), also exakt das, was openHAB selbst berechnet.
    Läuft dazu die isPointOf/isPartOf-Kette nach oben (Point -> Equipment ->
    Location), analog zu HABots eigener Auflösung ('checks ancestor groups
    one level at a time').

    Quelle B (Fallback je Ebene): rohe Tags + /rest/tags-Kategorisierung,
    für Items/Gruppen ohne semantics-Metadata.
    """
    location_tag = equipment_tag = property_tag = None
    visited = set()
    current = item
    depth = 0

    while current and depth < max_depth:
        depth += 1
        name = current.get("name")
        if not name or name in visited:
            break
        visited.add(name)

        meta = current.get("metadata", {}) or {}
        sem = meta.get("semantics")

        if sem:
            cat, tag = _classify_value(sem.get("value"))
            if cat == "Location" and not location_tag:
                location_tag = tag
            elif cat == "Equipment" and not equipment_tag:
                equipment_tag = tag
            elif cat == "Property" and not property_tag:
                property_tag = tag

            config = sem.get("config", {}) or {}
            relates_to = config.get("relatesTo")
            if relates_to:
                r_cat, r_tag = _classify_value(relates_to)
                if r_cat == "Property" and not property_tag:
                    property_tag = r_tag
                elif r_cat == "Equipment" and not equipment_tag:
                    equipment_tag = r_tag
                elif r_cat == "Location" and not location_tag:
                    location_tag = r_tag

            if location_tag and equipment_tag:
                break

            next_name = config.get("isPointOf") or config.get("isPartOf")
            if next_name and next_name in items_by_name:
                current = items_by_name[next_name]
                continue

        # Fallback B: rohe Tags dieses Items/dieser Gruppe
        for t in current.get("tags", []):
            cat = tag_category(t)
            short = t.split("_")[-1] if "_" in t else t
            if cat == "Location" and not location_tag:
                location_tag = short
            elif cat == "Equipment" and not equipment_tag:
                equipment_tag = short
            elif cat == "Property" and not property_tag:
                property_tag = short

        if location_tag and equipment_tag:
            break

        # eine Ebene über groupNames weiterlaufen
        next_group = None
        for g in current.get("groupNames", []):
            if g not in visited and g in items_by_name:
                next_group = g
                break
        if next_group:
            current = items_by_name[next_group]
        else:
            break

    return {"location_tag": location_tag, "equipment_tag": equipment_tag, "property_tag": property_tag}


# =====================================================================
# 5. SYNC: Items von openHAB holen (inkl. semantics-Metadata!) -> Chroma
# =====================================================================
@app.post("/api/sync")
def sync_items_to_vector_db():
    tag_registry_ok = fetch_openhab_tag_registry()
    build_synonym_tables_from_openhab()

    try:
        log.info("[SYNC] Rufe Items von openHAB ab (inkl. semantics-Metadata)...")
        # WICHTIG: ohne ?metadata=.+ liefert /rest/items KEIN Metadata-Feld
        # (im Unterschied zum Einzel-Item-GET, das es implizit mitliefert).
        response = requests.get(
            f"{OPENHAB_URL}/rest/items",
            headers=HEADERS_JSON,
            params={"metadata": ".+"},
            timeout=180,  # bei mehreren Zehntausend Items (s. items_total) reichen 60s oft nicht
        )
        response.raise_for_status()
        items = response.json()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Fehler beim openHAB-Verbindungsaufbau: {str(e)}")

    items_by_name = {i["name"]: i for i in items}

    ids, documents, metadatas = [], [], []
    log.info(f"[SYNC] Analysiere {len(items)} Items (Tag-Registry: {'live' if tag_registry_ok else 'YAML-Fallback'})...")

    for item in items:
        name = item.get("name", "")
        label = item.get("label", "") or name
        type_ = item.get("type", "")
        tags = item.get("tags", [])
        group_names = item.get("groupNames", [])

        sem = resolve_item_semantics(item, items_by_name)

        openbrain_hint = openbrain_role = ""
        meta = item.get("metadata", {}) or {}
        if "openbrain" in meta:
            ob = meta["openbrain"].get("value", "")
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
        if sem["location_tag"]:
            semantic_text += f" raum: {sem['location_tag'].lower()}."
        if sem["equipment_tag"]:
            semantic_text += f" geraetetyp: {sem['equipment_tag'].lower()}."
        if sem["property_tag"]:
            semantic_text += f" eigenschaft: {sem['property_tag'].lower()}."
        if openbrain_hint:
            semantic_text += f" info: {openbrain_hint.lower()}."
        if openbrain_role:
            semantic_text += f" rolle: {openbrain_role.lower()}."

        # match_text: normalisierter Volltext für den harten Substring-Filter
        # (Signalquelle B) -- unabhängig davon, ob die semantische
        # Klassifizierung (Signalquelle A) funktioniert hat.
        match_text = normalize_text(f"{name} {label} {tags_str} {groups_str}")
        # match_tokens: Wort-Grenzen-sichere Tokens (per Regex getrennt).
        # WICHTIG: verhindert False Positives durch zufällige Substrings in
        # zusammengeschriebenen Titeln/Namen, z.B. steckt "licht" rein
        # zeichenweise in "Herzlichtutmichverlangen" (Bach-Kantatentitel ohne
        # Leerzeichen) -- als eigenständiges TOKEN kommt "licht" darin aber
        # nicht vor, nur als Substring mitten in einem viel längeren Token.
        match_tokens = " ".join(re.findall(r"[a-z0-9]+", match_text))

        ids.append(name)
        documents.append(semantic_text)
        metadatas.append({
            "name": name,
            "type": type_,
            "label": label,
            "location_tag": sem["location_tag"] or "",
            "equipment_tag": sem["equipment_tag"] or "",
            "property_tag": sem["property_tag"] or "",
            "match_text": match_text,
            "match_tokens": match_tokens,
        })

    batch_size = 5000
    log.info("[SYNC] Schreibe Daten in die lokale Vektordatenbank...")
    for i in range(0, len(ids), batch_size):
        collection.upsert(
            ids=ids[i:i + batch_size],
            documents=documents[i:i + batch_size],
            metadatas=metadatas[i:i + batch_size],
        )

    # WICHTIG: Cache direkt aus den bereits im Speicher vorhandenen Sync-
    # Daten aufbauen, statt sie per collection.get() aus Chroma erneut zu
    # holen. Ein bare collection.get() über sehr viele Items (hier: 47.780)
    # kann in Chroma an interne Limits stoßen ("Error executing plan").
    # Der teure, paginierte collection.get()-Weg (rebuild_item_cache) bleibt
    # als Fallback für den Server-Neustart erhalten, wo kein In-Memory-Stand
    # existiert.
    set_item_cache(metadatas)
    classified = sum(1 for m in metadatas if m["location_tag"] or m["equipment_tag"] or m["property_tag"])
    log.info("[SYNC] Synchronisierung abgeschlossen.")
    return {
        "status": "success",
        "message": f"{len(ids)} Items synchronisiert.",
        "tag_registry_source": "openhab_live" if tag_registry_ok else "yaml_fallback",
        "items_with_semantic_classification": classified,
        "items_total": len(ids),
    }


@app.get("/api/tags")
def debug_tag_registry():
    return {
        "device_classes": {k: sorted(v) for k, v in DEVICE_SYNONYMS.items()},
        "location_classes": {k: sorted(v) for k, v in LOCATION_SYNONYMS.items()},
        "raw_tag_count": len(_TAG_PARENT),
    }


@app.get("/api/items/{item_name}")
def debug_item(item_name: str):
    """Zeigt, wie die Bridge ein konkretes Item klassifiziert hat -- zum
    Nachvollziehen, z.B. curl http://127.0.0.1:8000/api/items/iKueche_Hue_Lampen_Schalter"""
    try:
        result = collection.get(ids=[item_name])
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))
    if not result or not result.get("ids"):
        raise HTTPException(status_code=404, detail="Item nicht in der Vektordatenbank. Erst /api/sync ausführen.")
    return {"metadata": result["metadatas"][0], "document": result["documents"][0]}


@app.get("/health")
def health():
    return {"status": "ok", "items_cached": len(ITEM_CACHE)}


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
# 7. IN-MEMORY ITEM-CACHE (deterministische Python-Filterung statt
#    Chroma-`where`-Klauseln -- klein, schnell, leicht nachvollziehbar für
#    die typische Item-Anzahl einer Smart-Home-Installation)
# =====================================================================
ITEM_CACHE: List[Dict[str, Any]] = []
ITEM_NAMES: Set[str] = set()
ITEM_BY_NAME: Dict[str, Dict[str, Any]] = {}

# Chroma's bare collection.get() can fail on very large collections ("Error
# executing plan: Internal error") -- fetch in bounded pages instead.
CACHE_PAGE_SIZE = 2000


def set_item_cache(metadatas: List[Dict[str, Any]]):
    """Baut den In-Memory-Cache direkt aus bereits geladenen Metadaten auf
    (z.B. unmittelbar nach einem Sync) -- ohne erneuten Chroma-Request."""
    global ITEM_CACHE, ITEM_NAMES, ITEM_BY_NAME
    ITEM_CACHE = [dict(m) for m in metadatas]
    ITEM_NAMES = {m["name"] for m in ITEM_CACHE}
    ITEM_BY_NAME = {m["name"]: m for m in ITEM_CACHE}
    log.info(f"[CACHE] {len(ITEM_CACHE)} Items im Speicher-Cache (direkt aus Sync).")


def rebuild_item_cache():
    """Lädt den Cache paginiert aus Chroma -- nötig nach einem Server-
    Neustart, wenn kein In-Memory-Stand aus einem gerade gelaufenen Sync
    existiert (Chroma selbst ist persistent, der Python-Cache nicht)."""
    global ITEM_CACHE, ITEM_NAMES, ITEM_BY_NAME
    items: List[Dict[str, Any]] = []
    try:
        total = collection.count()
        offset = 0
        while offset < total:
            page = collection.get(include=["metadatas"], limit=CACHE_PAGE_SIZE, offset=offset)
            page_ids = page.get("ids", [])
            page_metas = page.get("metadatas", [])
            if not page_ids:
                break
            items.extend(dict(m) for m in page_metas)
            offset += len(page_ids)
        ITEM_CACHE = items
        ITEM_NAMES = {m["name"] for m in ITEM_CACHE}
        ITEM_BY_NAME = {m["name"]: m for m in ITEM_CACHE}
        log.info(f"[CACHE] {len(ITEM_CACHE)} Items paginiert aus Chroma geladen.")
    except Exception as e:
        log.warning(f"[CACHE] Konnte Item-Cache nicht laden ({e}). Erst /api/sync ausführen.")
        ITEM_CACHE = []
        ITEM_NAMES = set()
        ITEM_BY_NAME = {}


ACTIONABLE_TYPES_ON_OFF = {"Switch", "Dimmer", "Color", "Rollershutter"}


def _token_hits(tokens_str: str, words: Set[str]) -> bool:
    """Wort-Grenzen-sicherer Treffer: exaktes Token, Token-Präfix (Plural wie
    'lampen' vs. 'lampe'), oder Fuzzy nur zwischen ähnlich LANGEN Tokens.
    Die Längen-Bremse verhindert genau den Bach-Kantaten-Fall: ein 25 Zeichen
    langes Token wird nicht mehr fälschlich gegen ein 5-Zeichen-Wort wie
    'licht' gematcht, egal was rapidfuzz sonst an Teilstring-Heuristiken macht."""
    for token in tokens_str.split():
        for w in words:
            if not w:
                continue
            if token == w:
                return True
            if len(w) >= 4 and token.startswith(w):
                return True
            if abs(len(token) - len(w)) <= 2 and fuzz.ratio(token, w) >= FUZZY_MIN_SCORE:
                return True
    return False


def filter_candidates(
    device_tags: Set[str],
    location_tags: Set[str],
    action: Optional[str],
) -> List[Dict[str, Any]]:
    """Harter Python-Filter über den Item-Cache.

    WICHTIG (Fix ggü. der Vorversion): Signalquelle A (aufgelöste Semantic-
    Tags) hat Vorrang vor Signalquelle B (Text-Tokens). Hat ein Item bereits
    eine bekannte Klassifizierung für eine Dimension, wird NUR diese geprüft
    -- der Text-Fallback greift ausschließlich für Items OHNE jede
    Klassifizierung. Sonst würde ein Item, das korrekt als 'Speaker'
    klassifiziert ist, trotzdem durchrutschen, nur weil sein Name zufällig
    auch noch "licht" als Teilstring enthält.
    """

    def matches_device(it: Dict[str, Any]) -> bool:
        if not device_tags:
            return True
        has_semantic = bool(it.get("equipment_tag") or it.get("property_tag"))
        if has_semantic:
            return it.get("equipment_tag") in device_tags or it.get("property_tag") in device_tags
        words = set()
        for tag in device_tags:
            words |= DEVICE_SYNONYMS.get(tag, set())
        return _token_hits(it.get("match_tokens", ""), words)

    def matches_location(it: Dict[str, Any]) -> bool:
        if not location_tags:
            return True
        has_semantic = bool(it.get("location_tag"))
        if has_semantic:
            return it.get("location_tag") in location_tags
        words = set()
        for tag in location_tags:
            words |= LOCATION_SYNONYMS.get(tag, set())
        return _token_hits(it.get("match_tokens", ""), words)

    candidates = [it for it in ITEM_CACHE if matches_device(it) and matches_location(it)]
    candidates = narrow_by_type(candidates, action)
    return candidates


def narrow_by_type(candidates: List[Dict[str, Any]], action: Optional[str]) -> List[Dict[str, Any]]:
    """Blendet reine Hilfs-Punkte aus (Transition-Zeiten, Schedule-Strings,
    RGB-Text-Spiegel, Alarm-Modus...), die technisch zum selben Gerät gehören
    und daher dieselben Tags/Substrings mitbringen, aber nie das sind, was
    mit 'ist das Licht an' oder 'schalte an' gemeint ist."""
    if len(candidates) <= 1:
        return candidates

    non_string = [c for c in candidates if c.get("type") != "String"]
    if non_string:
        candidates = non_string
    if len(candidates) <= 1:
        return candidates

    if action in ("ON", "OFF"):
        typed = [c for c in candidates if c.get("type") in ACTIONABLE_TYPES_ON_OFF]
        if typed:
            candidates = typed
    elif action == "STATUS":
        switches = [c for c in candidates if c.get("type") == "Switch"]
        if switches:
            candidates = switches

    return candidates


INDEX_PATTERN_CACHE: Dict[str, re.Pattern] = {}


def extract_item_index(tokens_str: str, words: Set[str]) -> Optional[int]:
    """Findet eine Geräte-Nummer wie 'lampe1', 'lampe 2', 'steckdose3' als
    eigenes Token oder Token+Zahl-Kombination. Nutzt für dasselbe Wortset
    kompilierte Regexe wieder (Performance bei 47.780 Items)."""
    key = "|".join(sorted(words))
    pattern = INDEX_PATTERN_CACHE.get(key)
    if pattern is None:
        alts = "|".join(re.escape(w) for w in words if w)
        if not alts:
            return None
        pattern = re.compile(rf"\b(?:{alts})[_ ]?(\d+)\b")
        INDEX_PATTERN_CACHE[key] = pattern
    m = pattern.search(tokens_str)
    return int(m.group(1)) if m else None


def narrow_by_index(candidates: List[Dict[str, Any]], normalized_query: str, device_tags: Set[str]) -> List[Dict[str, Any]]:
    """Unterscheidet z.B. 'iBad_Hue_Lampen_Schalter' (ALLE Lampen im Raum)
    von 'iBad_Hue_Lampe1_Schalter' (EINE bestimmte Lampe):
    - Nennt die Anfrage eine Nummer ("Lampe 2", "Lampe2") -> nur Items mit
      genau dieser Nummer im Namen behalten.
    - Nennt die Anfrage KEINE Nummer -> das kollektive (nicht-nummerierte)
      Item bevorzugen, wenn eines existiert (das ist die naheliegendste
      Deutung von 'das Licht im Bad einschalten')."""
    if len(candidates) <= 1 or not device_tags:
        return candidates

    words: Set[str] = set()
    for tag in device_tags:
        words |= DEVICE_SYNONYMS.get(tag, set())
    if not words:
        return candidates

    query_index = extract_item_index(normalized_query, words)

    if query_index is not None:
        narrowed = [c for c in candidates if extract_item_index(c.get("match_tokens", ""), words) == query_index]
        return narrowed or candidates

    collective = [c for c in candidates if extract_item_index(c.get("match_tokens", ""), words) is None]
    return collective or candidates


# =====================================================================
# 8. KONTEXT-GEDÄCHTNIS AUS DER CONVERSATION (statt globaler Variable)
# =====================================================================
ITEM_REF_PATTERN = re.compile(r"`([A-Za-z0-9_]+)`")


def find_last_used_item(messages: List[ChatMessage]) -> Optional[str]:
    for msg in reversed(messages[:-1]):
        if msg.role != "assistant":
            continue
        for candidate in ITEM_REF_PATTERN.findall(msg.content):
            if candidate in ITEM_NAMES:
                return candidate
    return None


def find_exact_item_reference(user_query: str) -> Optional[str]:
    """Power-User-Shortcut: wird der exakte Item-Name im Text genannt
    (z.B. 'Sende ON an iKueche_Hue_Lampen_Schalter'), diesen direkt nutzen
    und die ganze Synonym-/Fuzzy-Pipeline überspringen."""
    for token in re.findall(r"[A-Za-z0-9_]+", user_query):
        if token in ITEM_NAMES:
            return token
    return None


def get_item_meta(item_name: str) -> Optional[Dict[str, Any]]:
    return ITEM_BY_NAME.get(item_name)


# =====================================================================
# 9. NUMERISCHE WERTE (Prozent / Grad) SAUBER PARSEN
# =====================================================================
NUMBER_PATTERN = re.compile(r"(\d+(?:[.,]\d+)?)\s*(%|prozent|grad|°c?|°)?")


def extract_numeric_command(user_query: str) -> Optional[str]:
    match = NUMBER_PATTERN.search(user_query)
    if not match:
        return None
    value = match.group(1).replace(",", ".")
    if value.endswith(".0"):
        value = value[:-2]
    return value


# =====================================================================
# 10. RANKING INNERHALB DER KANDIDATEN (Embeddings nur als Tiebreaker,
#     NIEMALS mehr als Fallback über den gesamten unfiltierten Bestand,
#     solange mindestens eine Dimension aus der Query aufgelöst wurde)
# =====================================================================
def rank_candidates(normalized_query: str, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Rankt NUR innerhalb der bereits gefilterten (kleinen) Kandidatenliste.
    Fragt Chroma gezielt über 'where name $in [...]' ab, NIEMALS über den
    gesamten Bestand (bei 47.780 Items wäre n_results=len(ITEM_CACHE) eine
    Katastrophe für CPU-Zeit und Speicher)."""
    if len(candidates) <= 1:
        return candidates
    names = [c["name"] for c in candidates]
    try:
        results = collection.query(
            query_texts=[normalized_query],
            n_results=len(names),
            where={"name": {"$in": names}},
        )
        order = {name: idx for idx, name in enumerate(results["ids"][0])}
        ranked = sorted(candidates, key=lambda c: order.get(c["name"], len(order)))
        # Falls Chroma aus irgendeinem Grund weniger/andere IDs zurückgibt
        # als erwartet, nicht blind vertrauen -- nur übernehmen, wenn wirklich
        # alle Kandidaten im Ranking auftauchen.
        if len(order) >= len(candidates):
            return ranked
        return candidates
    except Exception as e:
        log.warning(f"[RANK] Vektor-Ranking fehlgeschlagen ({e}), behalte Cache-Reihenfolge.")
        return candidates


def free_vector_fallback(normalized_query: str, n: int = 5) -> List[Dict[str, Any]]:
    """Letzter Ausweg, wenn aus der Anfrage GAR kein Raum-/Geräte-Wort
    aufgelöst werden konnte: klassische, eng begrenzte Vektorsuche über den
    Gesamtbestand (n klein halten -- bei 47.780 Items keine großen n_results!)."""
    try:
        results = collection.query(query_texts=[normalized_query], n_results=n)
        return [ITEM_BY_NAME[i] for i in results["ids"][0] if i in ITEM_BY_NAME]
    except Exception as e:
        log.warning(f"[RANK] Freie Vektorsuche fehlgeschlagen ({e}).")
        return []


CLARIFY_MAX_OPTIONS = 4


# =====================================================================
# 11. HAUPT-AUFLÖSUNG EINER EINZELNEN ANFRAGE
# =====================================================================
def resolve_single_query(user_query: str, last_item_hint: Optional[str]) -> str:
    normalized_query = normalize_text(user_query)
    query_lower = user_query.lower()

    # Power-User-Shortcut: exakter Item-Name genannt -> direkt nutzen.
    exact_name = find_exact_item_reference(user_query)
    if exact_name:
        best_item_meta = get_item_meta(exact_name)
        return execute_resolved(exact_name, best_item_meta, query_lower, user_query)

    device_tags = resolve_tags(normalized_query, DEVICE_SYNONYMS)
    location_tags = resolve_tags(normalized_query, LOCATION_SYNONYMS)
    action = resolve_action(query_lower)

    is_short_followup = len(query_lower.split()) <= 4 and action and not device_tags and not location_tags

    if is_short_followup and last_item_hint:
        best_item_meta = get_item_meta(last_item_hint)
        if not best_item_meta:
            return "Ich konnte das zuvor verwendete Gerät nicht mehr finden. Bitte nenne Gerät oder Raum erneut."
        log.info(f"[MIDDLEWARE] Folge-Befehl erkannt. Nutze Kontext-Item: {last_item_hint}")
        return execute_resolved(last_item_hint, best_item_meta, query_lower, user_query)

    if not ITEM_CACHE:
        return "Die Item-Datenbank ist leer. Bitte zuerst /api/sync ausführen."

    # KERN-FIX: Wenn wir WIRKLICH kein Raum-/Gerätewort kennen, NIEMALS
    # filter_candidates() mit zwei leeren Mengen aufrufen -- das würde (per
    # Definition "kein Filter gesetzt = alles passt") den KOMPLETTEN
    # 47.780-Item-Bestand als "Kandidaten" zurückgeben, wovon dann nur die
    # ersten paar (in Cache-Reihenfolge bzw. zufälligem Ranking) als
    # "mehrdeutig" angezeigt würden -- genau der Bug, den du gesehen hast.
    if not device_tags and not location_tags:
        candidates = free_vector_fallback(normalized_query, n=3)
        if not candidates:
            return "Ich konnte kein passendes Gerät finden. Nenne mir gerne Gerätetyp und/oder Raum genauer."
        return execute_resolved(candidates[0]["name"], candidates[0], query_lower, user_query)

    candidates = filter_candidates(device_tags, location_tags, action)

    if not candidates:
        # Kein Item erfüllt beide Dimensionen -> eine davon lockern, statt
        # sofort auf den kompletten unfiltierten Bestand zu springen.
        if device_tags and location_tags:
            candidates = filter_candidates(device_tags, set(), action)
        if not candidates and location_tags:
            candidates = filter_candidates(set(), location_tags, action)
        if not candidates and device_tags:
            candidates = filter_candidates(device_tags, set(), action)

    if not candidates:
        return "Ich konnte kein Gerät finden, das zu Raum/Gerätetyp deiner Anfrage passt. Prüfe ggf. mit /api/tags, ob dieser Raum/Gerätetyp bekannt ist."

    # "Welche Lampen/Geräte gibt es im Bad?" -> Aufzählung statt Einzelauswahl.
    if action == "LIST":
        return build_list_response(candidates)

    # Kollektiv- vs. Einzel-Gerät unterscheiden (z.B. 'iBad_Hue_Lampen_Schalter'
    # = alle Lampen vs. 'iBad_Hue_Lampe1_Schalter' = eine bestimmte Lampe).
    candidates = narrow_by_index(candidates, normalized_query, device_tags)

    candidates = rank_candidates(normalized_query, candidates)

    if len(candidates) > 1:
        top, second = candidates[0], candidates[1]
        # Echte Mehrdeutigkeit nur, wenn sich die Kandidaten nicht schon
        # über den Item-Typ unterscheiden.
        if top.get("type") == second.get("type"):
            options = ", ".join(
                f"`{c['name']}`" + (f" ({c['label']})" if c["label"] != c["name"] else "")
                for c in candidates[:CLARIFY_MAX_OPTIONS]
            )
            return f"🤔 Das ist mehrdeutig, ich habe mehrere passende Geräte gefunden: {options}. Sag mir gerne den genauen Namen oder ein unterscheidendes Detail."

    best_item_meta = candidates[0]
    return execute_resolved(best_item_meta["name"], best_item_meta, query_lower, user_query)


def build_list_response(candidates: List[Dict[str, Any]]) -> str:
    """Aufzählung für 'Welche Lampen/Geräte/Items gibt es im Bad?'. Bevorzugt
    Gruppen-Items (repräsentieren ein ganzes Gerät wie 'Lampe1') und
    steuerbare Punkte; blendet reine String-Hilfspunkte aus, wenn genug
    andere Kandidaten da sind."""
    groups = [c for c in candidates if c.get("type") == "Group"]
    controllable = [c for c in candidates if c.get("type") in ("Switch", "Dimmer", "Color", "Rollershutter", "Number", "Contact")]
    shown = groups + [c for c in controllable if c not in groups]
    if not shown:
        shown = candidates

    seen: Set[str] = set()
    lines = []
    for c in shown:
        if c["name"] in seen:
            continue
        seen.add(c["name"])
        label_part = f" ({c['label']})" if c["label"] != c["name"] else ""
        lines.append(f"• `{c['name']}`{label_part}")
        if len(lines) >= 25:
            lines.append("… weitere vorhanden, bitte genauer eingrenzen (z.B. Raum oder Gerätetyp).")
            break

    if not lines:
        return "Ich konnte dazu keine Geräte finden."
    return "Gefundene Geräte:\n" + "\n".join(lines)


def execute_resolved(best_item_name: str, best_item_meta: Dict[str, Any], query_lower: str, user_query: str) -> str:
    best_item_label = best_item_meta["label"]
    item_type = best_item_meta.get("type", "")
    log.info(f"[MIDDLEWARE] Gewähltes Item: {best_item_name} ({best_item_label}) [Typ: {item_type}]")

    if best_item_meta.get("equipment_tag") in ("Speaker", "MediaPlayer", "Television") or \
       any(w in best_item_name.lower() for w in ["audio", "medialib", "play"]):
        if not any(w in query_lower for w in ["musik", "spiel", "song", "album", "interpret", "lautstaerke", "lautstärke", "sender", "kanal"]):
            return f"⚠️ Suchkonflikt: Ich habe das Medien-Item `{best_item_name}` gefunden, deine Frage bezog sich aber vermutlich nicht auf Medien. Bitte präzisiere."

    action = resolve_action(query_lower)

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
    if fetch_openhab_tag_registry():
        build_synonym_tables_from_openhab()
    else:
        build_synonym_tables_from_openhab()
    rebuild_item_cache()