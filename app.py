import os
import re
import uuid
import pickle
import logging
from typing import List, Dict, Any, Optional, Set, Tuple

import yaml
from fastapi import FastAPI, HTTPException, Security
from fastapi.security.api_key import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from rapidfuzz import fuzz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# python-openhab-rest-client (PyPI: python-openhab-rest-client, import name: openhab)
from openhab import OpenHABClient, Items, Tags

# =====================================================================
# 0. KONFIGURATION (.env statt Klartext-Credentials im Code)
# =====================================================================
load_dotenv()

OPENHAB_URL = os.getenv("OPENHAB_URL", "http://192.168.0.10:8080").rstrip("/")
OPENHAB_TOKEN = os.getenv("OPENHAB_TOKEN", "")
API_KEY = os.getenv("API_KEY", "your_local_key")
SYNONYMS_PATH = os.getenv("SYNONYMS_PATH", "./synonyms.yaml")
# Ersetzt den alten CHROMA_PATH: statt einer Vektordatenbank nur noch ein
# einziges Pickle-File mit Item-Metadaten + Dokumenten (s. Abschnitt 6/7).
CACHE_PATH = os.getenv("CACHE_PATH", "./item_cache.pkl")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("oh-ai-bridge")

app = FastAPI(title="openHAB Semantic Hybrid Bridge v6")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
api_key_header = APIKeyHeader(name="Authorization", auto_error=False)

# --- openHAB-Client (python-openhab-rest-client) -----------------------
oh_client = OpenHABClient(url=OPENHAB_URL, token=OPENHAB_TOKEN or None)
items_api = Items(oh_client)
tags_api = Tags(oh_client)


# --- HELFER: UMLAUT-BEREINIGUNG (jetzt defensiv gegen Nicht-Strings) ---
def normalize_text(text: Any) -> str:
    """ Konvertiert deutsche Umlaute in ASCII-Schreibweise (ä -> ae, etc.).
    Nimmt bewusst auch Nicht-Strings entgegen: YAML parst unquotierte
    'on'/'off'/'yes'/'no' als Booleans (YAML-1.1-Altlast) -- ein einzelner
    vergessener Quote in synonyms.yaml darf den ganzen Dienst nicht mehr
    zum Absturz bringen (siehe (str) statt (True or "").lower())."""
    if not isinstance(text, str):
        log.warning(f"[CONFIG] Erwartete Text-Zeichenkette, bekam {type(text).__name__}: {text!r}. "
                    f"Falls das aus synonyms.yaml stammt: 'on'/'off'/'yes'/'no' MÜSSEN in "
                    f"Anführungszeichen stehen, sonst parst YAML sie als Boolean.")
        text = "" if text is None else str(text)
    text = text.lower().strip()
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

DEVICE_SYNONYMS: Dict[str, Set[str]] = {}
LOCATION_SYNONYMS: Dict[str, Set[str]] = {}


def _clean_words(raw_words: List[Any], context: str) -> Set[str]:
    """Filtert nicht-String-Werte defensiv heraus (s. normalize_text-Docstring)
    statt beim ersten falschen YAML-Typ den ganzen Sync/Start abzubrechen."""
    cleaned = set()
    for w in raw_words:
        if not isinstance(w, str):
            log.warning(f"[CONFIG] Ignoriere Nicht-Text-Wert in synonyms.yaml ({context}): {w!r}. "
                        f"Vermutlich fehlen Anführungszeichen um 'on'/'off'/'yes'/'no' o.ä.")
            continue
        cleaned.add(normalize_text(w))
    return cleaned


def _merge_synonym_dict(base_cfg: Dict[str, List[Any]], target: Dict[str, Set[str]]):
    for tag, words in base_cfg.items():
        target.setdefault(tag, set())
        target[tag] |= _clean_words(words, f"equipment/location.{tag}")


ACTION_WORDS: Dict[str, Set[str]] = {}


def build_action_table():
    ACTION_WORDS.clear()
    for action, words in SYN_CONFIG.get("actions", {}).items():
        if not isinstance(action, str):
            log.warning(f"[CONFIG] Ignoriere Nicht-Text-Aktionsschlüssel in synonyms.yaml: {action!r} "
                        f"(vermutlich 'ON'/'OFF' ohne Anführungszeichen -> von YAML als Boolean geparst).")
            continue
        ACTION_WORDS[action] = _clean_words(words, f"actions.{action}")


build_action_table()


# =====================================================================
# 2. OPENHAB SEMANTIC MODEL (dieselbe Quelle wie HABot)
# =====================================================================
_TAG_PARENT: Dict[str, Optional[str]] = {}
_TAG_SYNONYMS_DE: Dict[str, List[str]] = {}
_TAG_CATEGORY_CACHE: Dict[str, Optional[str]] = {}
ROOT_CATEGORIES = {"Location", "Equipment", "Point", "Property"}


def fetch_openhab_tag_registry() -> bool:
    """Lädt die Tag-Hierarchie über Tags.getTags() (python-openhab-rest-client)
    statt eines rohen requests.get auf /rest/tags."""
    global _TAG_PARENT, _TAG_SYNONYMS_DE, _TAG_CATEGORY_CACHE
    _TAG_PARENT = {}
    _TAG_SYNONYMS_DE = {}
    _TAG_CATEGORY_CACHE = {}
    try:
        tags = tags_api.getTags(language="de")
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
        log.info(f"[SEMANTIC] {len(tags)} Tags über Tags.getTags() geladen.")
        return True
    except Exception as e:
        log.warning(f"[SEMANTIC] Tags.getTags() nicht verfügbar ({e}). Nutze YAML-Fallback für Räume/Geräte.")
        return False


def tag_category(uid: Optional[str]) -> Optional[str]:
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
    if query_lower.strip().startswith("welche") or any(
        p in query_lower for p in ["was gibt es", "was kannst du steuern", "welche gibt es"]
    ):
        return "LIST"

    tokens = re.findall(r"[a-z]+", query_lower)
    token_set = set(tokens)
    direct = {"on": "ON", "off": "OFF", "up": "UP", "down": "DOWN"}
    for word, mapped in direct.items():
        if word in token_set:
            return mapped

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
# 5. LEICHTGEWICHTIGER FALLBACK-RANKER (TF-IDF statt Embedding-Modell)
# =====================================================================
# Warum kein sentence-transformers/torch mehr:
# - Ziel-Hardware ist ein Raspberry Pi 3 Model B+ (1.4 GHz Cortex-A53, 1 GB
#   RAM) -- torch + ein Transformer-Modell sind dafür unverhältnismäßig
#   schwer (Speicher, Startzeit, und der Prozess spricht bei jedem Start
#   unauthentifiziert mit huggingface.co, was weder "ressourcenschonend"
#   noch "ohne Cloud" ist).
# - In dieser Architektur übernimmt die Vektorsuche ohnehin nur noch zwei
#   Nebenrollen: (a) Tiebreaker innerhalb einer bereits durch Tags/Text
#   hart gefilterten, kleinen Kandidatenliste, (b) letzter Ausweg, wenn GAR
#   kein Raum-/Gerätewort erkannt wurde. Für beide Fälle reicht ein
#   klassischer TF-IDF-Vektorraum (scikit-learn, reines C/NumPy, keine
#   Modell-Downloads, Sync von 47.780 Items dauert damit Sekundenbruchteile
#   statt eine GPU-Bibliothek zu laden) völlig aus.
_vectorizer: Optional[TfidfVectorizer] = None
_tfidf_matrix = None
_name_to_row: Dict[str, int] = {}


def build_tfidf_index(documents: List[str], names: List[str]):
    global _vectorizer, _tfidf_matrix, _name_to_row
    if not documents:
        _vectorizer = None
        _tfidf_matrix = None
        _name_to_row = {}
        return
    _vectorizer = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=1)
    _tfidf_matrix = _vectorizer.fit_transform(documents)
    _name_to_row = {name: i for i, name in enumerate(names)}
    log.info(f"[TFIDF] Index über {len(documents)} Dokumente gebaut "
             f"({_tfidf_matrix.shape[1]} Terme).")


def rank_candidates(normalized_query: str, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Rankt NUR innerhalb der bereits gefilterten (kleinen) Kandidatenliste."""
    if len(candidates) <= 1 or _vectorizer is None or _tfidf_matrix is None:
        return candidates
    try:
        rows = [_name_to_row[c["name"]] for c in candidates]
    except KeyError:
        return candidates
    sub = _tfidf_matrix[rows]
    qvec = _vectorizer.transform([normalized_query])
    sims = cosine_similarity(qvec, sub)[0]
    ranked = [c for _, c in sorted(zip(sims, candidates), key=lambda p: -p[0])]
    return ranked


def free_vector_fallback(normalized_query: str, n: int = 5) -> List[Dict[str, Any]]:
    """Letzter Ausweg, wenn aus der Anfrage GAR kein Raum-/Geräte-Wort
    aufgelöst werden konnte: TF-IDF-Cosine-Ähnlichkeit über den Gesamtbestand.
    Auf 47.780 Items ein einzelnes Sparse-Matrix-Produkt -- Millisekunden,
    auch auf einem Raspberry Pi 3."""
    if _vectorizer is None or _tfidf_matrix is None:
        return []
    try:
        qvec = _vectorizer.transform([normalized_query])
        sims = cosine_similarity(qvec, _tfidf_matrix)[0]
        top_idx = sims.argsort()[::-1][:n]
        return [ITEM_CACHE[i] for i in top_idx if sims[i] > 0]
    except Exception as e:
        log.warning(f"[TFIDF] Freie Suche fehlgeschlagen ({e}).")
        return []


# =====================================================================
# 6. SYNC: Items von openHAB holen (inkl. semantics-Metadata!) -> Cache
# =====================================================================
@app.post("/api/sync")
def sync_items_to_vector_db():
    tag_registry_ok = fetch_openhab_tag_registry()
    build_synonym_tables_from_openhab()

    try:
        log.info("[SYNC] Rufe Items von openHAB ab (Items.getItems, inkl. semantics-Metadata)...")
        # WICHTIG: Die installierte python-openhab-rest-client-Version heißt
        # die Methode tatsächlich `getItems`, nicht `getAllItems` (wie in der
        # PyPI-/GitHub-Doku beschrieben -- Doku und Code sind hier
        # offenbar auseinandergelaufen). Außerdem wirft die Library bei
        # HTTP-/Verbindungsfehlern KEINE Exception, sondern gibt
        # {"error": "..."} als normalen Rückgabewert zurück.
        items = items_api.getItems(metadata=".+")
        if isinstance(items, dict) and "error" in items:
            raise ValueError(items["error"])
        if not isinstance(items, list):
            raise ValueError(f"Unerwartetes Antwortformat von Items.getItems(): {type(items)}: {items!r}")
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

        match_text = normalize_text(f"{name} {label} {tags_str} {groups_str}")
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

    set_item_cache(metadatas)
    build_tfidf_index(documents, ids)

    try:
        with open(CACHE_PATH, "wb") as f:
            pickle.dump({"metadatas": metadatas, "documents": documents}, f)
        log.info(f"[SYNC] Cache nach {CACHE_PATH} persistiert.")
    except Exception as e:
        log.warning(f"[SYNC] Konnte Cache nicht persistieren ({e}). Nach einem Neustart ist erneut /api/sync nötig.")

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
    meta = ITEM_BY_NAME.get(item_name)
    if not meta:
        raise HTTPException(status_code=404, detail="Item nicht im Cache. Erst /api/sync ausführen.")
    return {"metadata": meta}


@app.get("/health")
def health():
    return {"status": "ok", "items_cached": len(ITEM_CACHE), "tfidf_ready": _vectorizer is not None}


# =====================================================================
# 7. IN-MEMORY ITEM-CACHE
# =====================================================================
ITEM_CACHE: List[Dict[str, Any]] = []
ITEM_NAMES: Set[str] = set()
ITEM_BY_NAME: Dict[str, Dict[str, Any]] = {}


def set_item_cache(metadatas: List[Dict[str, Any]]):
    global ITEM_CACHE, ITEM_NAMES, ITEM_BY_NAME
    ITEM_CACHE = [dict(m) for m in metadatas]
    ITEM_NAMES = {m["name"] for m in ITEM_CACHE}
    ITEM_BY_NAME = {m["name"]: m for m in ITEM_CACHE}
    log.info(f"[CACHE] {len(ITEM_CACHE)} Items im Speicher-Cache.")


def load_cache_from_disk() -> bool:
    """Beim Serverstart: lädt den zuletzt gespeicherten Stand von der
    Festplatte (ein einzelnes Pickle-File statt einer Vektordatenbank),
    damit nach einem Neustart nicht zwingend sofort /api/sync nötig ist."""
    if not os.path.exists(CACHE_PATH):
        log.warning(f"[CACHE] {CACHE_PATH} existiert noch nicht. Bitte /api/sync ausführen.")
        return False
    try:
        with open(CACHE_PATH, "rb") as f:
            data = pickle.load(f)
        metadatas = data.get("metadatas", [])
        documents = data.get("documents", [])
        set_item_cache(metadatas)
        build_tfidf_index(documents, [m["name"] for m in metadatas])
        log.info(f"[CACHE] {len(metadatas)} Items von {CACHE_PATH} geladen.")
        return True
    except Exception as e:
        log.warning(f"[CACHE] Konnte {CACHE_PATH} nicht laden ({e}). Bitte /api/sync ausführen.")
        return False


# =====================================================================
# 8. REST-INTERAKTION MIT OPENHAB (über python-openhab-rest-client)
# =====================================================================
# HINWEIS zur "Test Suite": python-openhab-rest-client bringt ein
# `openhab.tests`-Modul mit (z.B. ItemsTest.testSendCommand) mit. Laut
# eigener Dokumentation des Pakets führt das aber den echten Befehl trotzdem
# aus ("both would also really execute a sendCommand") -- es ist also KEIN
# Dry-Run, sondern nur ein try/except-Wrapper mit Print-Ausgabe um dieselbe
# Aktion. Für "darf dieses Item diesen Befehl überhaupt bekommen" bringt das
# keinen Sicherheitsgewinn gegenüber einem echten Aufruf. Deshalb prüfen wir
# stattdessen LOKAL (ohne Netzwerk-Aufruf, ohne Seiteneffekt), ob der
# Item-Typ den Befehl überhaupt sinnvoll unterstützt, BEVOR wir senden.
VALID_COMMANDS_BY_TYPE: Dict[str, Set[str]] = {
    "Switch": {"ON", "OFF"},
    "Dimmer": {"ON", "OFF", "INCREASE", "DECREASE"},
    "Color": {"ON", "OFF"},
    "Rollershutter": {"UP", "DOWN", "STOP", "MOVE"},
    "Number": set(),  # nur numerische Werte sinnvoll (s. Regex-Zweig oben in command_allowed)
}


def command_allowed(item_type: str, command: str) -> bool:
    """Lokale Plausibilitätsprüfung ohne Netzwerk-Aufruf: verhindert z.B.
    'ON' an ein Number- oder String-Item zu senden. Numerische Befehle
    (Prozent, Grad) sind für Dimmer/Number/Rollershutter/Color immer erlaubt."""
    if re.match(r"^-?\d+(\.\d+)?$", command):
        return item_type in ("Dimmer", "Number", "Rollershutter", "Color") or item_type.startswith("Number:")
    if item_type not in VALID_COMMANDS_BY_TYPE:
        # Unbekannter/nicht gelisteter Typ (String, Contact, DateTime, ...):
        # nicht blockieren, openHAB validiert selbst -- wir wollen nur die
        # eindeutig unsinnigen Fälle lokal abfangen, nicht raten.
        return True
    return command.upper() in VALID_COMMANDS_BY_TYPE[item_type]


def execute_openhab_command(item_name: str, command: str, item_type: str = "") -> str:
    if item_type and not command_allowed(item_type, command):
        return f"Abgelehnt: Befehl '{command}' passt nicht zum Item-Typ '{item_type}'."
    try:
        result = items_api.sendCommand(item_name, command)
    except Exception as e:
        log.warning(f"[OPENHAB] sendCommand({item_name}, {command}) hat eine Exception geworfen: {e}")
        return f"Fehler: {e}"
    # Die Library wirft bei HTTP-/Verbindungsfehlern KEINE Exception, sondern
    # gibt {"error": "..."} zurück (z.B. "Item not found.", "Item command
    # null."). Erfolg kommt als {"message": "OK"}.
    if isinstance(result, dict) and "error" in result:
        log.warning(f"[OPENHAB] sendCommand({item_name}, {command}) fehlgeschlagen: {result['error']}")
        return f"Fehler: {result['error']}"
    return "Erfolgreich geschaltet"


def get_openhab_state(item_name: str) -> str:
    try:
        result = items_api.getItemState(item_name)
    except Exception as e:
        log.warning(f"[OPENHAB] getItemState({item_name}) hat eine Exception geworfen: {e}")
        return "Unbekannt"
    # Erfolg: die Library gibt den rohen State-String direkt zurück (z.B.
    # "ON", "22.5", "NULL"), NICHT in ein dict verpackt. Ein dict deutet auf
    # {"error": "..."} hin (z.B. Item nicht gefunden).
    if isinstance(result, dict):
        if "error" in result:
            log.warning(f"[OPENHAB] getItemState({item_name}) fehlgeschlagen: {result['error']}")
        return "Unbekannt"
    if result is None:
        return "Unbekannt"
    return str(result)


# =====================================================================
# 9. KONTEXT-GEDÄCHTNIS AUS DER CONVERSATION (statt globaler Variable)
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
    for token in re.findall(r"[A-Za-z0-9_]+", user_query):
        if token in ITEM_NAMES:
            return token
    return None


def get_item_meta(item_name: str) -> Optional[Dict[str, Any]]:
    return ITEM_BY_NAME.get(item_name)


# =====================================================================
# 10. NUMERISCHE WERTE (Prozent / Grad) SAUBER PARSEN
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
# 11. KANDIDATEN-FILTERUNG, TYP-/INDEX-EINENGUNG
# =====================================================================
ACTIONABLE_TYPES_ON_OFF = {"Switch", "Dimmer", "Color", "Rollershutter"}


def _token_hits(tokens_str: str, words: Set[str]) -> bool:
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


def filter_candidates(device_tags: Set[str], location_tags: Set[str], action: Optional[str]) -> List[Dict[str, Any]]:
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
    return narrow_by_type(candidates, action)


def narrow_by_type(candidates: List[Dict[str, Any]], action: Optional[str]) -> List[Dict[str, Any]]:
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


CLARIFY_MAX_OPTIONS = 4


# =====================================================================
# 12. HAUPT-AUFLÖSUNG EINER EINZELNEN ANFRAGE
# =====================================================================
def resolve_single_query(user_query: str, last_item_hint: Optional[str]) -> str:
    normalized_query = normalize_text(user_query)
    query_lower = user_query.lower()

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

    if not device_tags and not location_tags:
        candidates = free_vector_fallback(normalized_query, n=3)
        if not candidates:
            return "Ich konnte kein passendes Gerät finden. Nenne mir gerne Gerätetyp und/oder Raum genauer."
        return execute_resolved(candidates[0]["name"], candidates[0], query_lower, user_query)

    candidates = filter_candidates(device_tags, location_tags, action)

    if not candidates:
        if device_tags and location_tags:
            candidates = filter_candidates(device_tags, set(), action)
        if not candidates and location_tags:
            candidates = filter_candidates(set(), location_tags, action)
        if not candidates and device_tags:
            candidates = filter_candidates(device_tags, set(), action)

    if not candidates:
        return "Ich konnte kein Gerät finden, das zu Raum/Gerätetyp deiner Anfrage passt. Prüfe ggf. mit /api/tags, ob dieser Raum/Gerätetyp bekannt ist."

    if action == "LIST":
        return build_list_response(candidates)

    candidates = narrow_by_index(candidates, normalized_query, device_tags)
    candidates = rank_candidates(normalized_query, candidates)

    if len(candidates) > 1:
        top, second = candidates[0], candidates[1]
        if top.get("type") == second.get("type"):
            options = ", ".join(
                f"`{c['name']}`" + (f" ({c['label']})" if c["label"] != c["name"] else "")
                for c in candidates[:CLARIFY_MAX_OPTIONS]
            )
            return f"🤔 Das ist mehrdeutig, ich habe mehrere passende Geräte gefunden: {options}. Sag mir gerne den genauen Namen oder ein unterscheidendes Detail."

    best_item_meta = candidates[0]
    return execute_resolved(best_item_meta["name"], best_item_meta, query_lower, user_query)


def build_list_response(candidates: List[Dict[str, Any]]) -> str:
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


def execute_resolved(best_item_name: str, best_item_meta: Optional[Dict[str, Any]], query_lower: str, user_query: str) -> str:
    if not best_item_meta:
        return f"Ich kenne das Item `{best_item_name}` nicht (nicht im Cache -- ggf. erst /api/sync ausführen)."

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
        status = execute_openhab_command(best_item_name, cmd, item_type)
        return f"🛑 **Befehl gesendet:**\n• **Ziel:** `{best_item_label}` (`{best_item_name}`)\n• **Kommando:** `{cmd}`\n• **Ergebnis:** {status}"

    if action == "ON":
        cmd = "ON" if item_type in ["Switch", "Dimmer"] else "UP"
        status = execute_openhab_command(best_item_name, cmd, item_type)
        return f"🟢 **Befehl gesendet:**\n• **Ziel:** `{best_item_label}` (`{best_item_name}`)\n• **Kommando:** `{cmd}`\n• **Ergebnis:** {status}"

    numeric_value = extract_numeric_command(user_query)
    if numeric_value is not None:
        status = execute_openhab_command(best_item_name, numeric_value, item_type)
        return f"🔢 **Wert geändert:**\n• **Ziel:** `{best_item_label}` (`{best_item_name}`)\n• **Neuer Wert:** `{numeric_value}`\n• **Ergebnis:** {status}"

    return f"🔍 Ich habe das Item `{best_item_label}` (`{best_item_name}`) gefunden. Sag mir gerne, ob ich es schalten oder abfragen soll."


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
    load_cache_from_disk()
    if fetch_openhab_tag_registry():
        build_synonym_tables_from_openhab()
    else:
        build_synonym_tables_from_openhab()