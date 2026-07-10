import os
import re
import uuid
import requests
from fastapi import FastAPI, HTTPException, Security
from fastapi.security.api_key import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.utils import embedding_functions

# --- KONFIGURATION ---
OPENHAB_URL = "http://<YOUR_OPENHAB_IP>:8080"
OPENHAB_TOKEN = "oh.your_actual_token_here"
API_KEY = "your_local_key"

# --- SYSTEM-INITIALISIERUNG ---
app = FastAPI(title="openHAB Semantic Hybrid Bridge v3")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

api_key_header = APIKeyHeader(name="Authorization", auto_error=False)

# ChromaDB Client & Embedding Modell auf der CPU
chroma_client = chromadb.PersistentClient(path="./chroma_db")
emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
collection = chroma_client.get_or_create_collection(name="openhab_items", embedding_function=emb_fn)

# Globaler Kurzzeit-Speicher für das Gedächtnis
LAST_USED_ITEM: Optional[Dict[str, Any]] = None

# --- REQUETS-MODELLE FÜR OPENAI COMPATIBILITY ---
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.7

# --- HELFER: UMLAUT-BEREINIGUNG ---
def normalize_text(text: str) -> str:
    """ Konvertiert deutsche Umlaute in die openHAB-Schreibweise (ä -> ae, etc.) """
    text = text.lower()
    replacements = {'ä': 'ae', 'ö': 'oe', 'ü': 'ue', 'ß': 'ss'}
    for umlaut, rep in replacements.items():
        text = text.replace(umlaut, rep)
    return text

# Map für vordefinierte Räume zur harten Filterung (Erweiterbar)
ROOM_KEYWORDS = ["bad", "kueche", "wohnzimmer", "schlafzimmer", "flur", "buero", "garten", "keller", "multimedia"]


# --- 1. OPTIMIERTER SYNC-MECHANISMUS (Upsert-Verfahren) ---
@app.post("/api/sync")
def sync_items_to_vector_db():
    """ Holt alle Items von openHAB und führt ein effizientes Upsert in ChromaDB durch """
    headers = {"Authorization": f"Bearer {OPENHAB_TOKEN}", "Accept": "application/json"}
    try:
        print("[SYNC] Rufe Items von openHAB ab...")
        response = requests.get(f"{OPENHAB_URL}/rest/items", headers=headers, timeout=60)
        response.raise_for_status()
        items = response.json()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Fehler beim openHAB-Verbindungsaufbau: {str(e)}")

    ids = []
    documents = []
    metadatas = []

    print(f"[SYNC] Analysiere {len(items)} Items...")
    for item in items:
        name = item.get("name", "")
        label = item.get("label", "") or name
        type_ = item.get("type", "")
        tags = item.get("tags", [])
        group_names = item.get("groupNames", [])
        
        # Extrahiere openbrain-Metadaten falls vorhanden
        openbrain_hint = ""
        openbrain_role = ""
        if "metadata" in item and "openbrain" in item["metadata"]:
            ob = item["metadata"]["openbrain"].get("value", "")
            hint_match = re.search(r'hint=([^,]+)', ob)
            role_match = re.search(r'role=([^,]+)', ob)
            if hint_match: openbrain_hint = hint_match.group(1)
            if role_match: openbrain_role = role_match.group(1)

        # Baue semantischen Textkörper
        tags_str = ", ".join(tags)
        groups_str = ", ".join(group_names)
        
        semantic_text = f"item: {label.lower()}. name: {name.lower()}. typ: {type_.lower()}. tags: {tags_str.lower()}. gruppen: {groups_str.lower()}."
        if openbrain_hint:
            semantic_text += f" info: {openbrain_hint.lower()}."
        if openbrain_role:
            semantic_text += f" rolle: {openbrain_role.lower()}."

        # Raum-Zuordnung mittels Umlaut-Bereinigung ermitteln
        detected_room = "global"
        normalized_name_block = normalize_text(semantic_text)
        for room in ROOM_KEYWORDS:
            if room in normalized_name_block:
                detected_room = room
                break

        ids.append(name)
        documents.append(semantic_text)
        metadatas.append({
            "name": name, 
            "type": type_, 
            "label": label,
            "room": detected_room
        })

    # In 5000er Batches in die DB schreiben (Überschreibt bestehende IDs automatisch)
    batch_size = 5000
    print("[SYNC] Schreibe Daten in die lokale Vektordatenbank...")
    for i in range(0, len(ids), batch_size):
        print(f"[SYNC] Batch {i} bis {min(i+batch_size, len(ids))} von {len(ids)}...")
        collection.upsert(
            ids=ids[i:i+batch_size],
            documents=documents[i:i+batch_size],
            metadatas=metadatas[i:i+batch_size]
        )
    
    print("[SYNC] Synchronisierung erfolgreich abgeschlossen.")
    return {"status": "success", "message": f"{len(ids)} Items synchronisiert (Upsert-Modus)."}


# --- 2. REST-API INTERAKTION MIT OPENHAB ---
def execute_openhab_command(item_name: str, command: str) -> str:
    headers = {"Authorization": f"Bearer {OPENHAB_TOKEN}", "Content-Type": "text/plain"}
    res = requests.post(f"{OPENHAB_URL}/rest/items/{item_name}", data=command, headers=headers)
    return "Erfolgreich geschaltet" if res.status_code == 200 else f"Fehler (Status {res.status_code})"

def get_openhab_state(item_name: str) -> str:
    headers = {"Authorization": f"Bearer {OPENHAB_TOKEN}"}
    res = requests.get(f"{OPENHAB_URL}/rest/items/{item_name}/state", headers=headers)
    return res.text if res.status_code == 200 else "Unbekannt"


# --- 3. DIE INTELLIGENTE INTELLIGENZ-SCHNITTSTELLE (CHAT) ---
@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest, api_key: str = Security(api_key_header)):
    global LAST_USED_ITEM
    
    if api_key and api_key.replace("Bearer ", "") != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API Key")

    user_query = request.messages[-1].content
    normalized_query = normalize_text(user_query)
    query_lower = user_query.lower()
    
    # --- PRÜFUNG: HANDELT ES SICH UM EINEN FOLGEBEFEHL? (Kurzzeitgedächtnis) ---
    is_short_command = len(query_lower.split()) <= 4
    has_action = any(w in query_lower for w in ["an", "ein", "aus", "off", "on", "hoch", "runter", "status", "wie", "wert"])
    has_room = any(room in normalized_query for room in ROOM_KEYWORDS)
    
    if is_short_command and has_action and not has_room and LAST_USED_ITEM:
        best_item_name = LAST_USED_ITEM["name"]
        best_item_meta = LAST_USED_ITEM["meta"]
        best_item_label = best_item_meta["label"]
        print(f"[MIDDLEWARE] Folge-Befehl erkannt. Nutze Gedächtnis-Item: {best_item_name}")
    else:
        # --- NORMALE RAUM- UND VEKTORSUCHE ---
        search_filter = {}
        for room in ROOM_KEYWORDS:
            if room in normalized_query:
                search_filter = {"room": room}
                print(f"[MIDDLEWARE] Raum '{room}' in Anfrage erkannt. Filtere Vektordatenbank.")
                break

        if search_filter:
            results = collection.query(query_texts=[normalized_query], n_results=3, where=search_filter)
        else:
            results = collection.query(query_texts=[normalized_query], n_results=3)
        
        if not results['ids'] or len(results['ids'][0]) == 0:
            return generate_openai_response("Ich konnte kein passendes Gerät finden.")
        
        best_item_name = results['ids'][0][0]
        best_item_meta = results['metadatas'][0][0]
        best_item_label = best_item_meta['label']

    print(f"[MIDDLEWARE] Gewähltes Item: {best_item_name} ({best_item_label}) [Typ: {best_item_meta['type']}]")

    # --- SCHUTZ VOR AUDIO- UND SYSTEMKONFLIKTEN ---
    item_name_lower = best_item_name.lower()
    if "audio" in item_name_lower or "medialib" in item_name_lower or "play" in item_name_lower:
        if not any(w in query_lower for w in ["musik", "spiel", "song", "album", "interpret", "lautstärke"]):
            return generate_openai_response(f"⚠️ Suchkonflikt: Ich habe das Musik-Item `{best_item_name}` gefunden, aber deine Frage bezog sich nicht auf Musik. Bitte frage präziser.")

    # --- OPTIMIERTE INTENT-WEICHE (FRAGEN HABEN PRIORITÄT) ---
    
    # 1. Status abfragen (Greift auch, wenn "an"/"aus" in einer Frage vorkommen)
    if any(w in query_lower for w in ["status", "wie", "zustand", "wert", "temperatur", "grad", "prozent"]) or \
       query_lower.strip().startswith(("sind ", "ist ", "hat ", "läuft ", "gibt ")):
        
        state = get_openhab_state(best_item_name)
        
        if state.upper() == "ON" and "an" in query_lower:
            answer = f"💡 **Ja**, `{best_item_label}` ist aktuell **an** (ON)."
        elif state.upper() == "OFF" and "aus" in query_lower:
            answer = f"💡 **Ja**, `{best_item_label}` ist aktuell **aus** (OFF)."
        else:
            answer = f"💡 **Status-Abfrage:**\n• **Item:** `{best_item_label}` (`{best_item_name}`)\n• **Aktueller Zustand:** **{state}**"
    
    # 2. Ausschalten
    elif any(w in query_lower for w in ["aus", "schließe", "runter", "off", "deaktivieren"]):
        cmd = "OFF" if best_item_meta['type'] in ["Switch", "Dimmer"] else "DOWN"
        status = execute_openhab_command(best_item_name, cmd)
        answer = f"🛑 **Befehl gesendet:**\n• **Ziel:** `{best_item_label}` (`{best_item_name}`)\n• **Kommando:** `{cmd}`\n• **Ergebnis:** {status}"
        
    # 3. Einschalten
    elif any(w in query_lower for w in ["an", "ein", "öffne", "hoch", "on", "aktivieren", "mach "]):
        cmd = "ON" if best_item_meta['type'] in ["Switch", "Dimmer"] else "UP"
        status = execute_openhab_command(best_item_name, cmd)
        answer = f"🟢 **Befehl gesendet:**\n• **Ziel:** `{best_item_label}` (`{best_item_name}`)\n• **Kommando:** `{cmd}`\n• **Ergebnis:** {status}"
    
    # 4. Numerischen Wert setzen (Dimmer / Rollos / Heizung)
    elif re.search(r'\d+', user_query):
        target_value = re.findall(r'\d+', user_query)[0]
        status = execute_openhab_command(best_item_name, str(target_value))
        answer = f"🔢 **Wert geändert:**\n• **Ziel:** `{best_item_label}` (`{best_item_name}`)\n• **Neuer Wert:** `{target_value}`\n• **Ergebnis:** {status}"
        
    else:
        answer = f"🔍 Ich habe das Item `{best_item_label}` (`{best_item_name}`) im System gefunden. Bitte sag mir genauer, ob ich es schalten oder abfragen soll."

    # --- INS KURZZEITGEDÄCHTNIS SCHREIBEN ---
    LAST_USED_ITEM = {
        "name": best_item_name,
        "meta": best_item_meta
    }

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
            "finish_reason": "stop"
        }]
    }

@app.get("/v1/models")
async def list_models():
    return {"object": "list", "data": [{"id": "oh-hybrid-local", "object": "model", "owned_by": "local"}]}
