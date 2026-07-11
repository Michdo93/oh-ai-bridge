# openHAB Semantic Hybrid Bridge

An ultra-fast, lightweight Python middleware designed to bridge **openHAB** with **Open WebUI** using semantic vector search on the CPU. It bypasses the need for large, resource-heavy LLMs by translating natural language into openHAB REST API commands using a local ChromaDB instance.

Version 4 adds real integration with openHAB's **Semantic Model** (the same Location/Equipment tag hierarchy HABot uses), so you're no longer limited to phrasing that matches an item's literal name — "turn on the light" now works even if the item is technically named `Bedroom_Lamp_Switch`.

## Features
- **Dual-Signal Semantic Resolution:** Reads openHAB's own computed `semantics` item-metadata (`Point_Control`, `relatesTo: Property_Light`, `isPointOf: <Equipment/Location group>`) — the same data HABot uses — **and** falls back to plain substring matching on an item's name/label/groups (e.g. `iKueche_Hue_Lampen_Schalter` literally contains "kueche" and "lampen"). Either signal is enough; you're not dependent on your installation being fully/consistently tagged.
- **Deterministic Candidate Filtering:** Instead of trusting a vector-similarity fallback whenever tag metadata is incomplete, the bridge hard-filters the actual item list in Python by room/device class first. Embeddings are only used to *rank* an already-relevant shortlist, never to search the entire, unrelated item corpus (e.g. media library entries no longer show up for a kitchen-light question).
- **Synonym & Fuzzy Matching:** A user-editable `synonyms.yaml` extends the tags/properties from openHAB with your own vocabulary (slang, abbreviations). RapidFuzz adds typo tolerance ("lich" → "licht") without needing an LLM.
- **Clarification on Genuine Ambiguity:** If multiple same-type devices remain after filtering, the bridge asks a short follow-up and lists the actual (unique) item **names**, not just their (often duplicated) labels.
- **Exact-Name Shortcut:** Naming the precise openHAB item ID in the chat (e.g. *"Sende ON an iKueche_Hue_Lampen_Schalter"*) bypasses synonym/fuzzy resolution entirely and uses it directly.
- **Context-Aware Follow-up Commands:** Simple phrases like *"Turn it off"* automatically target the last used device — resolved from the conversation history Open WebUI already sends, not from shared global state, so it stays correct across multiple concurrent chats/users.
- **Multi-Command Support:** Requests like *"Turn on the light and the heating"* are split and handled individually.
- **Strict Query Prioritization:** Differentiates between state checks (*"Are the lights on?"*) and actual toggle commands.
- **Audio & System Variable Protection:** Prevents unwanted media library triggers when adjusting smart home equipment.
- **Config via `.env`:** No more editing credentials directly in `app.py`.
- **Resource Efficient:** Still runs entirely on the CPU using `all-MiniLM-L6-v2` embeddings — no GPU, no LLM, no cloud.

---

## Installation & Setup

### 1. Pre-Installation
At first you have to run `apt install` to make some pre-installations:

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y python3-pip python3-venv git curl git-lfs
```

### 2. Docker
Then you have to install `Docker` because you should run `Open WebUI` inside a `container`:

```bash
# Add Docker's official GPG key:
sudo apt update
sudo apt install ca-certificates curl
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc

# Add the repository to Apt sources:
sudo tee /etc/apt/sources.list.d/docker.sources <<EOF
Types: deb
URIs: https://download.docker.com/linux/ubuntu
Suites: $(. /etc/os-release && echo "${UBUNTU_CODENAME:-$VERSION_CODENAME}")
Components: stable
Architectures: $(dpkg --print-architecture)
Signed-By: /etc/apt/keyrings/docker.asc
EOF

sudo apt update
sudo apt install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
sudo usermod -aG docker $USER
sudo systemctl enable docker
```

### 3. Pulling and Configuring the Open WebUI container
After installing `Docker` you can run this command for your `container`:

```bash
docker run -d \
  --network=host \
  -v open-webui:/app/backend/data \
  -e OPENAI_API_BASE_URL="http://127.0.0.1:8000/v1" \
  -e OPENAI_API_KEY="your_local_key" \
  --name open-webui \
  --restart always \
  ghcr.io/open-webui/open-webui:main
```

### 4. Clone & Prepare Environment
Clone this repository to your Linux host (e.g., your Proxmox VM):

```bash
cd ~
git clone https://github.com/Michdo93/oh-ai-bridge
cd oh-ai-bridge
```

Create a virtual environment and install dependencies:

```bash
python3 -m venv .
source bin/activate
pip install -r requirements.txt
```

### 5. Configuration

Credentials and paths are no longer hardcoded in `app.py` — copy the example environment file and fill in your values:

```bash
cp .env.example .env
nano .env
```

```ini
OPENHAB_URL=http://<YOUR_OPENHAB_IP>:8080
OPENHAB_TOKEN=oh.your_actual_token_here
API_KEY=your_local_key
SYNONYMS_PATH=./synonyms.yaml
CHROMA_PATH=./chroma_db
EMBEDDING_MODEL=all-MiniLM-L6-v2
```

**Never commit your `.env` file to git.**

Optionally, review `synonyms.yaml` and add any device names, room names, or German phrasing specific to your setup that openHAB's own semantic tags don't already cover (e.g. custom rooms like a "Smart-Home-Labor", or local slang for a device). This file is merged with whatever your openHAB instance reports live under `/rest/tags` — you don't need to duplicate anything openHAB already knows.

### 6. Setup systemd Service (Autostart)

To ensure the bridge starts automatically when your system boots, create a systemd service file:

```bash
sudo nano /etc/systemd/system/oh-ai-bridge.service
```

Paste the following configuration (adjust the username `openhab` if necessary):

```ini
[Unit]
Description=openHAB AI Hybrid Bridge
After=network.target

[Service]
Type=simple
User=openhab
WorkingDirectory=/home/openhab/oh-ai-bridge
EnvironmentFile=/home/openhab/oh-ai-bridge/.env
ExecStart=/home/openhab/oh-ai-bridge/bin/uvicorn app:app --host 0.0.0.0 --port 8000
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

Enable and start the service:

```bash
sudo systemctl daemon-reload
sudo systemctl enable oh-ai-bridge.service
sudo systemctl start oh-ai-bridge.service
```

---

## Database Synchronization

To pull your items from openHAB into the local ChromaDB vector database, trigger the initial sync via `curl`:

```bash
curl -X POST http://127.0.0.1:8000/api/sync
```

*Note: This utilizes an `upsert` mechanism. You can run this command anytime your openHAB items change without breaking existing data.*

During sync, the bridge requests `/rest/items?metadata=.+` (metadata is **not** included by default on the bulk items endpoint — only on a single-item `GET`, so this explicit request matters) and resolves, per item, its Location/Equipment/Property classification by following the `semantics` metadata's `isPointOf`/`isPartOf` chain upward — the same resolution HABot performs internally. Where that metadata is missing or incomplete, it falls back to categorizing the item's raw `tags` via `/rest/tags`, and independently, to plain substring matching against the item's own name/label/groups. The response tells you how many items ended up classified:

```json
{
  "status": "success",
  "message": "142 Items synchronisiert.",
  "tag_registry_source": "openhab_live",
  "items_with_semantic_classification": 118,
  "items_total": 142
}
```

If `tag_registry_source` is `yaml_fallback`, your openHAB version doesn't expose `/rest/tags` (older than 4.0, or the endpoint is disabled) — the bridge still works, relying on the room/device synonym lists in `synonyms.yaml` plus substring matching.

To see exactly how the bridge interpreted your tags overall, check:

```bash
curl http://127.0.0.1:8000/api/tags
```

To see exactly how one specific item was classified (very useful when debugging why a query does or doesn't match, e.g. the item from your own REST inspection):

```bash
curl http://127.0.0.1:8000/api/items/iKueche_Hue_Lampen_Schalter
```

A basic liveness check (also reports how many items are currently cached in memory) is available at:

```bash
curl http://127.0.0.1:8000/health
```

---

## Connecting Open WebUI

(Normally it should detect this configuration automatically.)

1. Open your **Open WebUI** interface.
2. Navigate to **Admin Settings** -> **Connections**.
3. Under **OpenAI API**, add a connection:
* **API URL:** `http://127.0.0.1:8000/v1` *(If Open WebUI runs in host network mode, otherwise use your VM's network IP on port 8000)*
* **API Key:** the value you set for `API_KEY` in your `.env`


4. Click **Save** and **Refresh**.
5. Select the model **`oh-hybrid-local`** from the chat dropdown and start controlling your home!

---

## How Query Resolution Works

1. If the message names an exact openHAB item ID, that item is used directly — no further resolution needed.
2. Otherwise the message is normalized (lowercased, umlauts folded: `ä` → `ae`, etc.).
3. **Device class** (Equipment *or* Property tag, e.g. both "Lightbulb" and "Light" count) and **room** are resolved against your openHAB Semantic Model + `synonyms.yaml`, first via exact/substring match, then via RapidFuzz for typo tolerance. A query can resolve to *several* candidate tags for the same word (e.g. "licht" matching both an Equipment and a Property tag) — all of them count.
4. **Action** (turn on/off, raise/lower, status query, or a numeric value) is resolved from the same normalized text — questions ("is the light on?") are prioritized over plain toggle words.
5. The full item list (cached in memory after each sync) is **hard-filtered in Python**: an item must match the resolved device class *or* room via either signal (resolved semantic tag, or a plain substring hit in its own name/label/groups) for every dimension the query actually resolved. Nothing is shown just because the embedding model thought it was similar. If filtering by both dimensions yields nothing, dimensions are relaxed one at a time (never both at once) before anything is dropped entirely.
6. Only if the query resolved *no* known room/device vocabulary at all does the bridge fall back to a free vector search over the whole item corpus, as a last resort.
7. The embedding model ranks the (already filtered, small) candidate list — it never decides *whether* an item is relevant, only their *order*.
8. If several same-type devices remain after filtering, the bridge asks you to clarify, listing the actual item names (not just labels, which are often duplicated, e.g. multiple items labeled "Schalter").
9. Follow-up commands ("turn it off") resolve the previously used device from the conversation history Open WebUI sends with every request — no shared state between different chats or users.

---

## Notes

- The exact JSON schema of `/rest/tags` and of the `semantics` item-metadata can vary slightly between openHAB versions (e.g. field names `uid` vs. `name`). The parsers in `app.py` try the common variants defensively; use `/api/tags` and `/api/items/{name}` after your first sync to confirm rooms/devices were classified as expected, and adjust field names in `resolve_item_semantics()` / `fetch_openhab_tag_registry()` if needed for your version.
- Room/device filtering is intentionally **hard** (an item must match, not just "seem similar"). If a query returns "kein passendes Gerät gefunden" even though you know the device exists, check `/api/items/{name}` for that device — most likely neither the semantics-metadata chain nor a substring match in its name/label/groups covers the word you used, which is exactly what `synonyms.yaml` is for.
- This project intentionally does not use an LLM, TensorFlow/Rasa/spaCy intent classifier, or a training database/admin UI — all of that would reintroduce the CPU/GPU cost this bridge is designed to avoid. `synonyms.yaml` is the lightweight, file-based equivalent.
