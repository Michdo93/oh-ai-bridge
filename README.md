# openHAB Semantic Hybrid Bridge

An ultra-fast, lightweight Python middleware designed to bridge **openHAB** with **Open WebUI** using classic NLP techniques on the CPU. It bypasses the need for large, resource-heavy LLMs — or even a GPU-hungry embedding model — by translating natural language into openHAB REST API commands using openHAB's own Semantic Model plus a small TF-IDF fallback ranker.

**Target hardware:** this bridge is deliberately built to run comfortably on something as modest as a **Raspberry Pi 3 Model B+** (1.4 GHz quad-core Cortex-A53, 1 GB RAM) — the same class of hardware many people already run their openHAB instance on. No GPU, no model downloads, no cloud calls.

## Features
- **openHAB REST access via `python-openhab-rest-client`:** all communication with openHAB (items, commands, state, semantic tags) goes through the official Python client library instead of hand-rolled `requests` calls.
- **Dual-Signal Semantic Resolution:** reads openHAB's own computed `semantics` item-metadata (`Point_Control`, `relatesTo: Property_Light`, `isPointOf: <Equipment/Location group>`) — the same data HABot uses — **and** falls back to word-boundary-safe text matching on an item's name/label/groups. Either signal is enough; you're not dependent on your installation being fully/consistently tagged.
- **Deterministic Candidate Filtering:** the actual item list is hard-filtered in Python by room/device class first. A lightweight TF-IDF ranker only orders an already-relevant shortlist — it never decides *whether* something is relevant, so unrelated items (media library tracks, test items, etc.) can't sneak in just because of a coincidental word fragment.
- **Synonym & Fuzzy Matching:** a user-editable `synonyms.yaml` extends openHAB's own tags/properties with your own vocabulary. RapidFuzz adds typo tolerance without needing an LLM.
- **Collective vs. individual devices:** distinguishes e.g. `iBad_Hue_Lampen_Schalter` (all bathroom lamps) from `iBad_Hue_Lampe1_Schalter` (one specific lamp) — defaults to the collective device unless a number is given.
- **Device listings:** "Welche Lampen/Geräte gibt es im Bad?" triggers an enumeration instead of picking a single item.
- **Local command-type validation:** before sending a command, the bridge checks locally whether the item's type can even sensibly accept it (e.g. refuses to send `ON` to a `Number` item) — no wasted round trip, no side effect, just a quick sanity check.
- **Clarification on Genuine Ambiguity:** lists actual item **names**, not just labels (which are frequently duplicated, e.g. many items labeled "Schalter").
- **Exact-Name Shortcut:** naming the precise openHAB item ID in the chat bypasses synonym/fuzzy resolution entirely.
- **Context-Aware Follow-ups:** resolved from the conversation history Open WebUI sends with every request — no shared state between different chats or users.
- **Config via `.env`:** no more editing credentials directly in `app.py`.

---

## Why TF-IDF instead of an embedding model?

Earlier versions used `sentence-transformers` + `torch` + `chromadb` for semantic vector search. On a Raspberry Pi 3 B+ this is a poor fit:
- Loading a transformer model needs real RAM and CPU time, on hardware that has neither to spare.
- It also unauthenticated-pings `huggingface.co` on every startup ("Warning: You are sending unauthenticated requests to the HF Hub") — not what "resource-friendly, no cloud" is supposed to mean.

In this architecture, room/device resolution is handled deterministically by openHAB's Semantic Model + `synonyms.yaml` first. The remaining, much smaller job — ranking a handful of already-filtered candidates, or falling back when literally no known vocabulary was recognized — doesn't need a neural model at all. A classic **TF-IDF vector space** (`scikit-learn`, pure NumPy/C, no downloads) does that job in milliseconds, even over tens of thousands of items, and needs no network access ever.

If you later want to experiment with genuinely small on-device models beyond TF-IDF (e.g. a distilled/quantized ONNX sentence encoder via `fastembed`, or a tiny `spaCy` German pipeline for lemmatization), that's a reasonable next step — but given how little semantic heavy-lifting is actually left for the ranker in this design, it's unlikely to move the needle much versus the complexity/resource cost it adds on a Pi 3.

---

## Installation & Setup

### 1. Pre-Installation
```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y python3-pip python3-venv git curl
```

### 2. Docker (for Open WebUI)
```bash
sudo apt update
sudo apt install ca-certificates curl
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc

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

### 3. Open WebUI container
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
```bash
cd ~
git clone https://github.com/Michdo93/oh-ai-bridge
cd oh-ai-bridge
python3 -m venv .
source bin/activate
pip install -r requirements.txt
```

### 5. Configuration
```bash
cp .env.example .env
nano .env
```
```ini
OPENHAB_URL=http://<YOUR_OPENHAB_IP>:8080
OPENHAB_TOKEN=oh.your_actual_token_here
API_KEY=your_local_key
SYNONYMS_PATH=./synonyms.yaml
CACHE_PATH=./item_cache.pkl
```
**Never commit your `.env` file to git.**

**⚠️ YAML gotcha when editing `synonyms.yaml`:** YAML (1.1) parses the bare, unquoted words `on`, `off`, `yes`, `no`, `true`, `false` as **booleans**, not text — even as dictionary keys. If you add or edit entries containing these words, always quote them: `"on"`, `"ON"`, `"off"`, etc. The bridge validates the config defensively at startup and logs a warning if it finds a non-string value instead of crashing, but a quoted value is the correct fix.

### 6. Setup systemd Service (Autostart)
```bash
sudo nano /etc/systemd/system/oh-ai-bridge.service
```
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
```bash
sudo systemctl daemon-reload
sudo systemctl enable oh-ai-bridge.service
sudo systemctl start oh-ai-bridge.service
```

---

## Database Synchronization

```bash
curl -X POST http://127.0.0.1:8000/api/sync
```

This fetches all items (via `Items.getAllItems(metadata=".+")`), resolves each item's Location/Equipment/Property classification by following the `semantics` metadata's `isPointOf`/`isPartOf` chain (falling back to `Tags.getTags()` categorization, and independently to plain word-boundary text matching), builds a TF-IDF index over the results, and persists everything to a single file (`CACHE_PATH`) so a later restart doesn't require an immediate re-sync.

```json
{
  "status": "success",
  "message": "142 Items synchronisiert.",
  "tag_registry_source": "openhab_live",
  "items_with_semantic_classification": 118,
  "items_total": 142
}
```

If `tag_registry_source` is `yaml_fallback`, your openHAB version doesn't expose the tags API (older than 4.0) — the bridge still works, relying on `synonyms.yaml` plus text matching.

Debug endpoints:
```bash
curl http://127.0.0.1:8000/api/tags                                  # resolved vocabulary overview
curl http://127.0.0.1:8000/api/items/iBad_Hue_Lampen_Schalter        # how one specific item was classified
curl http://127.0.0.1:8000/health                                     # liveness + cache/TF-IDF status
```

---

## Connecting Open WebUI

1. **Admin Settings** → **Connections** → add an **OpenAI API** connection:
   - **API URL:** `http://127.0.0.1:8000/v1`
   - **API Key:** the value you set for `API_KEY` in your `.env`
2. Save, refresh, select **`oh-hybrid-local`** from the model dropdown.

---

## How Query Resolution Works

1. Exact openHAB item ID mentioned in the message → used directly, no further resolution.
2. Message normalized (lowercased, umlauts folded: `ä` → `ae`).
3. **Device class** (Equipment *or* Property tag) and **room** resolved against openHAB's Semantic Model + `synonyms.yaml` (exact/substring match, then RapidFuzz for typos). A word can resolve to several candidate tags at once.
4. **Action** resolved: literal openHAB commands (`ON`/`OFF`/`UP`/`DOWN`) are recognized directly as language-independent tokens; German phrasing and questions ("ist ... an?") are recognized via `synonyms.yaml`; "welche ... gibt es" triggers a device **listing** instead of picking a single item.
5. The full item list (cached after sync) is **hard-filtered in Python**: an item must match the resolved device class *and* room via either signal (semantic tag or word-boundary text hit) for every dimension the query resolved. Auxiliary helper points (transition timers, RGB text mirrors, alarm-mode strings) are filtered out in favor of the primary controllable point (typically the `Switch`).
6. Collective vs. numbered devices are disambiguated: no number in the query → the collective/unnumbered device is preferred; a number given → only the matching numbered device is kept.
7. Only if the query resolved *no* known room/device vocabulary at all does the bridge fall back to a small TF-IDF similarity search over the whole item corpus.
8. Before sending any command, the item's type is checked locally against a small allow-list (e.g. a `Number` item can't receive `ON`) to avoid pointless/incorrect REST calls.
9. If several same-type devices remain after all filtering, the bridge asks you to clarify, listing actual item names.
10. Follow-up commands ("turn it off") resolve the previously used device from the conversation history — no shared state between different chats or users.

---

## Notes

- **`python-openhab-rest-client`'s built-in `openhab.tests` module is intentionally *not* used** for command validation: per its own documentation, calling e.g. `ItemsTest.testSendCommand(...)` still executes the real command — it's a try/except wrapper around the same action, not a dry run. It gives no safety benefit over just calling `sendCommand` directly. The local type-based `command_allowed()` check in this bridge covers the actual goal ("would this command even make sense for this item?") without any side effect or extra network round trip.
- The exact JSON schema of `Tags.getTags()` and the `semantics` item-metadata can vary slightly between openHAB versions. The parsers try common field-name variants defensively; use `/api/tags` and `/api/items/{name}` after a sync to confirm rooms/devices were classified as expected.
- Room/device filtering is intentionally **hard** (an item must match, not just "seem similar"). If a query returns "kein passendes Gerät gefunden" for a device you know exists, check `/api/items/{name}` for it — most likely neither the semantics chain nor a text match covers the word you used, which is exactly what `synonyms.yaml` is for.
- This project intentionally avoids an LLM, a TensorFlow/Rasa/spaCy intent classifier, a training database/admin UI, and (as of this version) even a transformer embedding model — all of that would reintroduce CPU/GPU/memory cost disproportionate to a Raspberry Pi 3 B+. `synonyms.yaml` + TF-IDF is the lightweight equivalent.
