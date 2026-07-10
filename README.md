# openHAB Semantic Hybrid Bridge

An ultra-fast, lightweight Python middleware designed to bridge **openHAB** with **Open WebUI** using semantic vector search on the CPU. It bypasses the need for large, resource-heavy LLMs by translating natural language into openHAB REST API commands using a local ChromaDB instance.

## Features
- **Context-Aware Follow-up Commands:** Simple phrases like *"Turn it off"* automatically target the last used device.
- **Strict Query Prioritization:** Differentiates between state checks (*"Are the lights on?"*) and actual toggle commands.
- **Smart Room Filtering:** Uses regex and Umlaut-normalization (`ä` -> `ae`, etc.) to isolate search queries to specific rooms.
- **Audio & System Variable Protection:** Prevents unwanted media library triggers when adjusting smart home equipment.
- **Resource Efficient:** Runs entirely on the CPU using `all-MiniLM-L6-v2` embeddings.

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

Open `app.py` and configure your openHAB credentials:

```python
OPENHAB_URL = "http://<YOUR_OPENHAB_IP>:8080"
OPENHAB_TOKEN = "oh.your_actual_token_here"
API_KEY = "your_local_key"
```

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
curl -X POST [http://127.0.0.1:8000/api/sync](http://127.0.0.1:8000/api/sync)
```

*Note: This utilizes an `upsert` mechanism. You can run this command anytime your openHAB items change without breaking existing data.*

---

## Connecting Open WebUI

(Normally it should detect this confiugrations automatically.)

1. Open your **Open WebUI** interface.
2. Navigate to **Admin Settings** -> **Connections**.
3. Under **OpenAI API**, add a connection:
* **API URL:** `http://127.0.0.1:8000/v1` *(If Open WebUI runs in host network mode, otherwise use your VM's network IP on port 8000)*
* **API Key:** `local-oh-key`


4. Click **Save** and **Refresh**.
5. Select the model **`oh-hybrid-local`** from the chat dropdown and start controlling your home!

---
