# AgroVision AI

Sistema de monitoramento operacional que combina visao computacional (YOLOv8) e um LLM local (Ollama) para detectar objetos em video e descrever a situacao em linguagem natural.

## Pre-requisitos

- Python 3.11.x
- [Ollama](https://ollama.com/download) instalado

## Setup

### Windows (PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
```

### Linux / macOS

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

## Ollama

Em um terminal separado:

```bash
ollama serve
ollama pull llama3
```

Use `llama3.2:3b` se a maquina for fraca.

## Rodar o backend

```bash
python -m uvicorn app:app --reload
```

Dashboard em `http://127.0.0.1:8000/`.

## Configuracao (`.env`)

| Variavel | Default | Descricao |
|---|---|---|
| `OLLAMA_URL` | `http://127.0.0.1:11434/api/chat` | Endpoint do Ollama |
| `OLLAMA_MODEL` | `llama3` | Modelo LLM |
| `CAMERA_SOURCE` | URL Caltrans | Webcam (int), stream HLS/RTSP ou arquivo |
| `AGENT_EVENT_LIMIT` | `12` | Quantos eventos o agente ve por chamada |

## Rotas HTTP

| Metodo | Rota | O que faz |
|---|---|---|
| GET | `/` | Dashboard |
| GET | `/health` | Status + `ollama_available` |
| GET | `/camera/status` | Estado da captura |
| GET | `/agent/status` | Preview do contexto enviado ao LLM |
| GET | `/events` | Ultimas deteccoes |
| GET | `/frame` | Snapshot JPEG atual |
| GET | `/video_feed` | Stream MJPEG continuo |
| POST | `/chat` | Pergunta ao agente |

## Exemplo de chamada ao chat

```bash
curl -X POST -H "Content-Type: application/json" \
  -d '{"question":"o que esta acontecendo?", "history": []}' \
  http://127.0.0.1:8000/chat
```

## Troubleshooting

| Sintoma | Onde olhar |
|---|---|
| `Could not import module "app"` | Terminal esta na pasta errada; rode na raiz do projeto |
| `/video_feed` nao abre mas `/camera/status` diz `online: true` | Problema no gerador MJPEG em `services/video_monitor.py` |
| Agente responde de forma generica | Conferir `/agent/status`: se `events_in_context` for 0, o YOLO nao esta detectando |
| PowerShell bloqueia ativacao do `.venv` | `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` |

## Licenca

MIT - veja `LICENSE`.
