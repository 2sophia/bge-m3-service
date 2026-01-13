# BGE-M3 Embeddings Service

Servizio di embeddings ad alte prestazioni per Sophia, basato su **BGE-M3** di BAAI. Supporta embeddings **densi**, **sparse** e **ColBERT**, oltre al reranking di documenti.

## 🚀 Quick Start (NVIDIA GPU)

```bash
docker run -d --gpus all sophiacloud/bge-m3-service:cuda
```
## 🚀 Quick Start (AMD GPU)

```bash
docker run --device=/dev/kfd --device=/dev/dri --group-add video sophiacloud/bge-m3-service:rocm
```
## 🚀 Quick Start (CPU – x86 / ARM)

```bash
docker run -d sophiacloud/bge-m3-service:cpu
```

---

## 🖥️ Piattaforme supportate

| Piattaforma | Stato | Image |
|------------|------|-------|
| CPU x86_64 | ✅ | `:cpu` |
| CPU ARM64 | ✅ | `:cpu` |
| NVIDIA GPU (CUDA) | ✅ | `:cuda` |
| Jetson | ❌ | roadmap futura |

---
## Caratteristiche

- **Embeddings Multi-Modalità**: Dense, Sparse e ColBERT
- **Reranking**: Riordina documenti in base alla rilevanza rispetto a una query
- **Batching Intelligente**: Ottimizza le chiamate GPU raggruppando richieste multiple
- **GPU & CPU**: Supporto completo per CUDA e CPU
- **API RESTful**: Interfaccia FastAPI con documentazione automatica
- **Docker Ready**: Immagine Docker ottimizzata già pubblicata

## API Endpoints

### Health Check

```bash
GET /healthz
```

**Risposta:**
```json
{
  "status": "ok",
  "model": "BAAI/bge-m3",
  "device": "cuda"
}
```

### Embeddings

```bash
POST /v1/embeddings
```

**Request Body:**
```json
{
  "input": "Questo è un testo di esempio",
  "return_dense": true,
  "return_sparse": true,
  "return_colbert": false
}
```

**O con array:**
```json
{
  "input": [
    "Primo testo",
    "Secondo testo"
  ],
  "return_dense": true,
  "return_sparse": false,
  "return_colbert": false
}
```

**Risposta:**
```json
{
  "results": [
    {
      "index": 0,
      "embeddings": {
        "dense": [0.123, -0.456, ...],
        "sparse": {
          "indices": [101, 2023, ...],
          "values": [0.89, 0.76, ...]
        }
      }
    }
  ]
}
```

### Reranking

```bash
POST /v1/rerank
```

**Request Body:**
```json
{
  "query": "Come funziona l'intelligenza artificiale?",
  "documents": [
    "L'IA è un campo dell'informatica...",
    "La pasta è buonissima...",
    "I modelli di machine learning..."
  ]
}
```

**Risposta (ordinata per rilevanza):**
```json
{
  "results": [
    {
      "index": 0,
      "document": {
        "text": "L'IA è un campo dell'informatica..."
      },
      "relevance_score": 0.95
    },
    {
      "index": 2,
      "document": {
        "text": "I modelli di machine learning..."
      },
      "relevance_score": 0.82
    },
    {
      "index": 1,
      "document": {
        "text": "La pasta è buonissima..."
      },
      "relevance_score": 0.12
    }
  ]
}
```

## Esempi di Utilizzo

### Python

```python
import requests

# Embeddings
response = requests.post(
    "http://localhost:8004/v1/embeddings",
    json={
        "input": "Testo da embeddare",
        "return_dense": True,
        "return_sparse": True
    }
)
result = response.json()
print(result["results"][0]["embeddings"]["dense"])

# Reranking
response = requests.post(
    "http://localhost:8004/v1/rerank",
    json={
        "query": "intelligenza artificiale",
        "documents": [
            "L'IA è un campo dell'informatica",
            "La pizza margherita",
            "Machine learning e deep learning"
        ]
    }
)
ranked = response.json()
for item in ranked["results"]:
    print(f"Score: {item['relevance_score']:.3f} - {item['document']['text']}")
```

### cURL

```bash
# Health check
curl http://localhost:8004/healthz

# Embeddings
curl -X POST http://localhost:8004/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "input": "Testo di esempio",
    "return_dense": true,
    "return_sparse": true
  }'

# Reranking
curl -X POST http://localhost:8004/v1/rerank \
  -H "Content-Type: application/json" \
  -d '{
    "query": "machine learning",
    "documents": [
      "Deep learning è una tecnica di ML",
      "La pasta al pomodoro",
      "Neural networks e AI"
    ]
  }'
```

## Build Locale

### Prerequisiti
- Docker e Docker Compose
- NVIDIA Docker runtime (per GPU)

### Build dell'Immagine

```bash
cd bge-m3-service
docker build -t sophiacloud/bge-m3-service:1.0.0 .
```

### Sviluppo Locale (senza Docker)

```bash
# Crea virtual environment
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# oppure: venv\Scripts\activate  # Windows

# Installa dipendenze
pip install -r requirements.txt

# Avvia il servizio
python main.py
```

## Documentazione API Interattiva

Dopo aver avviato il servizio, visita:

- **Swagger UI**: http://localhost:8004/docs
- **ReDoc**: http://localhost:8004/redoc

## Limiti

- **Max input per richiesta**: 1024 testi
- **Max documenti per rerank**: 1024
- **Lunghezza massima testo**: 5000 caratteri (configurabile)
- **Lunghezza massima query**: 256 caratteri (configurabile)

## Performance

Il servizio implementa un sistema di **batching automatico** che:
- Raggruppa richieste multiple per ottimizzare l'uso della GPU
- Riduce la latenza media quando ci sono richieste concorrenti
- Supporta fino a 16 richieste simultanee per batch (configurabile)

## Troubleshooting

### GPU non rilevata

Verifica che il container abbia accesso alla GPU:
```bash
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi
```

### Out of Memory (OOM)

Riduci la frazione di memoria allocata:
```bash
docker run -e BGE_MODEL_MEMORY=0.15 ...
```

O riduci il batch size:
```bash
docker run -e M3_BATCH_SIZE=4 ...
```

### Timeout

Aumenta i timeout per richieste complesse:
```bash
docker run -e M3_REQUEST_TIMEOUT=60 -e M3_GPU_TIMEOUT=30 ...
```

## Variabili d'Ambiente

| Variabile | Default | Descrizione |
|-----------|---------|-------------|
| `M3_PORT` | `8004` | Porta su cui il servizio ascolta |
| `M3_DEVICE` | `cuda` | Device da usare (`cuda` o `cpu`) |
| `M3_MODEL_ID` | `BAAI/bge-m3` | ID del modello da HuggingFace |
| `BGE_MODEL_MEMORY` | `0.20` | Frazione di VRAM da allocare (0.0-1.0) |
| `M3_BATCH_SIZE` | `8` | Dimensione batch per GPU |
| `M3_MAX_REQUESTS` | `16` | Massimo numero di richieste da raggruppare |
| `M3_MAX_LENGTH` | `5000` | Lunghezza massima del testo |
| `M3_MAX_Q_LENGTH` | `256` | Lunghezza massima query (reranking) |
| `M3_FLUSH_TIMEOUT` | `0.01` | Timeout accumulo batch (secondi) |
| `M3_REQUEST_TIMEOUT` | `30` | Timeout richiesta HTTP (secondi) |
| `M3_GPU_TIMEOUT` | `10` | Timeout operazioni GPU (secondi) |
| `M3_DEBUG` | `true` | Abilita log di debug |
| `M3_RERANK_WEIGHTS` | `0.30, 0.65, 0.05` | Pesi per dense, sparse, colbert nel reranking |

---

## Licenza

This service uses the model "BAAI/bge-m3", which is licensed under
the MIT License.

The model and its derived versions are not covered by the license
applied to this service.

The service code, API, and orchestration logic are licensed under
the PolyForm Noncommercial License 1.0.0.



