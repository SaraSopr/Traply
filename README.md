![logo](image.png)
# 🗺️ Travel Recommender System

Hybrid recommendation engine con **persistenza PostgreSQL**.
I dati Apify vengono fetchati **una sola volta** e conservati nel DB per sempre.

---

## Struttura

```
travel_recommender/
├── main.py                              # Pipeline end-to-end
├── .env                                 # APIFY_API_TOKEN, DATABASE_URL
├── requirements.txt
│
├── src/
│   ├── collectors/
│   │   ├── apify_collector.py           # Fetch Apify + save DB in real-time
│   │   └── synthetic_users.py           # Genera utenti e rating sintetici
│   ├── recommender/
│   │   ├── hybrid_recommender.py        # CB + CF + Time-Aware + Vector
│   │   ├── time_aware.py                # Layer temporale contestuale
│   │   └── vector_layer.py              # Embeddings sentence-BERT
│   └── utils/
│       └── database.py                  # PostgreSQL manager (SQLAlchemy)
│
├── tests/
│   ├── test_project_structure.py
│   ├── test_synthetic_users.py
│   ├── test_time_aware.py
│   ├── test_database_utils.py
│   └── test_hybrid_recommender.py
│
└── *.py                                 # wrapper legacy (compatibilità)
```

---

## Setup (10 minuti)

### 1. Avvia PostgreSQL con Docker
```bash
docker run --name travel-db \
  -e POSTGRES_PASSWORD=travel123 \
  -e POSTGRES_DB=travel_recommender \
  -p 5432:5432 -d postgres:15
```

### 2. Configura il progetto
```bash
pip install -r requirements.txt
```

Nel file `.env` imposta:
```bash
APIFY_API_TOKEN=apify_api_YOUR_TOKEN
DATABASE_URL=postgresql://postgres:travel123@localhost:5432/travel_recommender
```

### 3. Lancia la pipeline
```bash
# Prima volta: fetcha da Apify e salva nel DB
python main.py --city "Roma, Italia" --n-users 300

# Volte successive: legge dal DB (zero costi Apify!)
python main.py --city "Roma, Italia"

# Forza aggiornamento dati dopo 30 giorni
python main.py --city "Roma, Italia" --force-refresh
```

---

## Logica anti-spreco Apify

```
Ogni chiamata a main.py:
    ┌─────────────────────────┐
    │ Dati Roma nel DB?       │
    │ e fetchati < 30 giorni? │
    └────────┬────────────────┘
             │ Sì ──→ carica dal DB  💰 GRATIS
             │ No ──→ chiama Apify
             │         └─→ salva nel DB categoria per categoria
             │             (sicuro anche se crasha a metà)
```

---

## Tabelle DB

| Tabella | Contenuto |
|---------|-----------|
| `activities` | POI fetchati da Apify (permanenti) |
| `users` | Profili utente reali + sintetici |
| `ratings` | Interazioni utente-attività |
| `recommendation_cache` | Raccomandazioni precalcolate (TTL 24h) |
| `apify_fetch_log` | Registro chiamate + costi Apify |

---

## Prossimi step

- [x] Vector layer (sentence-BERT embeddings su PostgreSQL + pgvector)
- [ ] Itinerary optimizer (TSP con OR-Tools)
- [ ] API FastAPI per esporre il recommender
- [ ] LLM layer per narrazione naturale dell'itinerario

---

## Test

```bash
python -m unittest discover -s tests -v
```
