![logo](image.png)
# 🗺️ Travel Recommender System

Hybrid recommendation engine con persistenza PostgreSQL.
I dati Apify vengono fetchati una sola volta e riutilizzati dal DB.

---

## Project structure

```
Traply/
├── main.py                              # End-to-end pipeline
├── .env                                 # APIFY_API_TOKEN, DATABASE_URL
├── requirements.txt
│
├── src/
│   ├── collectors/
│   │   ├── apify_collector.py           # Apify fetch + realtime DB persistence
│   │   └── synthetic_users.py           # Synthetic users and ratings generation
│   ├── recommender/
│   │   ├── hybrid_recommender.py        # CB + CF + Time-Aware + Vector
│   │   ├── time_aware.py                # Temporal contextual layer
│   │   └── vector_layer.py              # sentence-BERT embeddings
│   └── utils/
│       └── database.py                  # PostgreSQL manager (SQLAlchemy + pgvector)
│
├── tests/
│   ├── test_project_structure.py
│   ├── test_synthetic_users.py
│   ├── test_time_aware.py
│   ├── test_database_utils.py
│   └── test_hybrid_recommender.py
```

---

## Setup

### 1) Start PostgreSQL (Docker)
```bash
docker run --name travel-db \
  -e POSTGRES_PASSWORD=travel123 \
  -e POSTGRES_DB=travel_recommender \
  -p 5432:5432 -d postgres:15
```

### 2) Install dependencies
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Set `.env` with:
```bash
APIFY_API_TOKEN=apify_api_YOUR_TOKEN
DATABASE_URL=postgresql://postgres:travel123@localhost:5432/travel_recommender
```

### 3) Run pipeline
```bash
# First run: fetch from Apify and save into DB
./.venv/bin/python main.py --city "Rome, Italy" --n-users 300

# Next runs: read directly from DB (no additional Apify cost)
./.venv/bin/python main.py --city "Rome, Italy"

# Force refresh after stale data window
./.venv/bin/python main.py --city "Rome, Italy" --force-refresh
```

---

## Domain keys (experience_type)

The runtime pipeline uses English-only domain labels:

`culture`, `nature`, `food`, `shopping`, `nightlife`, `leisure`, `accommodation`, `other`.

---

## Apify cost-saving logic

```
Every run of `main.py`:
    ┌─────────────────────────┐
  │ City data already in DB?│
  │ and fetched < 30 days?  │
    └────────┬────────────────┘
       │ Yes ──→ load from DB
       │ No  ──→ call Apify
       │         └─→ save to DB category-by-category
       │             (safe even if process stops)
```

---

## DB tables

| Tabella | Contenuto |
|---------|-----------|
| `activities` | POIs fetched from Apify |
| `users` | Real + synthetic user profiles |
| `ratings` | User-activity interactions |
| `recommendation_cache` | Cached recommendations (TTL 24h) |
| `apify_fetch_log` | Apify fetch audit log |

---

## Status

- [x] Vector layer (sentence-BERT embeddings su PostgreSQL + pgvector)
- [ ] Itinerary optimizer (TSP con OR-Tools)
- [ ] API FastAPI per esporre il recommender
- [ ] LLM layer per narrazione naturale dell'itinerario

---

## Test

```bash
./.venv/bin/python -m unittest discover -s tests -p 'test_*.py'
```
