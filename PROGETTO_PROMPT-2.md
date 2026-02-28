# PROMPT — Travel Recommender System (Master's Thesis)

## Language instruction
**Always respond in Italian**, regardless of the language used in this prompt.

---

## Project context

I am developing a **hybrid recommendation system for personalized travel itineraries** as a master's thesis project, with the goal of evolving it into a startup.

The system is called **Wandr.AI** and generates optimized daily itineraries based on user preferences, temporal context, and real place data.

---

## Technology stack (DO NOT change)

- **Language**: Python 3.11+
- **Database**: PostgreSQL (single system — no MongoDB, Redis, SQLite)
- **ORM**: SQLAlchemy 2.0 with `pg_insert` for UPSERT
- **POI data**: Apify SDK (`compass/crawler-google-places`)
- **ML**: scikit-learn, scipy (SVD), numpy
- **Config**: python-dotenv (`.env` file)

---

## System architecture (already designed)

```
travel_recommender/
├── main.py
├── .env                          # APIFY_API_TOKEN, DATABASE_URL
├── requirements.txt
└── src/
    ├── collectors/
    │   ├── apify_collector.py    # Fetch POI from Google Maps via Apify
    │   └── synthetic_users.py   # Generate synthetic users and ratings
    ├── recommender/
    │   ├── hybrid_recommender.py # CB + CF + Time-Aware ensemble
    │   └── time_aware.py        # Time-Aware Recommendation layer
    └── utils/
        └── database.py          # Full PostgreSQL manager
```

---

## PostgreSQL DB schema (already defined)

### Table `users`
```sql
user_id            TEXT PRIMARY KEY
archetype          TEXT               -- family / couple / solo / group
preferences_json   JSONB              -- {cultura: 0.8, cibo: 0.7, ...}
budget_max         INTEGER DEFAULT 3  -- scale 1-4
trip_days          INTEGER DEFAULT 3
group_size         INTEGER DEFAULT 2
age_range          TEXT
is_synthetic       BOOLEAN DEFAULT TRUE
n_interactions     INTEGER DEFAULT 0  -- interaction counter
alpha              FLOAT   DEFAULT 1.0 -- CB vs CF weight (progressive profiling)
profile_updated_at TIMESTAMP
created_at         TIMESTAMP
```

### Table `activities`
```sql
activity_id     TEXT PRIMARY KEY
place_id        TEXT UNIQUE        -- Google Maps ID (from Apify)
name            TEXT
category        TEXT               -- museum / restaurant / park / ...
experience_type TEXT               -- cultura / cibo / natura / shopping / vita_notturna / svago / alloggio
city            TEXT
address         TEXT
lat             FLOAT
lng             FLOAT
rating          FLOAT
review_count    INTEGER
price_tier      INTEGER            -- 1 (free) → 4 (luxury)
phone           TEXT
website         TEXT
description     TEXT
opening_hours   TEXT               -- JSON string
tags            TEXT               -- JSON list
google_maps_url TEXT
fetched_at      TIMESTAMP
updated_at      TIMESTAMP
```

### Table `ratings`
```sql
id          SERIAL PRIMARY KEY
user_id     TEXT
activity_id TEXT
rating      FLOAT                  -- 1.0 → 5.0
event_type  TEXT DEFAULT 'rating'  -- rating / visit / skip
archetype   TEXT
created_at  TIMESTAMP
UNIQUE(user_id, activity_id)
```

### Table `recommendation_cache`
```sql
id           SERIAL PRIMARY KEY
user_id      TEXT
context_hash TEXT                   -- MD5 of context parameters
results_json TEXT                   -- JSON list of ordered activity_ids
generated_at TIMESTAMP
UNIQUE(user_id, context_hash)
```

### Table `apify_fetch_log`
```sql
id           SERIAL PRIMARY KEY
city         TEXT
category     TEXT
n_results    INTEGER
actor_run_id TEXT
cost_usd     FLOAT
fetched_at   TIMESTAMP
```

---

## Core logic already implemented

### 1. Hybrid recommender — formula

```
score_finale(u, i, t) = α·CB(u,i) + (1-α)·CF(u,i) + γ·Time(i,t)

where:
  α    = user's adaptive alpha (read from DB)
  γ    = 0.25 (time-aware weight, configurable)
  CB   = cosine similarity TF-IDF (features: category, experience_type, tags, description)
  CF   = SVD matrix factorization (scipy.sparse.linalg.svds, k=20 latent factors)
  Time = TimeAwareScorer (slot × season × weekend boost)
```

### 2. Progressive profiling — adaptive alpha

```python
# Alpha formula: damped exponential decay
alpha = max(0.20, 1.0 * (0.92 ** n_interactions))

# n=0   → α=1.00  (cold start, CB only)
# n=5   → α=0.66
# n=10  → α=0.43
# n=20  → α=0.19  (CF dominant)
# n=50+ → α=0.20  (floor)
```

### 3. Profile update — EMA (Exponential Moving Average)

```python
# On every new rating:
learning_rate = 0.3
feedback = rating / 5.0
prefs[experience_type] = (1 - 0.3) * old_value + 0.3 * feedback
```

### 4. Temporal context

```python
# Daily time slots:
# 06-11 → morning | 12-17 → afternoon | 18-22 → evening | 23-05 → night

# Category × slot compatibility prior (grounded in tourism literature):
TIME_PRIOR = {
    "cultura":       [0.90, 0.75, 0.35, 0.05],  # [morn, aftn, even, nght]
    "natura":        [0.85, 0.90, 0.60, 0.10],
    "cibo":          [0.55, 0.75, 0.95, 0.65],
    "shopping":      [0.50, 0.95, 0.55, 0.05],
    "vita_notturna": [0.05, 0.15, 0.80, 1.00],
    "svago":         [0.70, 0.85, 0.65, 0.20],
}
# Prior is updated with learn_from_interactions() → learned model
```

### 5. Apify cost saving logic

```python
# Before calling Apify → check DB
if db.city_already_fetched(city):
    return db.load_activities(city)    # zero cost
else:
    df = apify.fetch_city(city, db=db) # fetch AND save in real-time per category
```

### 6. User archetypes

| Archetype | Avg budget | Strong preferences    | Weak preferences |
|-----------|------------|-----------------------|------------------|
| family    | 1-2        | natura, svago         | vita_notturna    |
| couple    | 2-3        | cibo, cultura         | —                |
| solo      | 1-2        | cultura, svago        | shopping         |
| group     | 2-3        | vita_notturna, svago  | cultura          |

---

## Development roadmap

### ✅ COMPLETED
- [x] `apify_collector.py` — fetch + real-time DB save per category
- [x] `synthetic_users.py` — user and rating generation with archetypes
- [x] `hybrid_recommender.py` — CB + CF + Time-Aware
- [x] `time_aware.py` — TemporalContext + TimeAwareScorer + learn_from_interactions
- [x] `database.py` — PostgreSQL, 5 tables, UPSERT, progressive profiling
- [x] `main.py` — end-to-end pipeline

### ❌ TO IMPLEMENT (in priority order)

**1. Vector Layer** — pgvector + sentence-BERT embeddings
- Add column `embedding vector(384)` to `activities` table
- Generate embeddings with `sentence-transformers` (model: `paraphrase-multilingual-MiniLM-L12-v2`)
- ANN search with pgvector `<=>` operator
- Integrate as 4th component: `score = α·CB + β·CF + γ·Time + δ·Vector`

**2. Cold Start module**
- Onboarding with 6-8 probe activities (Active Learning)
- Archetype default fallback
- Measurable learning curve

**3. Evaluation module**
- Precision@K, NDCG@K, Coverage, Diversity
- Learning curve (X = n_interactions, Y = NDCG@10)
- Baseline comparison (popularity) vs system
- Thesis-ready plots

---

## Environment setup (run once)

```bash
# 1. PostgreSQL via Docker
docker run --name travel-db \
  -e POSTGRES_PASSWORD=travel123 \
  -e POSTGRES_DB=travel_recommender \
  -p 5432:5432 -d postgres:15

# 2. Install dependencies
pip install -r requirements.txt

# 3. .env file
APIFY_API_TOKEN=apify_api_YOUR_TOKEN
DATABASE_URL=postgresql://postgres:travel123@localhost:5432/travel_recommender

# 4. First run
python main.py --city "YOUR CITY" --n-users 300
```

---

## Requirements

```
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
scipy>=1.11.0
sqlalchemy>=2.0.0
psycopg2-binary>=2.9.0
apify-client>=1.6.0
python-dotenv>=1.0.0
sentence-transformers>=2.2.0   # for Vector Layer (next step)
tqdm>=4.66.0
jupyter>=1.0.0
matplotlib>=3.7.0
seaborn>=0.12.0
```

---

## Theoretical references (for thesis)

- **Hybrid RS**: Burke (2002) — "Hybrid Recommender Systems: Survey and Experiments"
- **Matrix Factorization**: Koren et al. (2009) — "Matrix Factorization Techniques for Recommender Systems"
- **Context-Aware RS**: Adomavicius & Tuzhilin (2015) — "Context-Aware Recommender Systems"
- **Time-Aware**: Baltrunas et al. (2011) — "Context-Aware Matrix Factorization for Recommender Systems"
- **Cold Start**: Schein et al. (2002) — "Methods and Metrics for Cold-Start Recommendations"
- **EMA profiling**: standard signal processing applied to user modeling

---

## Instructions for the AI

When receiving this prompt:

1. **Always respond in Italian** — this is mandatory regardless of the prompt language
2. **Do not reinvent the architecture** — it is already defined, follow it
3. **Always use PostgreSQL** — no SQLite, Redis, or MongoDB
4. **Keep naming conventions consistent** — `experience_type`, `hybrid_score`, `alpha`, `time_score`
5. **Every new module** must have a docstring with a citable theoretical reference
6. **The next step to implement** is the Vector Layer with pgvector
7. **Before writing code** briefly explain the theory behind the component being added
8. **Files always go in** `src/recommender/`, `src/utils/`, or `src/collectors/`
9. **Do not simplify** the architecture — maintain professional production-grade quality
