"""
Database Manager — Travel Recommender
=======================================
Production-grade persistence layer with PostgreSQL.

Apify cost-saving logic:
    1. Before calling Apify → check if data already exists in DB
    2. If present and fresh (< 30 days) → use DB data
    3. Only if missing → call Apify and save immediately

Tables:
    activities          — POIs fetched from Apify
    users               — User profiles (real + synthetic)
    ratings             — User-activity interactions
    recommendation_cache — Precomputed recommendations (TTL 24h)
    apify_fetch_log     — Apify call and cost log

Local PostgreSQL setup (Docker, zero config):
    docker run --name travel-db -e POSTGRES_PASSWORD=travel123 \
               -e POSTGRES_DB=travel_recommender \
               -p 5432:5432 -d postgres:15

    DATABASE_URL=postgresql://postgres:travel123@localhost:5432/travel_recommender
"""

import os
import json
import hashlib
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional

from sqlalchemy import (
    create_engine, Column, String, Float, Integer,
    Text, DateTime, Boolean, UniqueConstraint,
    inspect, text, Index
)
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy.exc import IntegrityError
from sqlalchemy.dialects.postgresql import insert as pg_insert
try:
    from pgvector.sqlalchemy import Vector
except ModuleNotFoundError:
    from sqlalchemy import Text

    def Vector(_dimension):
        return Text

log = logging.getLogger(__name__)
Base = declarative_base()

# ── Days before Apify data is considered stale
CACHE_TTL_DAYS = 30


# ══════════════════════════════════════════════════════════════════════════════
#  DB MODELS
# ══════════════════════════════════════════════════════════════════════════════

class Activity(Base):
    __tablename__ = "activities"

    activity_id     = Column(String,  primary_key=True)
    place_id        = Column(String,  unique=True, nullable=False, index=True)
    name            = Column(String,  nullable=False)
    category        = Column(String,  index=True)
    experience_type = Column(String,  index=True)
    city            = Column(String,  index=True)
    address         = Column(String)
    lat             = Column(Float)
    lng             = Column(Float)
    rating          = Column(Float)
    review_count    = Column(Integer, default=0)
    price_tier      = Column(Integer)
    phone           = Column(String)
    website         = Column(String)
    description     = Column(Text)
    opening_hours   = Column(Text)    # JSON string
    tags            = Column(Text)    # JSON list
    google_maps_url = Column(String)
    embedding       = Column(Vector(384))
    fetched_at      = Column(DateTime, default=datetime.utcnow)
    updated_at      = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    __table_args__ = (
        Index("ix_activities_city_category", "city", "category"),
    )


class User(Base):
    __tablename__ = "users"

    user_id          = Column(String,  primary_key=True)
    archetype        = Column(String,  index=True)
    preferences_json = Column(Text)
    budget_max       = Column(Integer, default=3)
    trip_days        = Column(Integer, default=3)
    group_size       = Column(Integer, default=2)
    age_range        = Column(String)
    is_synthetic     = Column(Boolean, default=True)
    n_interactions   = Column(Integer, default=0)
    alpha            = Column(Float, default=1.0)
    profile_updated_at = Column(DateTime)
    created_at       = Column(DateTime, default=datetime.utcnow)


class Rating(Base):
    __tablename__ = "ratings"
    __table_args__ = (
        UniqueConstraint("user_id", "activity_id", name="uq_user_activity"),
        Index("ix_ratings_user_id", "user_id"),
        Index("ix_ratings_activity_id", "activity_id"),
    )

    id          = Column(Integer, primary_key=True, autoincrement=True)
    user_id     = Column(String,  nullable=False)
    activity_id = Column(String,  nullable=False)
    rating      = Column(Float)
    event_type  = Column(String,  default="rating")  # rating / visit / skip
    archetype   = Column(String)
    created_at  = Column(DateTime, default=datetime.utcnow)


class RecommendationCache(Base):
    __tablename__ = "recommendation_cache"

    id           = Column(Integer,  primary_key=True, autoincrement=True)
    user_id      = Column(String,   index=True)
    context_hash = Column(String)
    results_json = Column(Text)
    generated_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        UniqueConstraint("user_id", "context_hash", name="uq_cache_user_context"),
    )


class ApifyFetchLog(Base):
    __tablename__ = "apify_fetch_log"

    id           = Column(Integer, primary_key=True, autoincrement=True)
    city         = Column(String,  index=True)
    category     = Column(String)
    n_results    = Column(Integer)
    actor_run_id = Column(String)
    cost_usd     = Column(Float)
    fetched_at   = Column(DateTime, default=datetime.utcnow)


# ══════════════════════════════════════════════════════════════════════════════
#  DATABASE MANAGER
# ══════════════════════════════════════════════════════════════════════════════

class DatabaseManager:
    """
    Main interface for all DB operations.

    Usage:
        db = DatabaseManager()   # reads DATABASE_URL from .env
        db.init_db()
        df = db.get_or_fetch_activities("Rome, Italy", fetcher_fn)
    """

    def __init__(self, db_url: Optional[str] = None):
        url = db_url or os.getenv("DATABASE_URL")
        if not url:
            raise ValueError(
                "DATABASE_URL not found.\n"
                "Add it to .env:\n"
                "DATABASE_URL=postgresql://postgres:travel123@localhost:5432/travel_recommender"
            )
        self.engine  = create_engine(url, pool_pre_ping=True, pool_size=5, max_overflow=10)
        self.Session = sessionmaker(bind=self.engine)
        log.info(f"DB connected: {url.split('@')[-1]}")  # password not logged

    def init_db(self):
        """Creates all tables if they do not exist. Idempotent."""
        Base.metadata.create_all(self.engine)
        self._ensure_postgres_vector_layer()
        log.info("DB schema initialized.")
        self._print_summary()

    def _ensure_postgres_vector_layer(self):
        """Idempotent migration for pgvector + user profile columns."""
        with self.engine.begin() as conn:
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            conn.execute(text("ALTER TABLE activities ADD COLUMN IF NOT EXISTS embedding vector(384)"))
            conn.execute(text(
                """
                CREATE INDEX IF NOT EXISTS ix_activities_embedding_ivfflat
                ON activities
                USING ivfflat (embedding vector_cosine_ops)
                WITH (lists = 100)
                """
            ))
            conn.execute(text("ALTER TABLE users ADD COLUMN IF NOT EXISTS n_interactions INTEGER DEFAULT 0"))
            conn.execute(text("ALTER TABLE users ADD COLUMN IF NOT EXISTS alpha FLOAT DEFAULT 1.0"))
            conn.execute(text("ALTER TABLE users ADD COLUMN IF NOT EXISTS profile_updated_at TIMESTAMP"))

    # ── MAIN PATTERN: get-or-fetch ──────────────────────────────────────────

    def get_or_fetch_activities(
        self,
        city: str,
        fetcher_fn,                    # ApifyCollector.fetch_city function
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        """
        Core pattern: check DB first, then call Apify only if needed.

            ┌─────────────────┐
                 │ Data in DB?     │
            │ e < 30 giorni?  │
            └──────┬──────────┘
                     │ Yes → return from DB 💰 zero cost
                     │ No  → call Apify → save DB → return
                   └──────────────────────────────────────

        Args:
            city:          E.g. "Rome, Italy"
            fetcher_fn:    ApifyCollector(token).fetch_city  (callable)
            force_refresh: True = re-fetch even if data exists
        """
        if not force_refresh:
            fresh = self._load_fresh_activities(city)
            if fresh is not None:
                n = len(fresh)
                print(f"✅ '{city}': {n} POIs loaded from DB. No Apify call. 💰")
                return fresh

        # Missing or stale data → call Apify
        print(f"🌐 '{city}' not in DB (or stale). Calling Apify...")
        df = fetcher_fn(city=city)

        if not df.empty:
            n_new = self.save_activities(df, city=city)
            print(f"💾 {n_new} new POIs saved to DB for '{city}'.")

        return df

    # ── ACTIVITIES ────────────────────────────────────────────────────────────

    def save_activities(self, df: pd.DataFrame, city: str = None) -> int:
        """
        Saves POIs in DB. Uses UPSERT on place_id:
        - if POI already exists → update rating and review_count
        - if new → insert

        Returns: number of inserted/updated rows
        """
        if df.empty:
            return 0

        if city:
            df = df.copy()
            df["city"] = df["city"].fillna(city)

        records = df.to_dict("records")
        count   = 0

        with self.Session() as session:
            for row in records:
                stmt = pg_insert(Activity).values(
                    activity_id     = row.get("activity_id"),
                    place_id        = row.get("place_id") or row.get("activity_id"),
                    name            = row.get("name", ""),
                    category        = row.get("category"),
                    experience_type = row.get("experience_type"),
                    city            = row.get("city", city),
                    address         = row.get("address"),
                    lat             = row.get("lat"),
                    lng             = row.get("lng"),
                    rating          = row.get("rating"),
                    review_count    = int(row.get("review_count") or 0),
                    price_tier      = int(row.get("price_tier") or 2),
                    phone           = row.get("phone"),
                    website         = row.get("website"),
                    description     = row.get("description"),
                    opening_hours   = row.get("opening_hours"),
                    tags            = row.get("tags"),
                    google_maps_url = row.get("google_maps_url"),
                    fetched_at      = datetime.utcnow(),
                    updated_at      = datetime.utcnow(),
                ).on_conflict_do_update(
                    index_elements=["place_id"],
                    set_={
                        "rating":       row.get("rating"),
                        "review_count": int(row.get("review_count") or 0),
                        "updated_at":   datetime.utcnow(),
                    }
                )
                session.execute(stmt)
                count += 1
            session.commit()

        log.info(f"Upsert completed: {count} activities for '{city}'.")
        return count

    def load_activities(
        self,
        city: Optional[str]            = None,
        category: Optional[str]        = None,
        experience_type: Optional[str] = None,
        min_rating: float              = 0.0,
        price_tier_max: int            = 4,
    ) -> pd.DataFrame:
        """Loads activities from DB with filters. Never calls Apify."""
        with self.Session() as session:
            q = session.query(Activity)
            if city:
                q = q.filter(Activity.city.ilike(f"%{city}%"))
            if category:
                q = q.filter(Activity.category == category)
            if experience_type:
                q = q.filter(Activity.experience_type == experience_type)
            if min_rating > 0:
                q = q.filter(Activity.rating >= min_rating)
            if price_tier_max < 4:
                q = q.filter(Activity.price_tier <= price_tier_max)
            rows = q.all()

        if not rows:
            return pd.DataFrame()

        return pd.DataFrame([
            {c.key: getattr(r, c.key) for c in Activity.__table__.columns}
            for r in rows
        ])

    def _load_fresh_activities(self, city: str) -> Optional[pd.DataFrame]:
        """Returns DF if city data exists in DB and is fresh, else None."""
        with self.Session() as session:
            cutoff = datetime.utcnow() - timedelta(days=CACHE_TTL_DAYS)
            count  = session.query(Activity).filter(
                Activity.city.ilike(f"%{city}%"),
                Activity.fetched_at >= cutoff,
            ).count()

        if count == 0:
            return None
        return self.load_activities(city=city)

    # ── USERS ────────────────────────────────────────────────────────────────

    def save_users(self, df: pd.DataFrame) -> int:
        """Saves users with INSERT OR IGNORE on user_id."""
        count = 0
        with self.Session() as session:
            for _, row in df.iterrows():
                stmt = pg_insert(User).values(
                    user_id          = row["user_id"],
                    archetype        = row.get("archetype"),
                    preferences_json = row.get("preferences_json"),
                    budget_max       = int(row.get("budget_max") or 3),
                    trip_days        = int(row.get("trip_days") or 3),
                    group_size       = int(row.get("group_size") or 2),
                    age_range        = row.get("age_range"),
                    is_synthetic     = bool(row.get("is_synthetic", True)),
                    n_interactions   = int(row.get("n_interactions") or 0),
                    alpha            = float(row.get("alpha") or 1.0),
                    profile_updated_at = row.get("profile_updated_at"),
                ).on_conflict_do_nothing(index_elements=["user_id"])
                session.execute(stmt)
                count += 1
            session.commit()
        log.info(f"Users saved: {count}.")
        return count

    def load_users(self) -> pd.DataFrame:
        with self.Session() as session:
            rows = session.query(User).all()
        if not rows:
            return pd.DataFrame()
        return pd.DataFrame([
            {c.key: getattr(r, c.key) for c in User.__table__.columns}
            for r in rows
        ])

    # ── RATINGS ──────────────────────────────────────────────────────────────

    def save_ratings(self, df: pd.DataFrame) -> int:
        """Saves ratings with UPSERT: updates if (user, activity) already exists."""
        count = 0
        with self.Session() as session:
            for _, row in df.iterrows():
                stmt = pg_insert(Rating).values(
                    user_id     = row["user_id"],
                    activity_id = row["activity_id"],
                    rating      = float(row.get("rating") or 0),
                    event_type  = row.get("event_type", "rating"),
                    archetype   = row.get("archetype"),
                ).on_conflict_do_update(
                    constraint="uq_user_activity",
                    set_={"rating": float(row.get("rating") or 0)}
                )
                session.execute(stmt)
                count += 1
            session.commit()
        log.info(f"Ratings saved: {count}.")
        return count

    def load_ratings(self, user_id: Optional[str] = None) -> pd.DataFrame:
        with self.Session() as session:
            q = session.query(Rating)
            if user_id:
                q = q.filter(Rating.user_id == user_id)
            rows = q.all()
        if not rows:
            return pd.DataFrame()
        return pd.DataFrame([
            {c.key: getattr(r, c.key) for c in Rating.__table__.columns}
            for r in rows
        ])

    def add_interaction(
        self, user_id: str, activity_id: str,
        rating: float, event_type: str = "rating"
    ):
        """Adds/updates a single real-time interaction."""
        with self.Session() as session:
            stmt = pg_insert(Rating).values(
                user_id=user_id, activity_id=activity_id,
                rating=rating, event_type=event_type
            ).on_conflict_do_update(
                constraint="uq_user_activity",
                set_={"rating": rating, "event_type": event_type}
            )
            session.execute(stmt)

            user = session.query(User).filter(User.user_id == user_id).first()
            if user is not None:
                new_interactions = int((user.n_interactions or 0) + 1)
                user.n_interactions = new_interactions
                user.alpha = self._adaptive_alpha(new_interactions)
                user.profile_updated_at = datetime.utcnow()

            session.commit()
        log.info(f"Interaction: {user_id} → {activity_id} ({rating}⭐)")

    def get_user_alpha(self, user_id: str) -> float:
        """Returns user alpha (with adaptive fallback)."""
        with self.Session() as session:
            user = session.query(User).filter(User.user_id == user_id).first()
        if user is None:
            return 1.0
        if user.alpha is not None:
            return float(user.alpha)
        return self._adaptive_alpha(int(user.n_interactions or 0))

    @staticmethod
    def _adaptive_alpha(n_interactions: int) -> float:
        return max(0.20, 1.0 * (0.92 ** max(0, int(n_interactions))))

    def city_already_fetched(self, city: str) -> bool:
        """True if city has fresh POIs in DB according to CACHE_TTL_DAYS."""
        return self._load_fresh_activities(city) is not None

    def save_activity_embeddings(self, embeddings_df: pd.DataFrame) -> int:
        """Saves 384d embeddings into activities.embedding by activity_id."""
        if embeddings_df.empty:
            return 0

        updated = 0
        with self.engine.begin() as conn:
            for _, row in embeddings_df.iterrows():
                emb = row.get("embedding")
                if emb is None:
                    continue
                vector_text = self._to_pgvector_literal(emb)
                result = conn.execute(
                    text(
                        """
                        UPDATE activities
                        SET embedding = CAST(:embedding AS vector),
                            updated_at = :updated_at
                        WHERE activity_id = :activity_id
                        """
                    ),
                    {
                        "embedding": vector_text,
                        "updated_at": datetime.utcnow(),
                        "activity_id": str(row["activity_id"]),
                    },
                )
                updated += int(result.rowcount or 0)

        log.info("Embeddings saved/updated: %s", updated)
        return updated

    def search_by_embedding(
        self,
        query_embedding,
        city: Optional[str] = None,
        top_k: int = 200,
    ) -> pd.DataFrame:
        """ANN search with pgvector `<=>` and normalized vector_score output."""
        query_vector = self._to_pgvector_literal(query_embedding)

        sql = """
            SELECT
                activity_id,
                (embedding <=> CAST(:query_embedding AS vector)) AS vector_distance,
                GREATEST(0.0, 1.0 - (embedding <=> CAST(:query_embedding AS vector))) AS vector_score
            FROM activities
            WHERE embedding IS NOT NULL
        """
        params = {"query_embedding": query_vector, "top_k": int(top_k)}

        if city:
            sql += " AND city ILIKE :city "
            params["city"] = f"%{city}%"

        sql += " ORDER BY embedding <=> CAST(:query_embedding AS vector) LIMIT :top_k"

        with self.engine.begin() as conn:
            rows = conn.execute(text(sql), params).fetchall()

        if not rows:
            return pd.DataFrame(columns=["activity_id", "vector_distance", "vector_score"])
        return pd.DataFrame(rows, columns=["activity_id", "vector_distance", "vector_score"])

    @staticmethod
    def _to_pgvector_literal(vector) -> str:
        arr = np.asarray(vector, dtype=float).flatten()
        return "[" + ",".join(f"{x:.8f}" for x in arr.tolist()) + "]"

    # ── RECOMMENDATION CACHE ────────────────────────────────────────────────

    def get_cached_recommendations(
        self, user_id: str, context: dict, ttl_hours: int = 24
    ) -> Optional[list]:
        """
        Returns recommendations from cache if age < ttl_hours.
        Avoids recomputing recommender on every request.
        """
        ctx_hash = hashlib.md5(json.dumps(context, sort_keys=True).encode()).hexdigest()
        cutoff   = datetime.utcnow() - timedelta(hours=ttl_hours)

        with self.Session() as session:
            row = session.query(RecommendationCache).filter(
                RecommendationCache.user_id      == user_id,
                RecommendationCache.context_hash == ctx_hash,
                RecommendationCache.generated_at >= cutoff,
            ).first()

        if row:
            log.info(f"Cache HIT for {user_id}")
            return json.loads(row.results_json)
        return None

    def cache_recommendations(self, user_id: str, context: dict, results: list):
        """Saves recommendations to cache."""
        ctx_hash = hashlib.md5(json.dumps(context, sort_keys=True).encode()).hexdigest()
        with self.Session() as session:
            stmt = pg_insert(RecommendationCache).values(
                user_id      = user_id,
                context_hash = ctx_hash,
                results_json = json.dumps(results),
                generated_at = datetime.utcnow(),
            ).on_conflict_do_update(
                constraint="uq_cache_user_context",
                set_={"results_json": json.dumps(results), "generated_at": datetime.utcnow()}
            )
            session.execute(stmt)
            session.commit()

    # ── APIFY FETCH LOG ─────────────────────────────────────────────────────

    def log_apify_fetch(
        self, city: str, category: str,
        n_results: int, actor_run_id: str = None, cost_usd: float = None
    ):
        with self.Session() as session:
            session.add(ApifyFetchLog(
                city=city, category=category, n_results=n_results,
                actor_run_id=actor_run_id, cost_usd=cost_usd
            ))
            session.commit()

    def get_apify_cost_report(self) -> pd.DataFrame:
        """Apify cost report by city."""
        with self.Session() as session:
            rows = session.execute(text("""
                SELECT city,
                       COUNT(*)        AS fetch_calls,
                       SUM(n_results)  AS total_poi,
                       SUM(cost_usd)   AS total_usd,
                       MIN(fetched_at) AS first_fetch,
                       MAX(fetched_at) AS last_fetch
                FROM apify_fetch_log
                GROUP BY city ORDER BY total_usd DESC
            """)).fetchall()
        cols = ["city","fetch_calls","total_poi","total_usd","first_fetch","last_fetch"]
        return pd.DataFrame(rows, columns=cols)

    # ── STATS ───────────────────────────────────────────────────────────────

    def stats(self) -> dict:
        with self.Session() as session:
            cities = session.execute(text(
                "SELECT city, COUNT(*) n FROM activities GROUP BY city ORDER BY n DESC"
            )).fetchall()
            return {
                "activities": session.query(Activity).count(),
                "users":      session.query(User).count(),
                "ratings":    session.query(Rating).count(),
                "cities":     [{"city": r[0], "n_poi": r[1]} for r in cities],
            }

    def _print_summary(self):
        inspector = inspect(self.engine)
        tables    = inspector.get_table_names()
        print(f"\n📦 PostgreSQL ready — {len(tables)} tables:")
        for t in tables:
            cols = [c["name"] for c in inspector.get_columns(t)]
            print(f"   └─ {t:30s} {len(cols)} columns")
        print()
