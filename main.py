"""
Main Pipeline — Travel Recommender
==================================
End-to-end flow with PostgreSQL:

    1. Init DB
    2. Check whether city data is already in DB
         → Yes: load from DB (zero Apify cost) 💰
         → No: call Apify and save immediately to DB
    3. Generate synthetic users (if not already in DB)
    4. Generate synthetic ratings
    5. Train the Hybrid Recommender
    6. Recommend + evaluate

Setup:
    Add to .env:
        APIFY_API_TOKEN=apify_api_xxxx
        DATABASE_URL=postgresql://postgres:travel123@localhost:5432/travel_recommender

    Start PostgreSQL with Docker (one line):
        docker run --name travel-db -e POSTGRES_PASSWORD=travel123 \
                             -e POSTGRES_DB=travel_recommender \
                             -p 5432:5432 -d postgres:15
"""

import os
import argparse
import logging
import pandas as pd
from dotenv import load_dotenv

from src.collectors.apify_collector import ApifyCollector, DEFAULT_CATEGORIES
from src.collectors.synthetic_users import SyntheticUserGenerator
from src.recommender.hybrid_recommender import HybridRecommender
from src.utils.database import DatabaseManager

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def run_pipeline(city: str, n_users: int = 300, top_n: int = 10,
                 force_refresh: bool = False):

    print(f"\n{'='*60}")
    print(f"  Travel Recommender — Production Pipeline")
    print(f"  City: {city} | Users: {n_users}")
    print(f"{'='*60}\n")

    # ── STEP 1: Init DB ──────────────────────────────────────────────────────
    db = DatabaseManager()
    db.init_db()

    # ── STEP 2: Activities — DB first, Apify only if needed ────────────────
    print("📍 Step 1/4 — Loading activities...")

    if not force_refresh and db.city_already_fetched(city):
        # ✅ Data already in DB — zero Apify calls
        activities_df = db.load_activities(city=city)
        print(f"✅ Loaded {len(activities_df)} POIs from DB (no Apify call) 💰\n")
    else:
        # 🌐 First run or forced refresh — call Apify and save to DB
        print("🌐 Fetching from Apify (one-time)...")
        token = os.getenv("APIFY_API_TOKEN")
        if not token:
            raise ValueError("APIFY_API_TOKEN not found in .env")

        collector     = ApifyCollector(api_token=token)
        activities_df = collector.fetch_city(
            city=city,
            categories=DEFAULT_CATEGORIES,
            max_per_category=80,
            db=db,
        )
        print(f"✅ {len(activities_df)} POIs fetched and saved to DB\n")

    if activities_df.empty:
        print("❌ No activities found. Check city and token.")
        return

    # ── STEP 3: Synthetic users ─────────────────────────────────────────────
    print("👥 Step 2/4 — Preparing synthetic users...")

    users_df = db.load_users()
    if len(users_df) < n_users:
        gen      = SyntheticUserGenerator(seed=42)
        users_df = gen.generate_users(n=n_users)
        db.save_users(users_df)
        print(f"✅ {len(users_df)} synthetic users saved to DB\n")
    else:
        print(f"✅ {len(users_df)} users already in DB\n")

    # ── STEP 4: Synthetic ratings ───────────────────────────────────────────
    print("⭐ Step 3/4 — Preparing ratings...")

    ratings_df = db.load_ratings()
    if ratings_df.empty:
        gen        = SyntheticUserGenerator(seed=42)
        ratings_df = gen.generate_ratings(users_df, activities_df, n_ratings_per_user=15)
        db.save_ratings(ratings_df)
        print(f"✅ {len(ratings_df)} ratings saved to DB\n")
    else:
        print(f"✅ {len(ratings_df)} ratings already in DB\n")

    # ── STEP 5: Training ────────────────────────────────────────────────────
    print("🤖 Step 4/4 — Training Hybrid Recommender...")
    split      = int(len(ratings_df) * 0.8)
    train_r    = ratings_df.iloc[:split]
    test_r     = ratings_df.iloc[split:]

    rec = HybridRecommender(alpha=0.5)
    rec.fit(activities_df, users_df, train_r, db=db)

    # ── STEP 6: Demo + evaluation ───────────────────────────────────────────
    print(f"\n{'─'*60}")
    print("  RECOMMENDATION DEMO")
    print(f"{'─'*60}")

    for _, user in users_df.sample(3, random_state=1).iterrows():
        uid     = user["user_id"]
        context = {"budget_max": user["budget_max"], "city": city}

        # Check cache first
        cached = db.get_cached_recommendations(uid, context)
        if cached:
            print(f"\n👤 {uid} [{user['archetype']}] — from cache")
            continue

        results = rec.recommend(uid, top_n=top_n, context=context, db=db)
        db.cache_recommendations(uid, context, results["activity_id"].tolist())
        print(f"\n👤 {uid} [{user['archetype']}]")
        print(results[["name","category","rating","price_tier","hybrid_score"]].to_string(index=False))

    # Metrics
    print(f"\n{'─'*60}")
    for k in [5, 10]:
        m = rec.evaluate(test_r, k=k)
        print(f"  Precision@{k}: {m['precision_at_k']:.4f} | NDCG@{k}: {m['ndcg_at_k']:.4f}")

    # Apify cost report
    print(f"\n{'─'*60}")
    print("  APIFY COST REPORT")
    cost_report = db.get_apify_cost_report()
    if not cost_report.empty:
        print(cost_report.to_string(index=False))

    # DB stats
    s = db.stats()
    print(f"\n📦 DB Stats: {s['activities']} POIs | {s['users']} users | {s['ratings']} ratings")
    print(f"   Cities: {[c['city'] for c in s['cities']]}")
    print(f"\n{'='*60}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--city",          type=str, default="Rome, Italy")
    parser.add_argument("--n-users",       type=int, default=300)
    parser.add_argument("--top-n",         type=int, default=10)
    parser.add_argument("--force-refresh", action="store_true",
                        help="Re-fetch from Apify even if data is already in DB")
    args = parser.parse_args()

    run_pipeline(city=args.city, n_users=args.n_users,
                 top_n=args.top_n, force_refresh=args.force_refresh)
