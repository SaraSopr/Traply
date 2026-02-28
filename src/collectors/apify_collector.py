"""
Apify Google Maps Collector
============================
Fetches POIs (Points of Interest) from Google Maps via Apify.

Cost-saving logic:
    - If a DatabaseManager is provided, each batch is saved to DB
        IMMEDIATELY after every fetched category.
    - If the process stops midway, already downloaded data
        remains safe in DB and is not re-fetched.

Usage:
    db        = DatabaseManager()
    collector = ApifyCollector(api_token="YOUR_TOKEN")

    # Production mode: fetch + automatic DB persistence
    df = collector.fetch_city("Rome, Italy", db=db)
"""

import os
import time
import json
import logging
import pandas as pd
from typing import Optional
from apify_client import ApifyClient

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


DEFAULT_CATEGORIES = [
    "tourist attraction", "museum", "restaurant", "cafe", "bar",
    "park", "art gallery", "historic site", "shopping mall", "market",
    "night club", "hotel", "theater", "zoo", "beach",
]

CATEGORY_TO_EXPERIENCE = {
    "museum": "cultura", "art gallery": "cultura",
    "historic site": "cultura", "theater": "cultura",
    "tourist attraction": "svago", "park": "natura",
    "beach": "natura", "zoo": "natura",
    "restaurant": "cibo", "cafe": "cibo", "bar": "cibo", "market": "cibo",
    "shopping mall": "shopping", "night club": "vita_notturna", "hotel": "alloggio",
}


class ApifyCollector:
    ACTOR_ID = "compass/crawler-google-places"

    def __init__(self, api_token: Optional[str] = None):
        token = api_token or os.getenv("APIFY_API_TOKEN")
        if not token:
            raise ValueError("Missing APIFY_API_TOKEN.")
        self.client = ApifyClient(token)
        log.info("ApifyClient initialized.")

    def fetch_city(self, city, categories=None, max_per_category=100,
                   language="it", db=None):
        """
        Fetches POIs for a city. If db is provided, it saves to DB
        after each category — data is safe even in case of crash.
        """
        categories  = categories or DEFAULT_CATEGORIES
        all_results = []

        log.info(f"Fetch '{city}' — {len(categories)} categories")

        for i, category in enumerate(categories):
            log.info(f"[{i+1}/{len(categories)}] {category}...")
            try:
                results = self._run_actor(city, category, max_per_category, language)
                all_results.extend(results)
                log.info(f"  → {len(results)} results")

                # 💾 Save immediately to DB after each category
                if db is not None and results:
                    batch_df = self._normalize(results)
                    if not batch_df.empty:
                        n = db.save_activities(batch_df, city=city)
                        db.log_apify_fetch(city=city, category=category, n_results=len(results))
                        log.info(f"  💾 {n} POIs saved to DB")

                time.sleep(1)
            except Exception as e:
                log.warning(f"  Error in '{category}': {e}")
                continue

        if not all_results:
            return pd.DataFrame()

        return self._normalize(all_results)

    def _run_actor(self, city, category, max_results, language):
        run = self.client.actor(self.ACTOR_ID).call(run_input={
            "searchStringsArray": [f"{category} in {city}"],
            "maxCrawledPlacesPerSearch": max_results,
            "language": language,
            "includeReviews": False,
            "includeImages": False,
            "additionalInfo": True,
        })
        items = list(self.client.dataset(run["defaultDatasetId"]).iterate_items())
        for item in items:
            item["_search_category"] = category
        return items

    def _normalize(self, raw_items):
        rows = []
        for item in raw_items:
            try:
                rows.append(self._extract_fields(item))
            except Exception:
                continue

        df = pd.DataFrame(rows)
        if df.empty:
            return df

        df = df.drop_duplicates(subset=["place_id"]).reset_index(drop=True)
        df.insert(0, "activity_id", [f"ACT_{i:04d}" for i in range(len(df))])
        return df.dropna(subset=["lat", "lng"])

    def _extract_fields(self, item):
        category = item.get("_search_category", "unknown")
        location = item.get("location", {}) or {}
        hours    = item.get("openingHours", []) or []
        return {
            "place_id":         item.get("placeId") or item.get("id"),
            "name":             item.get("title") or item.get("name"),
            "category":         category,
            "experience_type":  CATEGORY_TO_EXPERIENCE.get(category, "altro"),
            "address":          item.get("address"),
            "city":             item.get("city"),
            "lat":              location.get("lat") or item.get("lat"),
            "lng":              location.get("lng") or item.get("lng"),
            "rating":           item.get("totalScore") or item.get("rating"),
            "review_count":     item.get("reviewsCount") or 0,
            "price_tier":       self._parse_price(item.get("priceLevel")),
            "phone":            item.get("phone"),
            "website":          item.get("website"),
            "description":      item.get("description") or item.get("editorialSummary"),
            "opening_hours":    json.dumps(hours) if hours else None,
            "google_maps_url":  item.get("url"),
            "tags":             json.dumps(item.get("categories", [])),
        }

    @staticmethod
    def _parse_price(price_level):
        if price_level is None:
            return None
        mapping = {
            "PRICE_LEVEL_FREE": 1, "PRICE_LEVEL_INEXPENSIVE": 1,
            "PRICE_LEVEL_MODERATE": 2, "PRICE_LEVEL_EXPENSIVE": 3,
            "PRICE_LEVEL_VERY_EXPENSIVE": 4,
        }
        if isinstance(price_level, str):
            return mapping.get(price_level, 2)
        if isinstance(price_level, int):
            return min(max(price_level, 1), 4)
        return 2
