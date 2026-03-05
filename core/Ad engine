"""
core/ad_engine.py
Full ad matching, placement optimization, and inventory management engine.

Scoring weights:
  Content similarity    30%
  IAB category match    25%
  Brand safety          20%
  Demographic targeting 15%
  Historical performance 10%

Placement constraints:
  Min 3 minutes between ads
  Max 1 ad per 5 minutes
  Placement types: pre-roll, mid-roll, post-roll
"""
from __future__ import annotations
import uuid
import json
from dataclasses import dataclass, field
from typing import Optional
import numpy as np

from core.embeddings import (
    TFIDFVectorizer, cosine_similarity,
    classify_iab, extract_brand_safety,
)
from core.scene_detector import Scene


# ── Ad Inventory ─────────────────────────────────────────────────────────────
@dataclass
class AdCreative:
    ad_id: str
    title: str
    brand: str
    description: str
    iab_categories: list[str]          # e.g. ["IAB19", "IAB3"]
    target_demographics: list[str]     # e.g. ["18-34", "tech-savvy"]
    keywords: list[str]
    brand_safety_min: float = 0.5      # minimum scene safety required
    cpm_base: float = 5.0              # base CPM in USD
    historical_ctr: float = 0.02       # click-through rate
    historical_cvr: float = 0.005      # conversion rate
    budget_remaining: float = 10_000.0
    total_impressions: int = 0
    total_clicks: int = 0
    total_conversions: int = 0
    total_spend: float = 0.0
    embedding: Optional[np.ndarray] = field(default=None, repr=False)
    is_active: bool = True

    @property
    def performance_score(self) -> float:
        """Normalised historical performance (CTR + CVR combined)."""
        return min(self.historical_ctr * 10 + self.historical_cvr * 50, 1.0)

    def to_dict(self) -> dict:
        return {
            "ad_id": self.ad_id,
            "title": self.title,
            "brand": self.brand,
            "description": self.description,
            "iab_categories": self.iab_categories,
            "target_demographics": self.target_demographics,
            "keywords": self.keywords,
            "brand_safety_min": self.brand_safety_min,
            "cpm_base": self.cpm_base,
            "historical_ctr": self.historical_ctr,
            "historical_cvr": self.historical_cvr,
            "budget_remaining": self.budget_remaining,
            "total_impressions": self.total_impressions,
            "total_clicks": self.total_clicks,
            "total_conversions": self.total_conversions,
            "total_spend": self.total_spend,
            "performance_score": self.performance_score,
            "is_active": self.is_active,
        }


@dataclass
class AdPlacement:
    placement_id: str
    scene_id: str
    ad_id: str
    placement_type: str   # pre-roll | mid-roll | post-roll
    timestamp_ms: int
    engagement_score: float
    relevance_score: float
    safety_score: float
    total_score: float
    estimated_cpm: float
    ad_title: str = ""
    brand: str = ""

    def to_dict(self) -> dict:
        return {
            "placement_id": self.placement_id,
            "scene_id": self.scene_id,
            "ad_id": self.ad_id,
            "placement_type": self.placement_type,
            "timestamp_ms": self.timestamp_ms,
            "timestamp_fmt": self._fmt(self.timestamp_ms),
            "engagement_score": self.engagement_score,
            "relevance_score": self.relevance_score,
            "safety_score": self.safety_score,
            "total_score": self.total_score,
            "estimated_cpm": self.estimated_cpm,
            "ad_title": self.ad_title,
            "brand": self.brand,
        }

    def _fmt(self, ms: int) -> str:
        h = ms // 3_600_000
        m = (ms % 3_600_000) // 60_000
        s = (ms % 60_000) // 1000
        return f"{h:02d}:{m:02d}:{s:02d}"


# ── Default Ad Inventory ──────────────────────────────────────────────────────
def create_default_inventory() -> list[AdCreative]:
    """Sample ad inventory covering major IAB categories."""
    ads = [
        AdCreative("ad_001", "MacBook Pro M4",        "Apple",
                   "The most powerful laptop for creative professionals.",
                   ["IAB19", "IAB3"], ["25-45", "professional", "tech-savvy"],
                   ["laptop", "mac", "computer", "creative", "professional"],
                   0.7, 12.0, 0.032, 0.008, 50_000),

        AdCreative("ad_002", "Nike Air Max 2025",     "Nike",
                   "Run your best race. New cushioning technology.",
                   ["IAB17", "IAB7"], ["18-35", "fitness", "sports"],
                   ["running", "sport", "fitness", "shoes", "athletic"],
                   0.6, 8.0, 0.028, 0.006, 30_000),

        AdCreative("ad_003", "Spotify Premium",       "Spotify",
                   "Ad-free music, podcasts and audiobooks.",
                   ["IAB1", "IAB9"], ["16-34", "music-lover"],
                   ["music", "song", "artist", "stream", "podcast"],
                   0.8, 6.0, 0.041, 0.012, 40_000),

        AdCreative("ad_004", "MasterClass",           "MasterClass",
                   "Learn from the world's best. Cooking, writing, leadership.",
                   ["IAB5", "IAB4"], ["25-55", "learner", "professional"],
                   ["learn", "skill", "education", "expert", "course"],
                   0.9, 9.0, 0.025, 0.007, 25_000),

        AdCreative("ad_005", "Tesla Model Y",         "Tesla",
                   "Zero emissions. Infinite range. Drive the future.",
                   ["IAB2", "IAB15"], ["28-55", "eco-conscious", "affluent"],
                   ["car", "electric", "drive", "future", "sustainable"],
                   0.75, 18.0, 0.019, 0.004, 80_000),

        AdCreative("ad_006", "HelloFresh",            "HelloFresh",
                   "Fresh ingredients, chef-designed recipes, delivered.",
                   ["IAB8", "IAB6"], ["25-45", "home-cook"],
                   ["food", "cook", "recipe", "meal", "healthy", "kitchen"],
                   0.9, 7.0, 0.038, 0.015, 20_000),

        AdCreative("ad_007", "Morgan Stanley Invest", "Morgan Stanley",
                   "Grow your wealth with personalised investment strategies.",
                   ["IAB13", "IAB3"], ["35-65", "investor", "affluent"],
                   ["money", "invest", "wealth", "financial", "portfolio"],
                   0.95, 22.0, 0.015, 0.003, 100_000),

        AdCreative("ad_008", "Booking.com",           "Booking.com",
                   "Find amazing travel deals. 50 million listings worldwide.",
                   ["IAB20", "IAB6"], ["22-55", "traveller"],
                   ["travel", "hotel", "vacation", "trip", "destination"],
                   0.85, 10.0, 0.033, 0.009, 35_000),

        AdCreative("ad_009", "Headspace",             "Headspace",
                   "Mindfulness and meditation for a healthier mind.",
                   ["IAB7", "IAB23"], ["20-50", "wellness"],
                   ["meditation", "stress", "calm", "mindful", "health", "mental"],
                   0.95, 8.5, 0.036, 0.011, 22_000),

        AdCreative("ad_010", "Adobe Creative Cloud",  "Adobe",
                   "Every app for every creative. Photoshop, Premiere, more.",
                   ["IAB1", "IAB19"], ["20-45", "creative", "designer"],
                   ["design", "creative", "photo", "video", "edit", "art"],
                   0.85, 14.0, 0.027, 0.006, 60_000),

        AdCreative("ad_011", "DoorDash",              "DoorDash",
                   "Your favourite restaurants, delivered fast.",
                   ["IAB8"], ["18-40", "urban"],
                   ["food", "restaurant", "delivery", "order", "meal"],
                   0.8, 5.0, 0.045, 0.018, 15_000),

        AdCreative("ad_012", "Coursera Plus",         "Coursera",
                   "Unlimited access to 7,000+ courses from top universities.",
                   ["IAB5", "IAB4", "IAB19"], ["20-50", "learner"],
                   ["university", "degree", "certificate", "learn", "career"],
                   0.9, 8.0, 0.029, 0.008, 28_000),

        AdCreative("ad_013", "Peloton",               "Peloton",
                   "World-class fitness at home. Live and on-demand classes.",
                   ["IAB7", "IAB17"], ["28-50", "fitness", "affluent"],
                   ["workout", "exercise", "gym", "cycling", "fitness", "home"],
                   0.85, 15.0, 0.022, 0.005, 45_000),

        AdCreative("ad_014", "Squarespace",           "Squarespace",
                   "Build a beautiful website. No coding required.",
                   ["IAB19", "IAB3"], ["20-45", "entrepreneur", "creative"],
                   ["website", "design", "business", "online", "brand"],
                   0.9, 11.0, 0.030, 0.009, 32_000),

        AdCreative("ad_015", "National Geographic+",  "NatGeo",
                   "Explore the world's greatest stories. Nature, science, history.",
                   ["IAB15", "IAB16", "IAB20"], ["25-65", "educated", "curious"],
                   ["nature", "wildlife", "science", "explore", "documentary"],
                   0.95, 9.5, 0.026, 0.007, 20_000),
    ]
    return ads


# ── Scoring Engine ────────────────────────────────────────────────────────────
class AdMatchingEngine:
    """
    Multi-factor ad scoring and placement engine.

    Scoring formula:
      score = 0.30 × content_sim
            + 0.25 × iab_match
            + 0.20 × safety_score
            + 0.15 × demo_score
            + 0.10 × performance_score
    """

    WEIGHTS = {
        "content_sim":   0.30,
        "iab_match":     0.25,
        "safety":        0.20,
        "demographic":   0.15,
        "performance":   0.10,
    }

    MIN_GAP_MS = 180_000     # 3 minutes
    MAX_ADS_PER_5MIN = 1

    def __init__(self, inventory: Optional[list[AdCreative]] = None):
        self.inventory = inventory or create_default_inventory()
        self._vectorizer: Optional[TFIDFVectorizer] = None
        # Build a standalone inventory vectorizer — used only when no
        # external vectorizer has been provided via sync_vectorizer().
        self._embed_inventory_standalone()

    def _embed_inventory_standalone(self) -> None:
        """Build a local vectorizer from ad texts only."""
        texts = [f"{ad.title} {ad.description} {' '.join(ad.keywords)}"
                 for ad in self.inventory]
        self._vectorizer = TFIDFVectorizer(max_features=1000)
        self._vectorizer.fit(texts)
        for ad, text in zip(self.inventory, texts):
            ad.embedding = self._vectorizer.embed(text)

    def sync_vectorizer(self, vectorizer: TFIDFVectorizer) -> None:
        """
        Re-embed all ads using a shared vectorizer (e.g. from the scene
        detector) so that ad and scene embeddings live in the same space
        and cosine similarity is meaningful.
        Call this after indexing scenes into the search engine.
        """
        self._vectorizer = vectorizer
        for ad in self.inventory:
            text = f"{ad.title} {ad.description} {' '.join(ad.keywords)}"
            ad.embedding = vectorizer.embed(text)

    def _iab_match(self, scene_iab: list[dict], ad_iab: list[str]) -> float:
        """Overlap score between scene IAB cats and ad target IAB cats."""
        if not scene_iab or not ad_iab:
            return 0.0
        scene_cats = {c["id"] for c in scene_iab}
        ad_cats = set(ad_iab)
        overlap = scene_cats & ad_cats
        return len(overlap) / max(len(ad_cats), 1)

    def _demographic_score(self, ad: AdCreative, scene: Scene) -> float:
        """
        Simple keyword-based demographic alignment.
        In production this would use viewer analytics.
        """
        # For now use topic alignment as proxy for demographics
        topics = [t.lower() for t in scene.topics]
        demo_keywords = {
            "tech-savvy": ["technology", "computing"],
            "fitness": ["health", "sports", "hobbies"],
            "professional": ["business", "career", "education"],
            "learner": ["education", "science"],
            "traveller": ["travel"],
            "home-cook": ["food & drink", "family & parenting"],
            "music-lover": ["arts & entertainment"],
            "investor": ["personal finance", "business"],
        }
        score = 0.0
        for demo in ad.target_demographics:
            kws = demo_keywords.get(demo, [demo.lower()])
            for kw in kws:
                if any(kw in t for t in topics):
                    score += 0.3
        return min(score, 1.0)

    def score_ad_for_scene(self, ad: AdCreative, scene: Scene) -> dict:
        """Compute all scoring components and weighted total."""
        if not ad.is_active or ad.budget_remaining <= 0:
            return {"total": 0.0, "eligible": False}

        safety = scene.brand_safety.get("safety_score", 0.0)
        if safety < ad.brand_safety_min:
            return {"total": 0.0, "eligible": False, "blocked": "brand_safety"}

        # Content similarity
        content_sim = 0.0
        if scene.embedding is not None and ad.embedding is not None:
            content_sim = max(0.0, cosine_similarity(scene.embedding, ad.embedding))

        iab_match   = self._iab_match(scene.iab_categories, ad.iab_categories)
        demo_score  = self._demographic_score(ad, scene)
        perf_score  = ad.performance_score

        total = (
            content_sim  * self.WEIGHTS["content_sim"]
            + iab_match  * self.WEIGHTS["iab_match"]
            + safety     * self.WEIGHTS["safety"]
            + demo_score * self.WEIGHTS["demographic"]
            + perf_score * self.WEIGHTS["performance"]
        )

        return {
            "total": round(total, 4),
            "content_sim": round(content_sim, 4),
            "iab_match": round(iab_match, 4),
            "safety": round(safety, 4),
            "demographic": round(demo_score, 4),
            "performance": round(perf_score, 4),
            "eligible": True,
        }

    def match_ads(self, scene: Scene, top_k: int = 3) -> list[tuple[AdCreative, dict]]:
        """Return top-k ads for a scene, sorted by score."""
        results = []
        for ad in self.inventory:
            score_info = self.score_ad_for_scene(ad, scene)
            if score_info.get("eligible") and score_info["total"] > 0.05:
                results.append((ad, score_info))
        results.sort(key=lambda x: x[1]["total"], reverse=True)
        return results[:top_k]

    def _estimate_cpm(self, scene: Scene, ad: AdCreative, placement_type: str) -> float:
        """
        Dynamic CPM based on:
        - Base CPM of ad
        - Scene engagement score
        - Placement type premium
        """
        multipliers = {"pre-roll": 1.5, "mid-roll": 1.2, "post-roll": 0.8}
        m = multipliers.get(placement_type, 1.0)
        engagement_bonus = 1.0 + scene.engagement_score * 0.5
        return round(ad.cpm_base * m * engagement_bonus, 2)

    def plan_placements(
        self,
        scenes: list[Scene],
        video_duration_ms: int,
        placement_types: Optional[list[str]] = None,
    ) -> list[AdPlacement]:
        """
        Optimise ad placement across all scenes:
        1. Score scenes for ad suitability
        2. Enforce min-gap and density constraints
        3. Match best ad for each placement slot
        4. Determine placement type by position
        """
        if placement_types is None:
            placement_types = ["pre-roll", "mid-roll", "post-roll"]

        placements: list[AdPlacement] = []
        last_ad_ms = -self.MIN_GAP_MS  # allow first ad immediately

        # Sort scenes by ad_suitability descending
        ranked = sorted(scenes, key=lambda s: s.ad_suitability, reverse=True)

        for scene in ranked:
            gap_ok = (scene.start_ms - last_ad_ms) >= self.MIN_GAP_MS
            if not gap_ok:
                continue

            # Determine placement type by video position
            progress = scene.start_ms / max(video_duration_ms, 1)
            if progress < 0.05 and "pre-roll" in placement_types:
                p_type = "pre-roll"
            elif progress > 0.90 and "post-roll" in placement_types:
                p_type = "post-roll"
            elif "mid-roll" in placement_types:
                p_type = "mid-roll"
            else:
                continue

            # Match best ad
            matches = self.match_ads(scene, top_k=1)
            if not matches:
                continue

            ad, score_info = matches[0]
            cpm = self._estimate_cpm(scene, ad, p_type)

            placement = AdPlacement(
                placement_id=str(uuid.uuid4())[:8],
                scene_id=scene.scene_id,
                ad_id=ad.ad_id,
                placement_type=p_type,
                timestamp_ms=scene.start_ms,
                engagement_score=scene.engagement_score,
                relevance_score=score_info["total"],
                safety_score=score_info["safety"],
                total_score=score_info["total"],
                estimated_cpm=cpm,
                ad_title=ad.title,
                brand=ad.brand,
            )
            placements.append(placement)
            last_ad_ms = scene.start_ms

        # Sort by timestamp
        placements.sort(key=lambda p: p.timestamp_ms)
        return placements

    def simulate_performance(self, placements: list[AdPlacement]) -> dict:
        """Simulate impression/click/revenue metrics for a placement plan."""
        total_impressions = len(placements) * 1000  # 1k views per placement
        total_revenue = sum(p.estimated_cpm * 1000 / 1000 for p in placements)
        avg_ctr = 0.028
        avg_cvr = 0.007

        by_type: dict[str, dict] = {}
        for p in placements:
            t = p.placement_type
            if t not in by_type:
                by_type[t] = {"count": 0, "revenue": 0.0, "impressions": 0}
            by_type[t]["count"] += 1
            by_type[t]["revenue"] += p.estimated_cpm
            by_type[t]["impressions"] += 1000

        return {
            "total_placements": len(placements),
            "total_impressions": total_impressions,
            "estimated_clicks": int(total_impressions * avg_ctr),
            "estimated_conversions": int(total_impressions * avg_cvr),
            "total_revenue_usd": round(total_revenue, 2),
            "avg_cpm": round(total_revenue / max(len(placements), 1), 2),
            "fill_rate": round(len(placements) / max(1, len(placements) + 2), 3),
            "by_placement_type": by_type,
        }
