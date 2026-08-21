"""Ocean Entity Resolution Engine (3-Tier Cascade & Relative Date Parser).

Tier 1: RapidFuzz WRatio (Fast C++ string distance for exact / substring matches)
Tier 2: MiniLM Semantic Embeddings (all-MiniLM-L6-v2 vector cosine similarity)
Tier 3: Gemini Flash Lite Fallback (LLM semantic disambiguation)
"""

import logging
import os
import re
from datetime import datetime
from typing import Any, Callable, List, Optional, Tuple, Union

try:
    from rapidfuzz import fuzz, process
except ImportError:
    fuzz = None
    process = None

try:
    import numpy as np
except ImportError:
    np = None

SentenceTransformer = None

try:
    import dateparser
except ImportError:
    dateparser = None

try:
    from google import genai
    from google.genai import types
except ImportError:
    genai = None
    types = None

logger = logging.getLogger("notion-assistant.matcher")


class EntityResolver:
    """3-Tier Entity Resolution Cascade for candidate resolution."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._embedding_model = None

    def _get_embedding_model(self):
        """Lazy load MiniLM SentenceTransformer model."""
        if self._embedding_model is None:
            try:
                from sentence_transformers import SentenceTransformer
                logger.info("Loading SentenceTransformer model '%s'...", self.model_name)
                self._embedding_model = SentenceTransformer(self.model_name)
            except Exception as exc:
                logger.warning("Failed to load SentenceTransformer model '%s': %s", self.model_name, exc)
                self._embedding_model = None
        return self._embedding_model

    def resolve_entity(
        self,
        query: str,
        candidates: List[Any],
        key_fn: Optional[Callable[[Any], str]] = None,
        rapidfuzz_threshold: float = 75.0,
        embedding_threshold: float = 0.55,
    ) -> Tuple[Optional[Any], str, float]:
        """Resolve a user query to the best matching candidate using 3-tier cascade.

        Returns: (best_candidate, resolution_tier, score)
        Tier strings: 'exact_or_rapidfuzz', 'minilm_embedding', 'gemini_fallback', 'not_found'
        """
        if not query or not candidates:
            return None, "not_found", 0.0

        extract_title = key_fn if key_fn else (lambda c: c.get("title", "") if isinstance(c, dict) else str(c))
        candidate_titles = [extract_title(c) for c in candidates]

        query_clean = query.strip()

        # --- Tier 1: RapidFuzz WRatio ---
        if process and fuzz:
            top_matches = process.extract(query_clean, candidate_titles, limit=3, scorer=fuzz.WRatio)
            if top_matches:
                matched_title, score, match_idx = top_matches[0][0], float(top_matches[0][1]), int(top_matches[0][2])

                # Check if there is an ambiguous tie between top candidates
                if len(top_matches) >= 2 and 60.0 <= score < 95.0:
                    second_score = float(top_matches[1][1])
                    if (score - second_score) <= 12.0:
                        ambiguous_candidates = [candidates[int(m[2])] for m in top_matches if float(m[1]) >= 55.0]
                        if len(ambiguous_candidates) >= 2:
                            logger.info("Ambiguous match tie detected for query '%s': %s", query_clean, [extract_title(c) for c in ambiguous_candidates])
                            return ambiguous_candidates[:3], "ambiguous_menu", score / 100.0

                if score >= rapidfuzz_threshold:
                    logger.debug("Tier 1 RapidFuzz match: '%s' -> '%s' (score=%.1f)", query_clean, matched_title, score)
                    return candidates[match_idx], "exact_or_rapidfuzz", score / 100.0

        # --- Tier 2: MiniLM Semantic Embeddings ---
        model = self._get_embedding_model()
        if model is not None and np is not None:
            try:
                embeddings = model.encode(candidate_titles, normalize_embeddings=True)
                q_emb = model.encode(query_clean, normalize_embeddings=True)

                similarities = np.dot(embeddings, q_emb)
                best_idx = int(np.argmax(similarities))
                best_score = float(similarities[best_idx])

                if best_score >= embedding_threshold:
                    logger.debug(
                        "Tier 2 MiniLM embedding match: '%s' -> '%s' (sim=%.3f)",
                        query_clean,
                        candidate_titles[best_idx],
                        best_score,
                    )
                    return candidates[best_idx], "minilm_embedding", best_score
            except Exception as exc:
                logger.warning("Tier 2 MiniLM embedding resolution failed: %s", exc)

        # --- Tier 3: Gemini Flash Lite Fallback ---
        if genai is not None:
            try:
                api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
                if api_key:
                    client = genai.Client(api_key=api_key)
                    prompt = (
                        f"User said: '{query_clean}'\n\n"
                        f"Candidate list:\n"
                        + "\n".join(f"{idx}: {t}" for idx, t in enumerate(candidate_titles))
                        + "\n\nReturn ONLY the integer index of the intended candidate, or -1 if no candidate matches."
                    )

                    response = client.models.generate_content(
                        model="gemini-3.5-flash-lite",
                        contents=prompt,
                    )
                    text_resp = (response.text or "").strip()
                    match = re.search(r"-?\d+", text_resp)
                    if match:
                        idx = int(match.group(0))
                        if 0 <= idx < len(candidates):
                            logger.debug("Tier 3 Gemini match: '%s' -> '%s'", query_clean, candidate_titles[idx])
                            return candidates[idx], "gemini_fallback", 0.90
            except Exception as exc:
                logger.warning("Tier 3 Gemini LLM fallback resolution failed: %s", exc)

        return None, "not_found", 0.0


def resolve_natural_date(
    date_text: str,
    reference_dt: Optional[datetime] = None,
) -> Optional[str]:
    """Resolve a natural language date expression to YYYY-MM-DD format."""
    if not date_text:
        return None

    clean_text = date_text.strip().lower()
    ref = reference_dt or datetime.now()

    if re.match(r"^\d{4}-\d{2}-\d{2}$", clean_text):
        return clean_text

    # Built-in fast relative terms
    from datetime import timedelta
    if clean_text in ("today", "tonight"):
        return ref.strftime("%Y-%m-%d")
    elif clean_text == "tomorrow":
        return (ref + timedelta(days=1)).strftime("%Y-%m-%d")
    elif clean_text in ("day after tomorrow", "overmorrow"):
        return (ref + timedelta(days=2)).strftime("%Y-%m-%d")
    elif clean_text == "yesterday":
        return (ref - timedelta(days=1)).strftime("%Y-%m-%d")

    weekdays = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
    for idx, day_name in enumerate(weekdays):
        if day_name in clean_text:
            current_weekday = ref.weekday()
            days_ahead = (idx - current_weekday) % 7
            if days_ahead == 0 or "next" in clean_text:
                days_ahead += 7
            return (ref + timedelta(days=days_ahead)).strftime("%Y-%m-%d")

    # Fallback to dateparser if available
    if dateparser is not None:
        try:
            settings = {
                "PREFER_DATES_FROM": "future",
                "RELATIVE_BASE": ref.replace(tzinfo=None) if hasattr(ref, "tzinfo") and ref.tzinfo else ref,
            }
            dt = dateparser.parse(clean_text, settings=settings)
            if dt:
                return dt.strftime("%Y-%m-%d")
        except Exception as exc:
            logger.warning("dateparser failed for '%s': %s", date_text, exc)

    return None


# Global Singleton Instance
entity_resolver = EntityResolver()
