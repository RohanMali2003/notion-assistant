"""Duplicate detection engine for Notion subjects, tasks, and resources.

Uses exact matching, normalized token overlap, and fuzzy string similarity (SequenceMatcher)
to group duplicate or semantically overlapping items across Notion databases.
"""

from dataclasses import dataclass, field
import difflib
import logging
import re
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)

# Common stopwords and filler words to ignore when comparing titles
COMMON_STOPWORDS: Set[str] = {
    "a", "an", "the", "and", "or", "in", "on", "at", "to", "for", "of", "with",
    "by", "from", "into", "about", "fundamentals", "foundations", "basics",
    "architecture", "architectures", "introduction", "overview", "deep", "dive",
    "models", "model", "scaling", "theory", "mechanics", "comprehensive", "guide",
}


@dataclass
class DuplicateItem:
    """Represents a single Notion item in a duplicate cluster."""
    id: str
    title: str
    url: str
    created_time: Optional[str] = None
    status: Optional[str] = None
    due_date: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    raw_props: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DuplicateCluster:
    """Represents a group of duplicate or highly similar items."""
    category: str  # "Subject", "Task", "Resource"
    similarity_score: float  # 0.0 to 1.0
    match_reason: str
    items: List[DuplicateItem]
    recommended_action: str = ""


def normalize_title(title: str, remove_stopwords: bool = False) -> str:
    """Clean and normalize a title string for similarity comparison."""
    if not title:
        return ""
    # Lowercase and remove punctuation
    cleaned = re.sub(r"[^\w\s]", " ", title.lower()).strip()
    words = cleaned.split()
    if remove_stopwords:
        filtered = [w for w in words if w not in COMMON_STOPWORDS and len(w) > 1]
        if filtered:
            return " ".join(filtered)
    return " ".join(words)


def calculate_similarity(text1: str, text2: str) -> float:
    """Calculate hybrid similarity score between two texts using SequenceMatcher and token overlap."""
    norm1 = normalize_title(text1, remove_stopwords=False)
    norm2 = normalize_title(text2, remove_stopwords=False)

    if not norm1 or not norm2:
        return 0.0

    if norm1 == norm2:
        return 1.0

    # 1. SequenceMatcher ratio on normalized full text
    seq_ratio = difflib.SequenceMatcher(None, norm1, norm2).ratio()

    # 2. Token overlap without stopwords
    tokens1 = set(normalize_title(text1, remove_stopwords=True).split())
    tokens2 = set(normalize_title(text2, remove_stopwords=True).split())

    if tokens1 and tokens2:
        jaccard = len(tokens1.intersection(tokens2)) / len(tokens1.union(tokens2))
        containment = len(tokens1.intersection(tokens2)) / min(len(tokens1), len(tokens2))
    else:
        jaccard = 0.0
        containment = 0.0

    # 3. Token sort ratio
    sorted1 = " ".join(sorted(tokens1))
    sorted2 = " ".join(sorted(tokens2))
    token_sort_ratio = difflib.SequenceMatcher(None, sorted1, sorted2).ratio() if sorted1 and sorted2 else 0.0

    # Return max of direct sequence match, token sort match, or weighted combination
    token_hybrid = 0.5 * containment + 0.3 * jaccard + 0.2 * token_sort_ratio
    return max(seq_ratio, token_sort_ratio, token_hybrid)


def find_duplicate_clusters(
    items: List[DuplicateItem],
    category: str,
    threshold: float = 0.68,
) -> List[DuplicateCluster]:
    """Find groups of duplicates among a list of DuplicateItem objects."""
    if len(items) < 2:
        return []

    clusters: List[DuplicateCluster] = []
    visited: Set[str] = set()

    for i in range(len(items)):
        item_a = items[i]
        if item_a.id in visited:
            continue

        matched_items = [item_a]
        highest_score = 0.0
        reasons = []

        for j in range(i + 1, len(items)):
            item_b = items[j]
            if item_b.id in visited:
                continue

            score = calculate_similarity(item_a.title, item_b.title)

            # Check for URL match for resources
            if category == "Resource" and item_a.url and item_b.url and item_a.url == item_b.url:
                score = 1.0
                reasons.append("Identical URL")

            if score >= threshold:
                matched_items.append(item_b)
                highest_score = max(highest_score, score)
                if not reasons:
                    reasons.append(f"{int(score * 100)}% title similarity")

        if len(matched_items) > 1:
            for m in matched_items:
                visited.add(m.id)

            # Determine recommended action
            if category == "Subject":
                action = "Review resources under both subjects, merge if necessary, and delete the redundant subject page."
            elif category == "Task":
                action = "Mark older or redundant task as Done, or delete if created by mistake."
            else:
                action = "Delete duplicate resource entry."

            clusters.append(
                DuplicateCluster(
                    category=category,
                    similarity_score=round(highest_score, 2),
                    match_reason=", ".join(reasons) if reasons else "High similarity",
                    items=matched_items,
                    recommended_action=action,
                )
            )

    return clusters
