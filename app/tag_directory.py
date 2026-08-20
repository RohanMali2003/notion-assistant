"""Centralized Tag Directory and Tag Optimization engine for Ocean."""

import difflib
import logging
import re
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

# Canonical Tag Directory with domain keywords for fast zero-shot keyword matching
TAG_DIRECTORY: Dict[str, Dict[str, Any]] = {
    "AI Research": {
        "keywords": {"paper", "arxiv", "transformer", "llm", "gemini", "gemma", "attention", "moe", "deep learning", "neural", "diffusion", "reasoning", "benchmark", "weights", "fine-tuning", "lora", "embedding", "agent", "agents", "multimodal", "vision", "dataset"},
        "description": "Academic research papers, ML architecture, model training, and AI evaluations.",
    },
    "System Design": {
        "keywords": {"architecture", "scalability", "microservices", "load balancer", "caching", "database", "sql", "nosql", "sharding", "replication", "consistency", "high availability", "latency", "throughput", "system design"},
        "description": "System architecture, scalability patterns, distributed storage, and backend infra.",
    },
    "Distributed Systems": {
        "keywords": {"raft", "paxos", "consensus", "distributed", "grpc", "cap theorem", "byzantine", "cluster", "node", "fault tolerance", "election", "replication", "partition"},
        "description": "Consensus protocols, distributed networking, peer-to-peer, and fault tolerance.",
    },
    "Open Source": {
        "keywords": {"github", "repo", "repository", "git", "oss", "library", "framework", "package", "pr", "pull request", "release"},
        "description": "Open source tools, GitHub libraries, frameworks, and developer utilities.",
    },
    "Leetcode": {
        "keywords": {"leetcode", "algorithm", "data structure", "binary search", "graph", "dynamic programming", "tree", "dp", "bfs", "dfs", "sliding window", "two pointer", "heap", "trie"},
        "description": "Coding interview practice, algorithms, data structures, and problem reviews.",
    },
    "Learning": {
        "keywords": {"course", "tutorial", "study", "syllabus", "book", "lecture", "guide", "concept", "learn", "fundamentals"},
        "description": "Structured study plans, tutorials, and fundamental concepts.",
    },
    "Substack": {
        "keywords": {"substack", "essay", "post", "writing", "newsletter", "article", "thesis", "draft", "publication"},
        "description": "Long-form written essays, newsletters, and publication drafts.",
    },
    "Schoolwork": {
        "keywords": {"homework", "assignment", "exam", "quiz", "class", "coursework", "professor", "ta", "lecture", "cs", "university", "syllabus"},
        "description": "University assignments, exams, lectures, and academic coursework.",
    },
    "UMass Admin": {
        "keywords": {"umass", "gpaf", "cics", "scholarship", "onboarding", "admin", "dining", "berkshire", "payroll", "i20", "visa", "opt", "cpt", "academicworks"},
        "description": "University administration, employment forms, dining onboarding, and scholarships.",
    },
    "Finances": {
        "keywords": {"bank", "tax", "payment", "money", "salary", "expense", "bill", "tuition", "budget", "invoice", "receipt", "investment", "credit"},
        "description": "Personal finances, bills, taxes, payroll, and budget management.",
    },
    "Projects": {
        "keywords": {"project", "feature", "build", "mvp", "app", "dev", "hackathon", "deploy", "render", "docker", "frontend", "backend"},
        "description": "Active coding projects, product development, and side initiatives.",
    },
    "Career": {
        "keywords": {"job", "resume", "interview", "application", "referral", "recruiter", "offer", "linkedin", "internship", "career"},
        "description": "Job applications, resume updates, recruiter chats, and career milestones.",
    },
    "Philosophy": {
        "keywords": {"philosophy", "ethics", "stoicism", "mindset", "reflection", "thoughts", "consciousness", "life", "meaning", "rambling"},
        "description": "Philosophical notes, mental models, and personal reflections.",
    },
    "Personal Site": {
        "keywords": {"website", "portfolio", "blog", "domain", "personal site", "landing page"},
        "description": "Personal portfolio, website updates, and online presence.",
    },
    "Miscellaneous": {
        "keywords": set(),
        "description": "General catch-all for untagged or multi-disciplinary items.",
    },
}

CANONICAL_TAG_NAMES: List[str] = list(TAG_DIRECTORY.keys())


def match_closest_tag(inferred_tag: Optional[str] = None, text_content: str = "", min_score: int = 1) -> str:
    """Resolve an inferred tag or raw text into a canonical tag from the directory."""
    if inferred_tag:
        clean_inferred = inferred_tag.strip().title()
        # Direct exact or case-insensitive match
        for tag in CANONICAL_TAG_NAMES:
            if tag.lower() == clean_inferred.lower():
                return tag

        # Fuzzy string match on tag name
        best_match = difflib.get_close_matches(clean_inferred, CANONICAL_TAG_NAMES, n=1, cutoff=0.7)
        if best_match:
            return best_match[0]

    # Keyword match based on text content
    if text_content:
        words = set(re.findall(r"\b[a-zA-Z0-9_-]+\b", text_content.lower()))
        best_tag = "Miscellaneous"
        max_score = 0

        for tag, data in TAG_DIRECTORY.items():
            if tag == "Miscellaneous":
                continue
            keywords = data["keywords"]
            # Count keyword hits
            overlap = len(words.intersection(keywords))
            if overlap > max_score and overlap >= min_score:
                max_score = overlap
                best_tag = tag

        if max_score >= min_score:
            return best_tag

    return "Miscellaneous"


def find_tag_reclassification_suggestions(
    items: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Audit items tagged 'Miscellaneous' or generic tags and suggest more specific tags.

    Returns a list of suggestion dicts:
    [{"id": page_id, "title": title, "url": url, "current_tag": tag, "suggested_tag": better_tag, "reason": reason}]
    """
    suggestions = []

    for item in items:
        current_tag = item.get("current_tag") or "Miscellaneous"
        title = item.get("title", "")
        body_text = item.get("text", "")
        combined_text = f"{title} {body_text}"

        # If item is in Miscellaneous or has no tag, check if it strongly belongs elsewhere
        if current_tag in ("Miscellaneous", "", None):
            best_tag = match_closest_tag(None, combined_text, min_score=2)
            if best_tag != "Miscellaneous":
                suggestions.append({
                    "id": item.get("id", ""),
                    "title": title,
                    "url": item.get("url", ""),
                    "current_tag": current_tag or "Miscellaneous",
                    "suggested_tag": best_tag,
                    "reason": f"Content contains strong domain keywords matching '{best_tag}'",
                })

    return suggestions
