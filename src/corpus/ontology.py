"""Canonical taxonomy for the RankForge synthetic dataset.

Python is the source of truth; `data/ontology.json` is rendered from this.
"""
from __future__ import annotations

import re
import unicodedata

CATEGORIES: dict[str, list[str]] = {
    "LLM Systems & Inference": [
        "LLM inference cost optimization",
        "prompt caching",
        "dynamic batching",
        "model routing",
        "quantization",
        "GPU utilization",
        "serving latency",
        "KV cache management",
    ],
    "Machine Learning Fundamentals": [
        "supervised learning",
        "classification metrics",
        "regression models",
        "feature engineering",
        "overfitting",
        "cross validation",
        "gradient descent",
        "model evaluation",
    ],
    "Deep Learning Architectures": [
        "transformers",
        "convolutional networks",
        "recurrent networks",
        "attention mechanisms",
        "embedding models",
        "autoencoders",
        "normalization layers",
        "activation functions",
    ],
    "Computer Vision": [
        "image classification",
        "object detection",
        "semantic segmentation",
        "OCR systems",
        "image embeddings",
        "face recognition",
        "video understanding",
        "data augmentation",
    ],
    "Natural Language Processing": [
        "text classification",
        "named entity recognition",
        "sentiment analysis",
        "semantic search",
        "machine translation",
        "summarization",
        "tokenization",
        "text embeddings",
    ],
    "Distributed Systems": [
        "distributed caching",
        "consensus algorithms",
        "replication",
        "sharding",
        "load balancing",
        "message queues",
        "fault tolerance",
        "service discovery",
    ],
    "Databases & Storage": [
        "database indexing",
        "query optimization",
        "transactions",
        "replication strategies",
        "data partitioning",
        "OLTP vs OLAP",
        "NoSQL databases",
        "storage engines",
    ],
    "Networking & Protocols": [
        "HTTP protocols",
        "TCP congestion control",
        "DNS resolution",
        "TLS security",
        "API gateways",
        "rate limiting",
        "network latency",
        "web sockets",
    ],
    "Operating Systems": [
        "process scheduling",
        "memory management",
        "file systems",
        "thread synchronization",
        "virtual memory",
        "kernel design",
        "containers",
        "system calls",
    ],
    "System Design & Scalability": [
        "URL shortener design",
        "real-time chat design",
        "notification systems",
        "search systems",
        "event-driven architecture",
        "observability",
        "autoscaling",
        "multi-region architecture",
    ],
    "Backend Engineering": [
        "REST API design",
        "authentication",
        "caching strategies",
        "background jobs",
        "service reliability",
        "database migrations",
        "API versioning",
        "error handling",
    ],
    "Frontend & Web Development": [
        "React performance",
        "state management",
        "web accessibility",
        "frontend testing",
        "server-side rendering",
        "CSS layout",
        "browser rendering",
        "progressive web apps",
    ],
    "DevOps & Cloud Infrastructure": [
        "Kubernetes deployments",
        "CI/CD pipelines",
        "Docker containers",
        "cloud monitoring",
        "infrastructure as code",
        "autoscaling groups",
        "blue green deployment",
        "incident response",
    ],
    "Mobile App Development": [
        "Android performance",
        "iOS app architecture",
        "Flutter development",
        "mobile release pipelines",
        "crash reporting",
        "offline sync",
        "push notifications",
        "app store optimization",
    ],
    "Startups & Entrepreneurship": [
        "startup ideation",
        "MVP validation",
        "fundraising",
        "go-to-market strategy",
        "founder productivity",
        "market sizing",
        "pricing strategy",
        "product-market fit",
    ],
    "Product Management": [
        "roadmapping",
        "user research",
        "prioritization frameworks",
        "A/B testing",
        "product analytics",
        "launch planning",
        "customer feedback",
        "retention metrics",
    ],
    "Marketing & Growth": [
        "SEO strategy",
        "content marketing",
        "paid acquisition",
        "viral loops",
        "conversion optimization",
        "email campaigns",
        "brand positioning",
        "social media growth",
    ],
    "Stock Market & Investing": [
        "value investing",
        "growth stocks",
        "index funds",
        "technical analysis",
        "earnings reports",
        "risk management",
        "portfolio diversification",
        "options basics",
    ],
    "Personal Finance & Wealth": [
        "budgeting",
        "emergency funds",
        "retirement planning",
        "tax optimization",
        "credit scores",
        "real estate investing",
        "debt management",
        "compound interest",
    ],
    "Travel & Tourism": [
        "budget travel",
        "solo travel",
        "family vacations",
        "travel itineraries",
        "flight booking",
        "hotel selection",
        "travel safety",
        "cultural experiences",
    ],
    "Food & Cooking": [
        "Indian cooking",
        "Italian cuisine",
        "meal prep",
        "healthy recipes",
        "street food",
        "baking basics",
        "restaurant discovery",
        "food nutrition",
    ],
}

DOMAIN_GROUPS: dict[str, list[str]] = {
    "AI/ML": [
        "LLM Systems & Inference",
        "Machine Learning Fundamentals",
        "Deep Learning Architectures",
        "Computer Vision",
        "Natural Language Processing",
    ],
    "CS/Systems": [
        "Distributed Systems",
        "Databases & Storage",
        "Networking & Protocols",
        "Operating Systems",
        "System Design & Scalability",
    ],
    "Engineering": [
        "Backend Engineering",
        "Frontend & Web Development",
        "DevOps & Cloud Infrastructure",
        "Mobile App Development",
    ],
    "Business": [
        "Startups & Entrepreneurship",
        "Product Management",
        "Marketing & Growth",
    ],
    "Finance": [
        "Stock Market & Investing",
        "Personal Finance & Wealth",
    ],
    "Lifestyle": [
        "Travel & Tourism",
        "Food & Cooking",
    ],
}

CATEGORY_TO_DOMAIN: dict[str, str] = {
    cat: domain for domain, cats in DOMAIN_GROUPS.items() for cat in cats
}


def slugify(text: str) -> str:
    """Lowercase ASCII slug for use in cache filenames."""
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"[^a-zA-Z0-9]+", "_", text).strip("_").lower()
    return text


def iter_topics() -> list[tuple[str, str]]:
    """Return [(category, topic), ...] in deterministic order."""
    return [(cat, topic) for cat, topics in CATEGORIES.items() for topic in topics]


def assert_valid() -> None:
    """Sanity-check the ontology at import-time if needed."""
    all_cats = set(CATEGORIES)
    grouped = {c for cats in DOMAIN_GROUPS.values() for c in cats}
    missing = all_cats - grouped
    extra = grouped - all_cats
    if missing or extra:
        raise ValueError(
            f"DOMAIN_GROUPS mismatch — missing: {missing}, extra: {extra}"
        )


assert_valid()
