import re
from collections import Counter

from app.models import get_model_registry
from app.router_recommendation_service import recommend_router_updates
from app.schemas import RecommendationReviewItem, RecommendationReviewResponse

KNOWN_RULE_PATTERNS = (
    {
        "route_to": "code_specialist",
        "keywords": ("contains code", "code", "debug", "traceback", "python", "bug", "报错", "代码"),
        "comment": "Current routing already prioritizes coding and debug signals to code_specialist.",
    },
    {
        "route_to": "strong_reasoning",
        "keywords": ("compare", "trade off", "tradeoff", "analyze", "analysis", "reasoning", "why", "推理", "分析"),
        "comment": "Current routing already sends high-reasoning and comparison requests to strong_reasoning.",
    },
    {
        "route_to": "fast_general",
        "keywords": ("summarize", "summary", "bullet point", "brief summary", "tl dr", "总结", "概括", "摘要"),
        "comment": "Current routing already keeps lightweight summarization on fast_general.",
    },
)


def normalize_text(text: str) -> str:
    normalized = re.sub(r"[_\-]+", " ", text.casefold())
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized.strip()


def check_known_rule(condition: str, route_to: str) -> str | None:
    normalized_condition = normalize_text(condition)
    for pattern in KNOWN_RULE_PATTERNS:
        if pattern["route_to"] != route_to:
            continue
        if any(keyword in normalized_condition for keyword in pattern["keywords"]):
            return pattern["comment"]
    return None


def review_recommendations(limit: int = 50) -> RecommendationReviewResponse:
    recommendation_response = recommend_router_updates(limit=limit)
    registry = get_model_registry()
    reviews: list[RecommendationReviewItem] = []
    status_counts: Counter[str] = Counter()

    for item in recommendation_response.recommended_changes:
        review = _review_item(
            condition=item.condition,
            route_to=item.route_to,
            registry=registry,
        )
        reviews.append(review)
        status_counts[review.status] += 1

    summary = (
        f"Reviewed {len(reviews)} recommendation(s): "
        f"{status_counts.get('already_covered', 0)} already covered, "
        f"{status_counts.get('compatible', 0)} compatible, "
        f"{status_counts.get('unsupported_route', 0)} unsupported, "
        f"{status_counts.get('needs_manual_review', 0)} needing manual review."
    )
    notes = [
        f"Recommendation source summary: {recommendation_response.summary}",
        "This review is read-only; it does not modify router.py and does not auto-apply any recommendation.",
    ]
    if not reviews:
        notes.append("No recommendation items were available for review.")

    return RecommendationReviewResponse(
        summary=summary,
        reviews=reviews,
        notes=notes,
    )


def _review_item(condition: str, route_to: str, registry) -> RecommendationReviewItem:
    if not registry.has(route_to):
        return RecommendationReviewItem(
            condition=condition,
            route_to=route_to,
            status="unsupported_route",
            comment=f"Recommended route '{route_to}' does not exist in MODEL_REGISTRY.",
        )

    matched_comment = check_known_rule(condition, route_to)
    if matched_comment:
        return RecommendationReviewItem(
            condition=condition,
            route_to=route_to,
            status="already_covered",
            comment=matched_comment,
        )

    normalized_condition = normalize_text(condition)
    if len(normalized_condition) < 16 or normalized_condition in {"always", "default", "general"}:
        return RecommendationReviewItem(
            condition=condition,
            route_to=route_to,
            status="needs_manual_review",
            comment="Condition is too vague to verify against the current routing rules automatically.",
        )

    return RecommendationReviewItem(
        condition=condition,
        route_to=route_to,
        status="compatible",
        comment=f"Route '{route_to}' exists and the recommendation does not directly conflict with known rules.",
    )
