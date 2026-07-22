"""Shared state that flows through every node of the LangGraph workflow."""

from typing import TypedDict

from models import (
    FinalReport,
    RelevanceEvaluation,
    ResearchRequest,
    SearchQuery,
    SentimentAnalysis,
    YouTubeComment,
    YouTubeVideo,
)


class ResearchState(TypedDict):
    request: ResearchRequest

    search_queries: list[SearchQuery]
    search_attempts: int

    videos: list[YouTubeVideo]
    comments_by_video: dict[str, list[YouTubeComment]]

    relevance_evaluations: list[RelevanceEvaluation]
    average_relevance_score: float

    sentiment: SentimentAnalysis | None
    report: FinalReport | None

    error: str | None


def initial_state(request: ResearchRequest) -> ResearchState:
    """Build the starting state for a new research run."""
    return ResearchState(
        request=request,
        search_queries=[],
        search_attempts=0,
        videos=[],
        comments_by_video={},
        relevance_evaluations=[],
        average_relevance_score=0.0,
        sentiment=None,
        report=None,
        error=None,
    )
