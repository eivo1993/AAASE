"""Conditional routing functions used for the branching edges in the workflow graph."""

from config import settings
from workflow.state import ResearchState


def route_after_planning(state: ResearchState) -> str:
    """After search planning: bail out to END if planning itself failed."""
    if state.get("error"):
        return "end"
    return "search"


def route_after_search(state: ResearchState) -> str:
    """After YouTube search: bail out to END if the search itself failed."""
    if state.get("error"):
        return "end"
    return "relevance"


def route_after_relevance(state: ResearchState) -> str:
    """After relevance scoring: retry search, proceed to sentiment, or give up gracefully."""
    if state.get("error"):
        return "end"
    if state["average_relevance_score"] >= settings.min_relevance_score:
        return "sentiment"
    if state["search_attempts"] >= settings.max_search_attempts:
        return "sentiment"  # out of retries: proceed with best-effort data instead of failing
    return "retry"


def route_after_sentiment(state: ResearchState) -> str:
    """After sentiment analysis: bail out to END if it failed."""
    if state.get("error"):
        return "end"
    return "report"
