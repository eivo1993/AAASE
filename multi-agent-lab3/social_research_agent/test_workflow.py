"""A couple of simple workflow tests. Run directly: python test_workflow.py

Mocks the agent/provider calls so these run instantly with no real API calls.
The agents themselves are already verified against real APIs in earlier steps;
this just checks the workflow's routing/error-handling logic.
"""

from config import settings
from models import FinalReport, ResearchRequest, SearchQuery, SentimentAnalysis
from workflow import nodes
from workflow.graph import build_graph
from workflow.state import initial_state

FAKE_REPORT = FinalReport(
    topic="test",
    summary="stub",
    total_videos=0,
    total_comments=0,
    average_relevance_score=0.0,
    sentiment=SentimentAnalysis(),
)


def test_youtube_api_failure_ends_gracefully():
    """If the YouTube API call blows up, the workflow should stop with an error
    message instead of crashing or continuing on to sentiment/report."""
    nodes._search_planner.plan = lambda *a, **k: [SearchQuery(query="x", language="en")]
    nodes._youtube.search_videos = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("quota exceeded"))

    state = build_graph().invoke(initial_state(ResearchRequest(topic="anything")))

    assert state["error"] is not None and "quota exceeded" in state["error"]
    assert state["report"] is None
    print("OK: test_youtube_api_failure_ends_gracefully")


def test_max_search_attempts_no_infinite_loop():
    """If relevance never improves, retries must stop at MAX_SEARCH_ATTEMPTS
    instead of looping forever."""
    calls = {"n": 0}

    def fake_search_videos(*a, **k):
        calls["n"] += 1
        return []  # never finds anything relevant -> average score stays 0.0

    nodes._search_planner.plan = lambda *a, **k: [SearchQuery(query="x", language="en")]
    nodes._youtube.search_videos = fake_search_videos
    nodes._report_agent.generate = lambda *a, **k: FAKE_REPORT

    state = build_graph().invoke(initial_state(ResearchRequest(topic="never found")))

    assert calls["n"] == settings.max_search_attempts
    assert state["search_attempts"] == settings.max_search_attempts
    assert state["report"] is not None  # still gives a best-effort report, doesn't just hang/fail
    print("OK: test_max_search_attempts_no_infinite_loop")


if __name__ == "__main__":
    test_youtube_api_failure_ends_gracefully()
    test_max_search_attempts_no_infinite_loop()
    print("All tests passed.")
