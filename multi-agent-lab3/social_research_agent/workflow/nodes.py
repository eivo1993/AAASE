"""Node functions: each reads what it needs from state and returns a partial update."""

from agents.relevance_agent import RelevanceAgent
from agents.report_agent import ReportAgent
from agents.search_planner import SearchPlannerAgent
from agents.sentiment_agent import SentimentAgent
from providers.youtube_provider import YouTubeProvider
from workflow.state import ResearchState

_search_planner = SearchPlannerAgent()
_relevance_agent = RelevanceAgent()
_sentiment_agent = SentimentAgent()
_report_agent = ReportAgent()
_youtube = YouTubeProvider()


def search_planner_node(state: ResearchState) -> dict:
    previous = state["search_queries"] if state["search_attempts"] > 0 else None
    try:
        queries = _search_planner.plan(state["request"], previous_queries=previous)
        return {
            "search_queries": queries,
            "search_attempts": state["search_attempts"] + 1,
        }
    except Exception as e:
        return {"error": f"Search planning failed: {e}"}


def youtube_search_node(state: ResearchState) -> dict:
    request = state["request"]
    try:
        video_ids: list[str] = []
        for query in state["search_queries"]:
            for vid in _youtube.search_videos(query.query, max_results=request.num_videos):
                if vid not in video_ids:
                    video_ids.append(vid)
        video_ids = video_ids[: request.num_videos]

        videos = _youtube.get_video_details(video_ids)
        comments_by_video = {
            v.video_id: _youtube.get_comments(v.video_id, max_comments=request.num_comments)
            for v in videos
        }
        return {"videos": videos, "comments_by_video": comments_by_video}
    except Exception as e:
        return {"error": f"YouTube search failed: {e}"}


def relevance_node(state: ResearchState) -> dict:
    try:
        evaluations = _relevance_agent.evaluate(
            state["request"].topic, state["videos"], state["comments_by_video"]
        )
        average = _relevance_agent.average_score(evaluations)
        return {"relevance_evaluations": evaluations, "average_relevance_score": average}
    except Exception as e:
        return {"error": f"Relevance evaluation failed: {e}"}


def sentiment_node(state: ResearchState) -> dict:
    all_comments = [c for cs in state["comments_by_video"].values() for c in cs]
    try:
        sentiment = _sentiment_agent.analyze(state["request"].topic, all_comments)
        return {"sentiment": sentiment}
    except Exception as e:
        return {"error": f"Sentiment analysis failed: {e}"}


def report_node(state: ResearchState) -> dict:
    all_comments = [c for cs in state["comments_by_video"].values() for c in cs]
    try:
        report = _report_agent.generate(
            topic=state["request"].topic,
            videos=state["videos"],
            relevance_evaluations=state["relevance_evaluations"],
            sentiment=state["sentiment"],
            total_comments=len(all_comments),
        )
        return {"report": report}
    except Exception as e:
        return {"error": f"Report generation failed: {e}"}
