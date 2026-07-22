"""Builds and compiles the LangGraph research workflow."""

from langgraph.graph import END, START, StateGraph

from workflow.nodes import (
    relevance_node,
    report_node,
    search_planner_node,
    sentiment_node,
    youtube_search_node,
)
from workflow.routing import (
    route_after_planning,
    route_after_relevance,
    route_after_search,
    route_after_sentiment,
)
from workflow.state import ResearchState


def build_graph():
    graph = StateGraph(ResearchState)

    graph.add_node("search_planner", search_planner_node)
    graph.add_node("youtube_search", youtube_search_node)
    graph.add_node("relevance", relevance_node)
    graph.add_node("sentiment", sentiment_node)
    graph.add_node("report", report_node)

    graph.add_edge(START, "search_planner")

    graph.add_conditional_edges(
        "search_planner",
        route_after_planning,
        {"search": "youtube_search", "end": END},
    )
    graph.add_conditional_edges(
        "youtube_search",
        route_after_search,
        {"relevance": "relevance", "end": END},
    )
    graph.add_conditional_edges(
        "relevance",
        route_after_relevance,
        {"retry": "search_planner", "sentiment": "sentiment", "end": END},
    )
    graph.add_conditional_edges(
        "sentiment",
        route_after_sentiment,
        {"report": "report", "end": END},
    )
    graph.add_edge("report", END)

    return graph.compile()
