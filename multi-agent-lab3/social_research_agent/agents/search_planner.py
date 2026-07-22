"""Search Planner Agent: turns a research topic into YouTube search queries."""

from pydantic import BaseModel

from agents.prompts import SEARCH_PLANNER_RETRY_NOTE, SEARCH_PLANNER_SYSTEM
from models import ResearchRequest, SearchQuery
from providers.llm_provider import LLMProvider


class _SearchQueryList(BaseModel):
    """Internal wrapper: complete_structured needs one Pydantic model, not a bare list."""

    queries: list[SearchQuery]


class SearchPlannerAgent:
    def __init__(self, llm: LLMProvider | None = None):
        self._llm = llm or LLMProvider()

    def plan(
        self,
        request: ResearchRequest,
        previous_queries: list[SearchQuery] | None = None,
        num_queries: int = 4,
    ) -> list[SearchQuery]:
        """Produce `num_queries` search queries for the given research request."""
        prompt = (
            f'Topic: "{request.topic}"\n'
            f"Requested language: {request.language}\n"
            f"Number of queries to produce: {num_queries}"
        )

        if previous_queries:
            previous_text = "\n".join(f"- {q.query} ({q.language})" for q in previous_queries)
            prompt += "\n" + SEARCH_PLANNER_RETRY_NOTE.format(previous_queries=previous_text)

        result = self._llm.complete_structured(
            prompt=prompt,
            schema=_SearchQueryList,
            system=SEARCH_PLANNER_SYSTEM,
        )
        return result.queries
