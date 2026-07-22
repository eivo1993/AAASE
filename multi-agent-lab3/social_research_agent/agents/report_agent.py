"""Report Agent: assembles the final research report from already-analyzed data."""

from pydantic import BaseModel, Field

from agents.prompts import REPORT_SYSTEM
from agents.relevance_agent import RelevanceAgent
from models import FinalReport, RelevanceEvaluation, SentimentAnalysis, YouTubeVideo
from providers.llm_provider import LLMProvider


class _ReportContent(BaseModel):
    summary: str = ""
    key_insights: list[str] = Field(default_factory=list)


class ReportAgent:
    def __init__(self, llm: LLMProvider | None = None):
        self._llm = llm or LLMProvider()

    def generate(
        self,
        topic: str,
        videos: list[YouTubeVideo],
        relevance_evaluations: list[RelevanceEvaluation],
        sentiment: SentimentAnalysis,
        total_comments: int,
    ) -> FinalReport:
        """Build the final report. All statistics come from code, not the LLM."""
        average_relevance_score = RelevanceAgent.average_score(relevance_evaluations)

        prompt = (
            f'Topic: "{topic}"\n\n'
            f"Statistics:\n"
            f"- total videos analyzed: {len(videos)}\n"
            f"- total comments analyzed: {total_comments}\n"
            f"- average relevance score: {average_relevance_score:.1f}/100\n\n"
            f"Sentiment analysis:\n{sentiment.model_dump_json(indent=2)}"
        )

        content = self._llm.complete_structured(
            prompt=prompt,
            schema=_ReportContent,
            system=REPORT_SYSTEM,
        )

        return FinalReport(
            topic=topic,
            summary=content.summary,
            total_videos=len(videos),
            total_comments=total_comments,
            average_relevance_score=average_relevance_score,
            key_insights=content.key_insights,
            sentiment=sentiment,
            source_links=[v.url for v in videos],
        )
