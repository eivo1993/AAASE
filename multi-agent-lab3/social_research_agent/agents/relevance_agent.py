"""Relevance Agent: scores how well collected videos match the research topic."""

from pydantic import BaseModel

from agents.prompts import RELEVANCE_SYSTEM
from models import RelevanceEvaluation, YouTubeComment, YouTubeVideo
from providers.llm_provider import LLMProvider


class _RelevanceList(BaseModel):
    evaluations: list[RelevanceEvaluation]


class RelevanceAgent:
    def __init__(self, llm: LLMProvider | None = None):
        self._llm = llm or LLMProvider()

    def evaluate(
        self,
        topic: str,
        videos: list[YouTubeVideo],
        comments_by_video: dict[str, list[YouTubeComment]] | None = None,
        max_sample_comments: int = 3,
    ) -> list[RelevanceEvaluation]:
        """Score every video's relevance to `topic`. Returns one evaluation per video."""
        if not videos:
            return []

        comments_by_video = comments_by_video or {}
        prompt = f'Research topic: "{topic}"\n\nVideos:\n'
        for v in videos:
            sample = comments_by_video.get(v.video_id, [])[:max_sample_comments]
            sample_text = "; ".join(c.text[:120] for c in sample) or "(no comments available)"
            prompt += (
                f"\n- video_id: {v.video_id}\n"
                f"  title: {v.title}\n"
                f"  description: {v.description[:200]}\n"
                f"  sample comments: {sample_text}\n"
            )

        result = self._llm.complete_structured(
            prompt=prompt,
            schema=_RelevanceList,
            system=RELEVANCE_SYSTEM,
        )
        return result.evaluations

    @staticmethod
    def average_score(evaluations: list[RelevanceEvaluation]) -> float:
        """Mean relevance score across all evaluations. 0.0 if the list is empty."""
        if not evaluations:
            return 0.0
        return sum(e.score for e in evaluations) / len(evaluations)
