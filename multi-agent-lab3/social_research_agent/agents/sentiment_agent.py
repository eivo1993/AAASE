"""Sentiment Agent: extracts opinions and themes from collected comments."""

import random

from agents.prompts import SENTIMENT_SYSTEM
from models import SentimentAnalysis, YouTubeComment
from providers.llm_provider import LLMProvider


class SentimentAgent:
    def __init__(self, llm: LLMProvider | None = None):
        self._llm = llm or LLMProvider()

    def analyze(
        self, topic: str, comments: list[YouTubeComment], sample_size: int = 10
    ) -> SentimentAnalysis:
        """Summarize opinions and themes across a random sample of `comments`.

        Sampling keeps the prompt small, which is both faster and more reliable
        with smaller models. Empty input -> empty analysis.
        """
        if not comments:
            return SentimentAnalysis()

        sample = random.sample(comments, min(sample_size, len(comments)))
        comments_text = "\n".join(f"- {c.text}" for c in sample)
        prompt = f'Research topic: "{topic}"\n\nComments:\n{comments_text}'

        return self._llm.complete_structured(
            prompt=prompt,
            schema=SentimentAnalysis,
            system=SENTIMENT_SYSTEM,
        )
