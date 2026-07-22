"""Shared data models used across providers, agents, and the workflow."""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class ResearchRequest(BaseModel):
    """User input coming from the Streamlit form."""

    topic: str
    language: Literal["ar", "en", "both"] = "en"
    num_videos: int = 10
    num_comments: int = 20


class SearchQuery(BaseModel):
    """One search query produced by the Search Planner Agent."""

    query: str
    language: Literal["ar", "en"]


class YouTubeVideo(BaseModel):
    """A single YouTube video and its public statistics."""

    video_id: str
    title: str
    description: str = ""
    channel_title: str = ""
    published_at: str = ""
    view_count: int = 0
    like_count: int = 0
    comment_count: int = 0

    @property
    def url(self) -> str:
        return f"https://www.youtube.com/watch?v={self.video_id}"


class YouTubeComment(BaseModel):
    """A single public comment on a video."""

    comment_id: str
    video_id: str
    author: str = ""
    text: str
    like_count: int = 0
    published_at: str = ""


class RelevanceEvaluation(BaseModel):
    """Relevance Agent's judgment for a single video."""

    video_id: str
    score: int = Field(ge=0, le=100)
    reason: str


class SentimentAnalysis(BaseModel):
    """Sentiment Agent's findings across all collected comments."""

    positive_opinions: list[str] = Field(default_factory=list)
    negative_opinions: list[str] = Field(default_factory=list)
    neutral_opinions: list[str] = Field(default_factory=list)
    common_benefits: list[str] = Field(default_factory=list)
    common_concerns: list[str] = Field(default_factory=list)
    repeated_questions: list[str] = Field(default_factory=list)
    main_themes: list[str] = Field(default_factory=list)


class FinalReport(BaseModel):
    """The complete report shown in Streamlit and saved to disk."""

    topic: str
    summary: str
    total_videos: int
    total_comments: int
    average_relevance_score: float
    key_insights: list[str] = Field(default_factory=list)
    sentiment: SentimentAnalysis
    source_links: list[str] = Field(default_factory=list)
    generated_at: datetime = Field(default_factory=datetime.now)
