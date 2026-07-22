"""Central configuration loaded from environment variables (.env)."""

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    openrouter_api_key: str
    openrouter_model: str
    youtube_api_key: str

    max_search_attempts: int
    default_num_videos: int
    default_num_comments: int
    min_relevance_score: int


def load_settings() -> Settings:
    return Settings(
        openrouter_api_key=os.environ.get("OPENROUTER_API_KEY", ""),
        openrouter_model=os.environ.get("OPENROUTER_MODEL", "openai/gpt-4o-mini"),
        youtube_api_key=os.environ.get("YOUTUBE_API_KEY", ""),
        max_search_attempts=int(os.environ.get("MAX_SEARCH_ATTEMPTS", "3")),
        default_num_videos=int(os.environ.get("DEFAULT_NUM_VIDEOS", "10")),
        default_num_comments=int(os.environ.get("DEFAULT_NUM_COMMENTS", "20")),
        min_relevance_score=int(os.environ.get("MIN_RELEVANCE_SCORE", "60")),
    )


settings = load_settings()
