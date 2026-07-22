"""Thin wrapper around the official YouTube Data API v3."""

from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

from config import settings
from models import YouTubeComment, YouTubeVideo


class YouTubeProvider:
    def __init__(self, api_key: str | None = None):
        self._client = build(
            "youtube", "v3", developerKey=api_key or settings.youtube_api_key
        )

    def search_videos(self, query: str, max_results: int = 10) -> list[str]:
        """Search YouTube and return matching video IDs."""
        response = (
            self._client.search()
            .list(
                q=query,
                part="id",
                type="video",
                maxResults=max_results,
            )
            .execute()
        )
        return [item["id"]["videoId"] for item in response.get("items", [])]

    def get_video_details(self, video_ids: list[str]) -> list[YouTubeVideo]:
        """Fetch title, channel, and statistics for a batch of video IDs."""
        if not video_ids:
            return []

        response = (
            self._client.videos()
            .list(part="snippet,statistics", id=",".join(video_ids))
            .execute()
        )

        videos = []
        for item in response.get("items", []):
            snippet = item["snippet"]
            stats = item.get("statistics", {})
            videos.append(
                YouTubeVideo(
                    video_id=item["id"],
                    title=snippet.get("title", ""),
                    description=snippet.get("description", ""),
                    channel_title=snippet.get("channelTitle", ""),
                    published_at=snippet.get("publishedAt", ""),
                    view_count=int(stats.get("viewCount", 0)),
                    like_count=int(stats.get("likeCount", 0)),
                    comment_count=int(stats.get("commentCount", 0)),
                )
            )
        return videos

    def get_comments(self, video_id: str, max_comments: int = 20) -> list[YouTubeComment]:
        """Fetch top-level public comments. Returns [] if comments are disabled."""
        try:
            response = (
                self._client.commentThreads()
                .list(
                    part="snippet",
                    videoId=video_id,
                    maxResults=min(max_comments, 100),
                    textFormat="plainText",
                    order="relevance",
                )
                .execute()
            )
        except HttpError as e:
            if e.resp.status == 403 and "commentsDisabled" in str(e):
                return []
            raise

        comments = []
        for item in response.get("items", []):
            top_comment = item["snippet"]["topLevelComment"]["snippet"]
            comments.append(
                YouTubeComment(
                    comment_id=item["id"],
                    video_id=video_id,
                    author=top_comment.get("authorDisplayName", ""),
                    text=top_comment.get("textDisplay", ""),
                    like_count=top_comment.get("likeCount", 0),
                    published_at=top_comment.get("publishedAt", ""),
                )
            )
        return comments
