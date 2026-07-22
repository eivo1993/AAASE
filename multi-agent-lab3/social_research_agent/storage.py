"""Saves each research run's inputs, intermediate results, and final report to disk."""

import json
from datetime import datetime
from pathlib import Path

from models import FinalReport
from workflow.state import ResearchState

OUTPUTS_DIR = Path(__file__).parent / "outputs"


def save_run(state: ResearchState) -> Path:
    """Persist a completed (or failed) run under outputs/<timestamp>_<topic>/."""
    request = state["request"]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = OUTPUTS_DIR / f"{timestamp}_{_slugify(request.topic)}"
    run_dir.mkdir(parents=True, exist_ok=True)

    _write_json(run_dir / "request.json", request.model_dump(mode="json"))
    _write_json(
        run_dir / "search_queries.json",
        [q.model_dump(mode="json") for q in state["search_queries"]],
    )
    _write_json(
        run_dir / "youtube_results.json",
        {
            "videos": [v.model_dump(mode="json") for v in state["videos"]],
            "comments_by_video": {
                vid: [c.model_dump(mode="json") for c in comments]
                for vid, comments in state["comments_by_video"].items()
            },
        },
    )
    _write_json(
        run_dir / "relevance.json",
        [e.model_dump(mode="json") for e in state["relevance_evaluations"]],
    )

    if state["sentiment"]:
        _write_json(run_dir / "sentiment.json", state["sentiment"].model_dump(mode="json"))

    if state["report"]:
        _write_json(run_dir / "report.json", state["report"].model_dump(mode="json"))
        (run_dir / "report.md").write_text(report_to_markdown(state["report"]), encoding="utf-8")

    if state["error"]:
        (run_dir / "error.txt").write_text(state["error"], encoding="utf-8")

    return run_dir


def _slugify(text: str, max_len: int = 40) -> str:
    slug = "".join(c if c.isalnum() else "-" for c in text.strip())
    slug = "-".join(filter(None, slug.split("-")))
    return slug[:max_len] or "topic"


def _write_json(path: Path, data) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def report_to_markdown(report: FinalReport) -> str:
    lines = [
        f"# Research Report: {report.topic}",
        "",
        f"*Generated: {report.generated_at}*",
        "",
        "## Summary",
        report.summary,
        "",
        "## Statistics",
        f"- Videos analyzed: {report.total_videos}",
        f"- Comments analyzed: {report.total_comments}",
        f"- Average relevance score: {report.average_relevance_score:.1f}/100",
        "",
        "## Key Insights",
        *([f"- {insight}" for insight in report.key_insights] or ["- (none)"]),
        "",
        "## Sources",
        *(f"- {link}" for link in report.source_links),
    ]
    return "\n".join(lines)
