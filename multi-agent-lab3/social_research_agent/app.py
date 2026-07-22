"""Streamlit interface for the YouTube social research agent."""

import streamlit as st

from models import ResearchRequest
from storage import report_to_markdown, save_run
from workflow.graph import build_graph
from workflow.state import initial_state

st.set_page_config(page_title="YouTube Opinion Research", page_icon="🔍")
st.title("🔍 YouTube Opinion Research")

NODE_LABELS = {
    "search_planner": "Planning search queries",
    "youtube_search": "Searching YouTube and collecting videos/comments",
    "relevance": "Evaluating relevance to the topic",
    "sentiment": "Analyzing comment sentiment",
    "report": "Generating final report",
}

with st.form("research_form"):
    topic = st.text_input("Research topic", placeholder="e.g. AI in education")
    submitted = st.form_submit_button("Search 🔍")

if submitted:
    if not topic.strip():
        st.error("Please enter a research topic first.")
        st.stop()

    request = ResearchRequest(topic=topic, language="en", num_videos=5, num_comments=10)

    final_state = initial_state(request)
    graph = build_graph()

    with st.status("Running research...", expanded=True) as status:
        for step_output in graph.stream(final_state):
            for node_name, update in step_output.items():
                final_state.update(update)
                status.write(f"✅ {NODE_LABELS.get(node_name, node_name)}")
        status.update(label="Research complete", state="complete")

    if final_state.get("error"):
        st.error(f"An error occurred during research: {final_state['error']}")
        st.stop()

    report = final_state["report"]
    if not report:
        st.warning("Couldn't generate a report — try a different topic or more videos.")
        st.stop()

    saved_path = save_run(final_state)
    st.caption(f"Saved to: {saved_path}")

    st.subheader("📊 Results Summary")
    m1, m2, m3 = st.columns(3)
    m1.metric("Videos analyzed", report.total_videos)
    m2.metric("Comments analyzed", report.total_comments)
    m3.metric("Average relevance", f"{report.average_relevance_score:.0f}/100")

    st.subheader("📝 Report")
    st.write(report.summary)

    if report.key_insights:
        st.subheader("💡 Key Insights")
        for insight in report.key_insights:
            st.markdown(f"- {insight}")

    with st.expander("Sentiment analysis details"):
        sentiment = report.sentiment
        sections = [
            ("Positive opinions", sentiment.positive_opinions),
            ("Negative opinions", sentiment.negative_opinions),
            ("Neutral opinions", sentiment.neutral_opinions),
            ("Common benefits", sentiment.common_benefits),
            ("Common concerns", sentiment.common_concerns),
            ("Repeated questions", sentiment.repeated_questions),
            ("Main themes", sentiment.main_themes),
        ]
        for title, items in sections:
            st.markdown(f"**{title}:**")
            st.markdown("\n".join(f"- {x}" for x in items) or "- None")

    st.subheader("🔗 Sources")
    for link in report.source_links:
        st.markdown(f"- [{link}]({link})")

    st.download_button(
        "⬇️ Download Report (Markdown)",
        data=report_to_markdown(report),
        file_name="report.md",
        mime="text/markdown",
    )
