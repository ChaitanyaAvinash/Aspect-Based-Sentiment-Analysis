"""Streamlit demo — offline, CPU-only Aspect-Based Sentiment Analysis.

Loads the CPU deployment artifact (int8, via ModelService), preloads + warms up
once, ships cached example reviews, and highlights aspects colored by sentiment.
Runs fully offline — no downloads at demo time.

    streamlit run app/streamlit_app.py     # or: make demo
"""

from __future__ import annotations

import os

# Guarantee no network at demo time (weights + tokenizer are local).
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import pandas as pd
import streamlit as st

from absa.config import get_settings
from absa.serving.render import EXAMPLE_REVIEWS, SENTIMENT_COLORS, highlight_html
from absa.serving.service import ModelService

st.set_page_config(page_title="ABSA Demo", page_icon="🍽️", layout="centered")


@st.cache_resource(show_spinner="Loading model…")
def _load_service() -> ModelService:
    service = ModelService.load(get_settings())
    service.warmup()
    return service


def _legend() -> str:
    items = " &nbsp; ".join(
        f'<span style="border-bottom:3px solid {c};padding:0 4px">{s}</span>'
        for s, c in SENTIMENT_COLORS.items()
    )
    return f"<small>sentiment: {items}</small>"


def main() -> None:
    st.title("🍽️ Aspect-Based Sentiment Analysis")
    st.caption(
        "Type a product or restaurant review — aspects are extracted and "
        "highlighted, colored by their sentiment."
    )
    service = _load_service()

    if "review" not in st.session_state:
        st.session_state.review = EXAMPLE_REVIEWS[0]

    st.write("**Try an example:**")
    cols = st.columns(len(EXAMPLE_REVIEWS))
    for i, example in enumerate(EXAMPLE_REVIEWS):
        if cols[i].button(f"#{i + 1}", help=example, use_container_width=True):
            st.session_state.review = example

    text = st.text_area("Review", key="review", height=120)
    analyze = st.button("Analyze", type="primary")

    if analyze or text.strip():
        predictions = service.predict(text)
        st.markdown("#### Result")
        st.markdown(_legend(), unsafe_allow_html=True)
        st.markdown(
            f'<div style="font-size:1.15rem;line-height:2;margin:0.5rem 0">'
            f"{highlight_html(text, predictions)}</div>",
            unsafe_allow_html=True,
        )
        if predictions:
            frame = pd.DataFrame([p.to_dict() for p in predictions])
            st.dataframe(frame, use_container_width=True, hide_index=True)
        else:
            st.info("No aspects detected in this text.")
        st.caption(f"model: **{service.track}** · device: **{service.device}** · offline")


main()
