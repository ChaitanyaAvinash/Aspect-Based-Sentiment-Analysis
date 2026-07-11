"""Streamlit demo: offline CPU ABSA with a fast/accurate model switch."""

from __future__ import annotations

import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import pandas as pd
import streamlit as st

from absa.config import PROJECT_ROOT, get_settings
from absa.serving.render import EXAMPLE_REVIEWS, SENTIMENT_COLORS, highlight_html
from absa.serving.service import ModelService

st.set_page_config(page_title="ABSA Demo", layout="centered")


def _model_options() -> dict[str, str]:
    settings = get_settings()
    options: dict[str, str] = {}
    if (settings.resolve(settings.artifact_path) / "meta.json").exists():
        options["Fast (int8 bert, CPU)"] = "int8"
    if (PROJECT_ROOT / "artifacts" / "transformer" / "meta.json").exists():
        options["Accurate (deberta-v3, CPU)"] = "deberta"
    return options


@st.cache_resource(show_spinner="Loading model...")
def _load(kind: str) -> ModelService:
    if kind == "deberta":
        from absa.models.transformer import TransformerABSA

        model = TransformerABSA.load(PROJECT_ROOT / "artifacts" / "transformer", device="cpu")
        service = ModelService(model=model, track="deberta-v3 (fp32)", device="cpu")
    else:
        service = ModelService.load(get_settings())
    service.warmup()
    return service


def _legend() -> str:
    swatches = "&nbsp;&nbsp;".join(
        f'<span style="border-bottom:3px solid {c};padding:0 4px">{s}</span>'
        for s, c in SENTIMENT_COLORS.items()
    )
    return f"<small>sentiment: {swatches}</small>"


def main() -> None:
    st.title("Aspect-Based Sentiment Analysis")
    st.caption("Type a review; aspects are highlighted and colored by their sentiment.")

    options = _model_options()
    if not options:
        st.error("No trained model found. Run `make export` (or `make train`) first.")
        return
    choice = st.sidebar.radio("Model", list(options))
    service = _load(options[choice])
    st.sidebar.caption(f"track: {service.track} | device: {service.device} | offline")

    if "review" not in st.session_state:
        st.session_state.review = EXAMPLE_REVIEWS[0]

    st.write("**Try an example:**")
    cols = st.columns(len(EXAMPLE_REVIEWS))
    for i, example in enumerate(EXAMPLE_REVIEWS):
        if cols[i].button(f"#{i + 1}", help=example, use_container_width=True):
            st.session_state.review = example

    text = st.text_area("Review", key="review", height=120)
    if st.button("Analyze", type="primary") or text.strip():
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


main()
