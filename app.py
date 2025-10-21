from __future__ import annotations

from pathlib import Path

import streamlit as st

from src.align.mutual_align import EmbeddingAligner
from src.translation.translator import get_translator
from src.text.predict_text import predict_text_cefr_ensemble


def main() -> None:
    st.set_page_config(page_title="Kazakh CEFR Text Classifier", layout="wide")
    st.title("Kazakh CEFR Text Classifier")

    kaz_text = st.text_area("Enter Kazakh text:", height=200)
    russian_override = st.text_input("Optional: provide Russian translation override")
    model_path = st.text_input("Russian sentence model checkpoint", value="models/ru_cefr_sentence")
    weight = st.slider("Russian model weight α", min_value=0.0, max_value=1.0, value=0.6, step=0.05)

    if st.button("Classify") and kaz_text.strip():
        try:
            result = predict_text_cefr_ensemble(
                kaz_text,
                translator=get_translator(),
                aligner=EmbeddingAligner(),
                russian_text=russian_override or None,
                russian_model_checkpoint=Path(model_path),
                russian_weight=weight,
            )
        except Exception as exc:  # pragma: no cover
            st.error(f"Failed to classify text: {exc}")
            return

        st.subheader("Prediction")
        st.write(f"CEFR level: **{result['level']}** (confidence {result['confidence']:.2%})")
        st.write("Translation:")
        st.info(result["translation"])

        st.subheader("Probability Distribution")
        st.bar_chart(result["probabilities"])

        st.subheader("Kazakh Word-Level Distribution")
        st.json(result["kazakh_distribution"])

        st.subheader("Russian Sentence Model Distribution")
        st.json(result["russian_distribution"])

        st.subheader("Word Alignments")
        st.dataframe(result["word_alignments"])


if __name__ == "__main__":
    main()
