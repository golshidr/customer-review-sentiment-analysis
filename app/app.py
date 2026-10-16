"""Interactive demo: classify a review and show which words drove the prediction.

Run after training (``make classical`` and optionally ``make transformers``):

    python app/app.py                      # http://127.0.0.1:7860
    python app/app.py --models tfidf_lr_tuned distilbert --share

Classical models are explained with their exact per-word contributions; transformers with
Integrated Gradients (fewer steps than in the offline analysis, to keep the demo responsive).
"""

from __future__ import annotations

import argparse
from functools import lru_cache

from scipy.special import expit

from review_sentiment import explain
from review_sentiment.paths import ARTIFACTS_DIR
from review_sentiment.plotting import MODEL_LABELS
from review_sentiment.predictors import load_predictor

EXAMPLES = [
    "The plot was predictable, but the performances were so good that I didn't care.",
    "I wanted to love this phone. Battery life is terrible and support never answered.",
    "Not bad at all - the staff were friendly and the food arrived quickly.",
    "This is not a great film, but it is not a waste of time either.",
]


@lru_cache(maxsize=4)
def get_predictor(name: str):
    return load_predictor(name)


def analyse(text: str, model_name: str, ig_steps: int = 20, threshold: float = 0.15):
    """Return class probabilities and words tagged '+' / '-' when their attribution exceeds
    ``threshold`` times the largest absolute attribution in the review."""
    text = (text or "").strip()
    if not text:
        return {}, []
    predictor = get_predictor(model_name)
    p_pos = float(expit(predictor.margin([text])[0]))
    if predictor.kind == "classical":
        words, scores = explain.lr_attributions(predictor.pipeline, text)
    else:
        words, scores, _ = explain.integrated_gradients(predictor, text, n_steps=ig_steps)
    peak = max((abs(float(s)) for s in scores), default=0.0) or 1.0
    highlighted = []
    for w, s in zip(words, scores, strict=True):
        rel = float(s) / peak
        tag = "+" if rel > threshold else ("-" if rel < -threshold else None)
        highlighted.append((w + " ", tag))
    return {"positive": p_pos, "negative": 1 - p_pos}, highlighted


def build_demo(models: list[str]):
    import gradio as gr

    choices = [(MODEL_LABELS.get(m, m), m) for m in models]
    with gr.Blocks(title="Review sentiment explorer") as demo:
        gr.Markdown(
            "## Review sentiment explorer\n"
            "Models were trained on IMDb movie reviews only. Word colours show each word's "
            "contribution: blue (+) pushes towards *positive*, red (-) towards *negative*; "
            "uncoloured words had little influence."
        )
        with gr.Row():
            with gr.Column(scale=3):
                text = gr.Textbox(lines=6, label="Review", value=EXAMPLES[0])
                model = gr.Dropdown(choices=choices, value=models[0], label="Model")
                button = gr.Button("Analyse", variant="primary")
            with gr.Column(scale=2):
                probs = gr.Label(num_top_classes=2, label="Prediction")
        words = gr.HighlightedText(
            label="Word attributions",
            color_map={"+": "#2a78d6", "-": "#e34948"},
            show_legend=False,
        )
        gr.Examples(EXAMPLES, inputs=text)
        button.click(analyse, inputs=[text, model], outputs=[probs, words])
    return demo


def available_models() -> list[str]:
    order = ["tfidf_lr_tuned", "distilbert", "roberta", "deberta", "tfidf_lr_original"]
    return [m for m in order if (ARTIFACTS_DIR / m).exists()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    models = args.models or available_models()
    if not models:
        raise SystemExit("No trained models found in artifacts/. Run `make classical` first.")
    build_demo(models).launch(share=args.share, server_port=args.port)


if __name__ == "__main__":
    main()
