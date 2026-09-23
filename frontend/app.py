"""Streamlit demonstration interface for ChestXpert.

Run:
    streamlit run frontend/app.py

The interface is intentionally framed as a research/educational demo.
It does not provide a diagnosis.
"""

import io
import os

import numpy as np
import streamlit as st
from PIL import Image

from app.inference.predict import ChestXpertPredictor
from app.explainability.gradcam import GradCAM


st.set_page_config(
    page_title="ChestXpert",
    page_icon="🩻",
    layout="wide",
)

st.title("ChestXpert")
st.caption("Explainable Chest X-Ray AI — research / educational prototype")

st.warning(
    "This system is not a medical device and does not provide a diagnosis. "
    "All outputs require qualified clinician review."
)

checkpoint = st.sidebar.text_input(
    "Model checkpoint",
    os.getenv("CHESTXPERT_CHECKPOINT", "models/checkpoints/resnet18_latest.pt"),
)

uploaded = st.file_uploader(
    "Upload a chest X-ray",
    type=["png", "jpg", "jpeg"],
)

threshold = st.slider(
    "Flagging threshold",
    min_value=0.05,
    max_value=0.95,
    value=0.50,
    step=0.05,
)

if uploaded is None:
    st.info("Upload a chest X-ray to run the model.")
    st.stop()

image = Image.open(io.BytesIO(uploaded.getvalue())).convert("RGB")

if not os.path.exists(checkpoint):
    st.error(
        "No model checkpoint was found. Train the model first and provide "
        "the checkpoint path in the sidebar."
    )
    st.image(image, caption="Uploaded X-ray", use_container_width=True)
    st.stop()

try:
    predictor = ChestXpertPredictor(checkpoint)
    results = predictor.predict_from_image(image, threshold=threshold)
except AttributeError:
    # Predictor currently accepts a path; create a temporary upload safely.
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".png") as temp:
        image.save(temp.name)
        results = predictor.predict(temp.name, threshold=threshold)
except Exception as exc:
    st.error(f"Inference failed: {exc}")
    st.stop()

left, right = st.columns(2)

with left:
    st.subheader("Uploaded X-ray")
    st.image(image, use_container_width=True)

with right:
    st.subheader("Model output")
    for result in results:
        probability = result["probability"]
        st.write(f"**{result['finding']}** — {probability:.1%}")
        st.progress(float(probability))

flagged = [r for r in results if r["flagged"]]

st.subheader("Flagged findings")
if flagged:
    for result in flagged:
        st.write(
            f"- **{result['finding']}** — {result['probability']:.1%}"
        )
else:
    st.write("No configured finding crossed the selected threshold.")

st.info(
    "Probability scores represent model output, not clinical certainty. "
    "The model should be reviewed by a qualified clinician."
)
