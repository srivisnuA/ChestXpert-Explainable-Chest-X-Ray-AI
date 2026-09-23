"""Streamlit demonstration interface for ChestXpert.

Run:
    streamlit run frontend/streamlit_app.py

The interface is intentionally framed as a research/educational demo.
It does not provide a diagnosis.
"""

import os
import tempfile

import streamlit as st
from PIL import Image

from app.explainability.overlay import cam_to_overlay
from app.inference.predict import ChestXpertPredictor
from app.preprocessing.dicom import dicom_to_pil


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

threshold = st.sidebar.slider(
    "Flagging threshold",
    min_value=0.05,
    max_value=0.95,
    value=0.50,
    step=0.05,
)

uploaded = st.file_uploader(
    "Upload a chest X-ray",
    type=["png", "jpg", "jpeg", "dcm"],
)

if uploaded is None:
    st.info("Upload a chest X-ray image or DICOM file to run the model.")
    st.stop()

try:
    if uploaded.name.lower().endswith(".dcm"):
        with tempfile.NamedTemporaryFile(suffix=".dcm", delete=False) as temp:
            temp.write(uploaded.getvalue())
            dicom_path = temp.name
        try:
            image = dicom_to_pil(dicom_path)
        finally:
            os.unlink(dicom_path)
    else:
        image = Image.open(uploaded).convert("RGB")
except Exception as exc:
    st.error(f"Could not read the uploaded image: {exc}")
    st.stop()

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
        st.write(f"- **{result['finding']}** — {result['probability']:.1%}")
else:
    st.write("No configured finding crossed the selected threshold.")

st.divider()
st.subheader("Grad-CAM explanation")

st.caption(
    "Grad-CAM highlights image regions associated with a selected model output. "
    "It is an interpretability aid, not proof that a finding is present."
)

finding_options = [result["finding"] for result in results]
selected_finding = st.selectbox(
    "Finding to explain",
    finding_options,
    index=0,
)

try:
    cam = predictor.explain_finding(image, selected_finding)
    overlay = cam_to_overlay(image, cam.numpy(), alpha=0.45)

    explain_left, explain_right = st.columns(2)
    with explain_left:
        st.image(
            image,
            caption="Original X-ray",
            use_container_width=True,
        )
    with explain_right:
        st.image(
            overlay,
            caption=f"Grad-CAM — {selected_finding}",
            use_container_width=True,
        )
except Exception as exc:
    st.error(f"Grad-CAM generation failed: {exc}")

st.info(
    "Probability scores represent model output, not clinical certainty. "
    "The model should be reviewed by a qualified clinician."
)
