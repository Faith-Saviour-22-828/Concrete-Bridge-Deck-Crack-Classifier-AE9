"""
Concrete Bridge Deck Crack Detection - Streamlit App
GET 324 Mini-Project (Laboratory Exercise 10)

Run locally with:  streamlit run app.py
"""

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

st.set_page_config(
    page_title="Concrete Crack Detector",
    page_icon="\U0001F309",
    layout="centered",
)

IMG_SIZE = (128, 128)
MODEL_PATH = "model/crack_detector.keras"
CLASS_NAMES = ["Non-cracked", "Cracked"]


@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)


def preprocess_image(pil_image: Image.Image) -> np.ndarray:
    img = pil_image.convert("RGB").resize(IMG_SIZE)
    arr = tf.keras.utils.img_to_array(img)
    arr = np.expand_dims(arr, axis=0)
    return arr


def predict(model, pil_image: Image.Image):
    arr = preprocess_image(pil_image)
    prob_cracked = float(model.predict(arr, verbose=0)[0][0])
    label = CLASS_NAMES[1] if prob_cracked >= 0.5 else CLASS_NAMES[0]
    confidence = prob_cracked if prob_cracked >= 0.5 else 1 - prob_cracked
    return label, confidence, prob_cracked


def main():
    st.title("\U0001F309 Concrete Bridge Deck Crack Detector")
    st.write(
        "Upload a photo of a concrete surface (e.g. a bridge deck) and the model "
        "will classify it as **Cracked** or **Non-cracked**."
    )

    with st.sidebar:
        st.header("About")
        st.markdown(
            "- **Course:** GET 324 — Laboratory Exercise 10\n"
            "- **Task:** Binary image classification (cracked vs non-cracked)\n"
            "- **Model:** CNN / MobileNetV2 transfer learning\n"
            "- **Dataset:** Concrete Crack Images for Classification "
            "(Ozgenel & Gonenc Sorguc, Mendeley Data)"
        )

    model = load_model()

    uploaded_file = st.file_uploader(
        "Choose an image...", type=["jpg", "jpeg", "png"]
    )

    col1, col2 = st.columns(2)

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        with col1:
            st.image(image, caption="Uploaded image", use_container_width=True)

        with st.spinner("Analysing image..."):
            label, confidence, prob_cracked = predict(model, image)

        with col2:
            if label == "Cracked":
                st.error(f"### \U0001F6A8 {label}")
            else:
                st.success(f"### \u2705 {label}")
            st.metric("Confidence", f"{confidence * 100:.1f}%")
            st.progress(prob_cracked)
            st.caption(f"Raw model output (P(cracked) = {prob_cracked:.3f})")

        st.info(
            "\u26A0\uFE0F This tool supports visual inspection but does not replace "
            "professional structural assessment of bridge decks."
        )
    else:
        st.write("\u2b06\ufe0f Upload an image to get a prediction.")


if __name__ == "__main__":
    main()
