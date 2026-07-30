import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import (
    MobileNetV2,
    preprocess_input,
    decode_predictions,
)

st.set_page_config(
    page_title="Image Recognition - MobileNetV2",
    page_icon="🖼️",
    layout="centered",
)

@st.cache_resource
def load_model():
    return MobileNetV2(weights="imagenet")

model = load_model()

st.title("🖼️ Image Recognition")
st.write(
    "Upload a photo and MobileNetV2 will predict what object is present. "
    "This project uses transfer learning with a model already trained on ImageNet."
)

with st.expander("How this project works"):
    st.markdown("""
    **Workflow**

    `Image → Resize & Preprocess → MobileNetV2 → Class Probabilities → Top Predictions`

    - **Computer Vision:** the computer processes visual information.
    - **Deep Learning:** a neural network learns visual patterns.
    - **Transfer Learning:** we reuse MobileNetV2 instead of training a model from scratch.
    - **ImageNet:** the pretrained model recognizes 1,000 common object categories.
    """)

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png", "webp"],
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    st.subheader("Uploaded Image")
    st.image(image, caption="Your image", use_container_width=True)

    # MobileNetV2 expects a 224 x 224 image.
    resized = image.resize((224, 224))

    # Convert PIL image to NumPy array and add batch dimension.
    image_array = np.array(resized, dtype=np.float32)
    image_array = np.expand_dims(image_array, axis=0)

    # MobileNetV2 preprocessing scales pixels into the range expected by the model.
    image_array = preprocess_input(image_array)

    with st.spinner("Analyzing image..."):
        predictions = model.predict(image_array, verbose=0)
        decoded = decode_predictions(predictions, top=5)[0]

    st.subheader("🎯 Prediction Results")

    best_label = decoded[0][1].replace("_", " ")
    best_probability = decoded[0][2] * 100

    st.success(
        f"Most likely object: **{best_label.title()}** "
        f"({best_probability:.2f}% confidence)"
    )

    st.write("### Top 5 predictions")

    for rank, (_, label, probability) in enumerate(decoded, start=1):
        label = label.replace("_", " ").title()
        percentage = probability * 100

        st.write(f"**{rank}. {label}** — {percentage:.2f}%")
        st.progress(float(probability))

else:
    st.info("👆 Upload an image to start recognition.")

st.divider()
st.caption("Project 2 — Image Recognition | MobileNetV2 + ImageNet + Transfer Learning")
