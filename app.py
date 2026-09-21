import streamlit as st
import torch
from torchvision import models, transforms
from PIL import Image


# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="AI Image Recognition",
    page_icon="🖼️",
    layout="centered"
)


# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------

@st.cache_resource
def load_model():

    weights = models.MobileNet_V2_Weights.DEFAULT

    model = models.mobilenet_v2(
        weights=weights
    )

    model.eval()

    return model, weights


model, weights = load_model()


# --------------------------------------------------
# IMAGE PREPROCESSING
# --------------------------------------------------

preprocess = weights.transforms()


# --------------------------------------------------
# TITLE
# --------------------------------------------------

st.title("🖼️ AI Image Recognition")

st.write(
    "Upload an image and MobileNetV2 will predict "
    "what object is present in the image."
)

st.info(
    "This project uses PyTorch, Torchvision and "
    "a pretrained MobileNetV2 model."
)


# --------------------------------------------------
# HOW IT WORKS
# --------------------------------------------------

with st.expander("🔍 How this project works"):

    st.markdown(
        """
        ### Workflow

        **Image → Resize → Preprocessing → MobileNetV2 → Prediction**

        **Technologies used:**

        - Python
        - Streamlit
        - PyTorch
        - Torchvision
        - MobileNetV2
        - Pillow
        - ImageNet

        The model has already been trained on ImageNet,
        so we do not need to train a model from scratch.
        """
    )


# --------------------------------------------------
# IMAGE UPLOAD
# --------------------------------------------------

uploaded_file = st.file_uploader(
    "📤 Choose an image",
    type=[
        "jpg",
        "jpeg",
        "png",
        "webp"
    ]
)


# --------------------------------------------------
# PREDICTION
# --------------------------------------------------

if uploaded_file is not None:

    try:

        # Open image
        image = Image.open(
            uploaded_file
        ).convert("RGB")

        # Display image
        st.subheader("📷 Uploaded Image")

        st.image(
            image,
            caption="Your Image",
            use_container_width=True
        )

        # Image dimensions
        width, height = image.size

        st.write(
            f"**Image Size:** {width} × {height} pixels"
        )

        # --------------------------------------------------
        # PREPROCESS IMAGE
        # --------------------------------------------------

        input_tensor = preprocess(
            image
        )

        # Add batch dimension
        input_batch = input_tensor.unsqueeze(0)

        # --------------------------------------------------
        # MODEL PREDICTION
        # --------------------------------------------------

        with st.spinner(
            "🤖 Analyzing image..."
        ):

            with torch.no_grad():

                output = model(
                    input_batch
                )

        # --------------------------------------------------
        # CONVERT TO PROBABILITIES
        # --------------------------------------------------

        probabilities = torch.nn.functional.softmax(
            output[0],
            dim=0
        )

        # Top 5 predictions
        top_probabilities, top_indices = torch.topk(
            probabilities,
            5
        )

        categories = weights.meta["categories"]

        # --------------------------------------------------
        # MAIN PREDICTION
        # --------------------------------------------------

        best_index = top_indices[0].item()

        best_label = categories[
            best_index
        ]

        best_probability = (
            top_probabilities[0].item()
            * 100
        )

        # --------------------------------------------------
        # DISPLAY MAIN RESULT
        # --------------------------------------------------

        st.subheader("🎯 Prediction")

        st.success(
            f"Most likely object: **{best_label}**"
        )

        st.metric(
            "Confidence",
            f"{best_probability:.2f}%"
        )

        # --------------------------------------------------
        # TOP 5
        # --------------------------------------------------

        st.subheader(
            "📊 Top 5 Predictions"
        )

        for rank in range(5):

            index = top_indices[
                rank
            ].item()

            label = categories[
                index
            ]

            probability = (
                top_probabilities[
                    rank
                ].item()
            )

            percentage = (
                probability * 100
            )

            st.write(
                f"**{rank + 1}. {label}**"
            )

            st.progress(
                float(probability)
            )

            st.caption(
                f"Confidence: {percentage:.2f}%"
            )

        # --------------------------------------------------
        # MODEL INFORMATION
        # --------------------------------------------------

        st.divider()

        st.subheader(
            "ℹ️ Model Information"
        )

        col1, col2 = st.columns(2)

        with col1:

            st.write("**Framework**")
            st.write("PyTorch")

            st.write("**Model**")
            st.write("MobileNetV2")

        with col2:

            st.write("**Dataset**")
            st.write("ImageNet")

            st.write("**Classes**")
            st.write("1000")

    except Exception as error:

        st.error(
            "❌ An error occurred while processing the image."
        )

        st.code(
            str(error)
        )


else:

    st.info(
        "👆 Upload an image to start recognition."
    )


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "AI Image Recognition | "
    "PyTorch + Torchvision + MobileNetV2"
)