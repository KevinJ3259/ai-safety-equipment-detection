import io

import streamlit as st
from ultralytics import YOLO
from PIL import Image


st.set_page_config(
    page_title="AI Safety Equipment Detector",
    page_icon="👷",
    layout="wide",
)

st.title("👷 AI Safety Equipment Detection System")

st.write(
    "Upload an image to detect hard hats and safety vests "
    "using a custom-trained YOLO11 computer vision model."
)

# Load custom-trained model
MODEL_PATH = "runs/detect/training/hardhat_vest_v2/weights/best.pt"


@st.cache_resource
def load_model():
    return YOLO(MODEL_PATH)


model = load_model()

# Detection settings
st.sidebar.header("Detection Settings")

confidence = st.sidebar.slider(
    "Confidence Threshold",
    min_value=0.10,
    max_value=1.00,
    value=0.50,
    step=0.05,
)

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"],
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    # Run custom YOLO model
    results = model.predict(
        source=image,
        conf=confidence,
    )

    result = results[0]

    # result.plot() returns a BGR NumPy array
    annotated_image = result.plot()
    annotated_image = annotated_image[:, :, ::-1]

    # Display original and detected images
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original Image")
        st.image(image, width="stretch")

    with col2:
        st.subheader("Detection Results")
        st.image(annotated_image, width="stretch")

    # Detection summary
    st.subheader("Detected Safety Equipment")

    if len(result.boxes) == 0:
        st.warning("No safety equipment detected.")
    else:
        detection_counts = {}

        for box in result.boxes:
            class_id = int(box.cls[0])
            class_name = model.names[class_id]
            detection_counts[class_name] = (
                detection_counts.get(class_name, 0) + 1
            )

        for class_name, count in detection_counts.items():
            st.write(f"**{class_name}:** {count}")

        st.success(f"Total detections: {len(result.boxes)}")

        # Convert annotated image to a downloadable JPEG
        download_image = Image.fromarray(annotated_image)
        buffer = io.BytesIO()
        download_image.save(buffer, format="JPEG", quality=95)

        st.download_button(
            label="Download Detection Result",
            data=buffer.getvalue(),
            file_name="detection_result.jpg",
            mime="image/jpeg",
        )
