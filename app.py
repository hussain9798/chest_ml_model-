import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

# =========================
# Load Trained Model
# =========================
@st.cache_resource
def load_my_model():
    model = load_model("medical_scan_model.h5", compile=False)
    return model

model = load_my_model()

# =========================
# Streamlit Page Config
# =========================
st.set_page_config(
    page_title="Medical Scan Detection",
    page_icon="🩺",
    layout="centered"
)

# =========================
# Title
# =========================
st.title("🩺 Medical Scan Detection App")
st.write("Upload a medical scan image and the AI model will predict the result.")

# =========================
# Upload Image
# =========================
uploaded_file = st.file_uploader(
    "Choose a medical scan image",
    type=["jpg", "jpeg", "png"]
)

# =========================
# Prediction Function
# =========================
def preprocess_image(image):

    # Convert grayscale to RGB
    image = image.convert("RGB")

    # Resize image
    image = image.resize((224, 224))

    # Convert image to numpy array
    img_array = np.array(image)

    # Normalize image
    img_array = img_array / 255.0

    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)

    return img_array

# =========================
# Predict
# =========================
if uploaded_file is not None:

    # Open image
    image = Image.open(uploaded_file)

    # Display image
    st.image(image, caption="Uploaded Image", width=300)

    # Preprocess image
    processed_image = preprocess_image(image)

    # Prediction
    prediction = model.predict(processed_image)

    # Confidence score
    confidence = float(prediction[0][0])

    st.subheader("Prediction Result")

    # Binary Classification
    if confidence > 0.5:
        st.error("⚠️ Disease Detected")
        st.write(f"Confidence Score: {confidence:.2f}")
    else:
        st.success("✅ Normal Scan")
        st.write(f"Confidence Score: {1 - confidence:.2f}")

# =========================
# Footer
# =========================
st.markdown("---")
st.write("Built with Streamlit + TensorFlow")