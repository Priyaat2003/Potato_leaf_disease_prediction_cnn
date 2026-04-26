import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

# -------------------------------
# Load Model (FIXED)
# -------------------------------
model = load_model('cnn_model_clean.h5', compile=False)

# -------------------------------
# App Title
# -------------------------------
st.title("🥔 Potato Leaf Disease Detection")

st.markdown("Upload a potato leaf image to detect disease using CNN model")

# -------------------------------
# File Upload
# -------------------------------
uploaded_file = st.file_uploader("Choose an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    
    # Display Image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # -------------------------------
    # Preprocessing
    # -------------------------------
    img = image.resize((224, 224))
    img = np.array(img)

    # Handle grayscale or RGBA
    if len(img.shape) == 2:
        img = np.stack((img,) * 3, axis=-1)

    if img.shape[-1] == 4:
        img = img[:, :, :3]

    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    # -------------------------------
    # Prediction
    # -------------------------------
    prediction = model.predict(img)

    class_names = [
        "Early Blight",
        "Late Blight",
        "Healthy"
    ]

    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction) * 100

    # -------------------------------
    # Output
    # -------------------------------
    st.success(f"Prediction: {predicted_class}")
    st.info(f"Confidence: {confidence:.2f}%")

    # -------------------------------
    # Extra Info (Nice UI)
    # -------------------------------
    if predicted_class == "Healthy":
        st.write("✅ The plant looks healthy.")
    else:
        st.write("⚠️ Disease detected. Consider treatment.")

else:
    st.warning("Please upload an image to get prediction.")