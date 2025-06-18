import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json

# Load model
model = tf.keras.models.load_model("your_model.h5")

with open("class_names.json", "r") as f:
    class_names = json.load(f)

# UI
st.title("🍛 Indian Food Image Classifier")
st.write("Upload an image and get its predicted dish!")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess image
    image = image.resize((224, 224))
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction)

    st.markdown(f"### 🍽️ Predicted Dish: **{predicted_class}**")
    st.markdown(f"### 🔍 Confidence: **{confidence:.2f}**")
