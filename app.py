import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

# Load the trained model
model = tf.keras.models.load_model("mnist_model.h5")

st.title("🧠 Digit Classifier")
st.write("Upload a 28x28 grayscale image of a handwritten digit.")

uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert('L').resize((28, 28))
    img_array = 255 - np.array(image)  # Invert if needed
    img_array = img_array / 255.0
    img_array = img_array.reshape(1, 28, 28, 1)

    st.image(image, caption="Uploaded Image", width=150)

    prediction = model.predict(img_array)
    st.success(f"Predicted Digit: {np.argmax(prediction)}")
