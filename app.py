import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import numpy as np

# Load the pre-trained model
@st.cache_resource
def load_fracture_model():
    model = load_model("model2.h5")  # Make sure to place the saved model in the correct path
    return model

model = load_fracture_model()

# Set up the app
st.title("🦴 Bone Fracture Detection")
st.write("Upload an X-ray image to check for bone fractures.")

# File uploader
uploaded_file = st.file_uploader("Choose an X-ray image...", type=["jpg", "jpeg", "png"])

# When an image is uploaded
if uploaded_file is not None:
    # Display the image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded X-ray", use_column_width=True)

    # Preprocess the image
    image_resized = image.resize((224, 224))  # Resize to 224x224 (the size your model expects)
    image_array = img_to_array(image_resized)
    image_array = image_array / 255.0  # Normalize the image
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension

    # Make prediction
    prediction = model.predict(image_array)[0]
    predicted_class = np.argmax(prediction)  # Get the index of the class with the highest score

    # Show result based on the predicted class
    st.write("🔍 Predicting...")

    if predicted_class == 0:
        st.error(f"💥 Uncertain (Confidence: {prediction[0]:.2f})")
    elif predicted_class == 1:
        st.success(f"✅ Fractured (Confidence: {prediction[1]:.2f})")
    else:
        st.warning(f"⚠️ Not Fractured (Confidence: {prediction[2]:.2f})")
