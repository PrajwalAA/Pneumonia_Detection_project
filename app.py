import os
import numpy as np
import streamlit as st
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from PIL import Image # Used for opening image files from Streamlit uploader

# --- Configuration ---
# Path to your trained Keras model file
# IMPORTANT: Ensure 'pneumonia_model.h5' is in the same directory as this Streamlit script
MODEL_PATH = "pneumonia_model.keras" 

# Define image size (must match the input size of your model)
IMG_HEIGHT, IMG_WIDTH = 150, 150

# --- Streamlit UI Setup ---
st.set_page_config(
    page_title="Pneumonia X-Ray Predictor",
    page_icon="🩺",
    layout="centered",
    initial_sidebar_state="auto",
)

st.title("Pneumonia X-Ray Predictor")
st.markdown(
    """
    Upload chest X-ray images below to get a prediction on whether they show signs of Pneumonia or are Normal.
    """
)

# --- Load the trained model ---
@st.cache_resource # Cache the model loading to avoid reloading on every rerun
def get_model():
    try:
        model = load_model(MODEL_PATH)
        st.success(f"Model '{MODEL_PATH}' loaded successfully.")
        return model
    except Exception as e:
        st.error(f"Error loading model from '{MODEL_PATH}': {e}")
        st.warning("Please ensure 'pneumonia_model.h5' is in the same directory as this script.")
        st.stop() # Stop the app if model cannot be loaded
    return None # Should not be reached

model = get_model()

# --- File Uploader ---
uploaded_files = st.file_uploader(
    "Choose X-Ray Image(s)",
    type=["png", "jpg", "jpeg"],
    accept_multiple_files=True,
    help="Select one or more chest X-ray images for prediction."
)

# --- Prediction and Display Function ---
def predict_and_display_image(uploaded_file, model_instance):
    """
    Processes a single uploaded image, makes a prediction, and displays
    the image along with its prediction using Streamlit.
    """
    file_name = uploaded_file.name
    
    try:
        # --- Preprocess image ---
        # Open the image file using PIL
        img = Image.open(uploaded_file)
        
        # Resize and convert to array for model input
        # Note: image.load_img directly works with file paths, so we convert PIL Image to array
        # and then resize using PIL for consistency with target_size
        img = img.resize((IMG_WIDTH, IMG_HEIGHT))
        img_array = image.img_to_array(img)
        
        # Normalize pixel values
        img_array = img_array / 255.0
        
        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)

        # --- Make prediction ---
        prediction = model_instance.predict(img_array)
        
        # Determine result and confidence based on model output
        result = "PNEUMONIA" if prediction[0][0] > 0.5 else "NORMAL"
        confidence = prediction[0][0] if prediction[0][0] > 0.5 else 1 - prediction[0][0]

        # --- Display image and prediction ---
        st.image(uploaded_file, caption=file_name, use_column_width=True)

        # Display prediction results with colored text
        if result == "PNEUMONIA":
            st.markdown(f"#### 🛑 Prediction: <span style='color:red; font-weight:bold;'>{result}</span>", unsafe_allow_html=True)
        else:
            st.markdown(f"#### ✅ Prediction: <span style='color:green; font-weight:bold;'>{result}</span>", unsafe_allow_html=True)
            
        st.write(f"Confidence: {confidence * 100:.2f}%")
        st.markdown("---") # Separator for multiple images

    except Exception as e:
        st.error(f"Could not process image '{file_name}': {e}")
        st.markdown("---")

# --- Process Uploaded Files ---
if uploaded_files:
    st.header("Prediction Results:")
    for uploaded_file in uploaded_files:
        predict_and_display_image(uploaded_file, model)
else:
    st.info("Upload an image above to see the prediction result.")

st.markdown(
    """
    <style>
    .reportview-container .main .block-container {
        padding-top: 2rem;
        padding-right: 2rem;
        padding-left: 2rem;
        padding-bottom: 2rem;
    }
    .stFileUploader {
        margin-top: 20px;
        margin-bottom: 20px;
    }
    </style>
    """, unsafe_allow_html=True
)
