import streamlit as st
import tensorflow as tf
import numpy as np
import json
from PIL import Image

# Page configuration
st.set_page_config(
    page_title="Plant Disease Detection System",
    page_icon="🌿",
    layout="wide"
)

# Load model
model = tf.keras.models.load_model("trained_model.keras")

# Load disease information
with open("disease_info.json") as f:
    disease_data = json.load(f)

# Class names
class_name = [
'Apple___Apple_scab',
'Apple___Black_rot',
'Apple___Cedar_apple_rust',
'Apple___healthy',
'Blueberry___healthy',
'Cherry_(including_sour)___Powdery_mildew',
'Cherry_(including_sour)___healthy',
'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
'Corn_(maize)___Common_rust_',
'Corn_(maize)___Northern_Leaf_Blight',
'Corn_(maize)___healthy',
'Grape___Black_rot',
'Grape___Esca_(Black_Measles)',
'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
'Grape___healthy',
'Orange___Haunglongbing_(Citrus_greening)',
'Peach___Bacterial_spot',
'Peach___healthy',
'Pepper,_bell___Bacterial_spot',
'Pepper,_bell___healthy',
'Potato___Early_blight',
'Potato___Late_blight',
'Potato___healthy',
'Raspberry___healthy',
'Soybean___healthy',
'Squash___Powdery_mildew',
'Strawberry___Leaf_scorch',
'Strawberry___healthy',
'Tomato___Bacterial_spot',
'Tomato___Early_blight',
'Tomato___Late_blight',
'Tomato___Leaf_Mold',
'Tomato___Septoria_leaf_spot',
'Tomato___Spider_mites Two-spotted_spider_mite',
'Tomato___Target_Spot',
'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
'Tomato___Tomato_mosaic_virus',
'Tomato___healthy'
]

# Prediction function
def model_prediction(image):
    img = image.resize((128,128))
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)

    result_index = np.argmax(prediction)
    confidence = np.max(prediction) * 100

    return result_index, confidence

# Sidebar
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page",["Home","About","Disease Recognition"])

# Home Page
if(app_mode=="Home"):

    st.header("🌿 Plant Disease Recognition System")

    st.markdown("""
Welcome to the **Plant Disease Recognition System**.

This AI-powered web application helps farmers and researchers detect plant diseases using leaf images.

### How It Works
1️⃣ Upload a plant leaf image  
2️⃣ AI model analyzes the image  
3️⃣ Disease is predicted instantly  
4️⃣ Get disease description, treatment, and prevention tips  

### Benefits
✔ Early disease detection  
✔ Helps farmers protect crops  
✔ AI-based fast diagnosis  
""")

# About Page
elif(app_mode=="About"):

    st.header("About This Project")

    st.markdown("""
This project uses **Deep Learning (CNN)** to detect plant diseases from leaf images.

### Dataset
The dataset contains **87,000+ images** of healthy and diseased plant leaves categorized into **38 classes**.

Dataset split:
- Training set
- Validation set
- Test set

### Technologies Used
- Python
- TensorFlow / Keras
- Streamlit
- OpenCV
- NumPy
""")

# Prediction Page
elif(app_mode=="Disease Recognition"):

    st.header("🔍 Plant Disease Detection")

    uploaded_file = st.file_uploader(
        "Upload a plant leaf image",
        type=["jpg","jpeg","png"]
    )

    if uploaded_file is not None:

        image = Image.open(uploaded_file)

        col1, col2 = st.columns(2)

        with col1:
            st.image(image, caption="Uploaded Image", use_column_width=True)

        with col2:
            st.write("Click the button below to analyze the plant leaf.")

            if st.button("Predict Disease"):

                with st.spinner("Analyzing image..."):

                    result_index, confidence = model_prediction(image)

                    disease = class_name[result_index]

                    display_disease = disease.replace("___"," - ").replace("_"," ")

                    st.success(f"🌱 Disease Detected: {display_disease}")

                    st.info(f"Prediction Confidence: {confidence:.2f}%")

                    if disease in disease_data:

                        st.subheader("📖 Disease Description")
                        st.write(disease_data[disease]["description"])

                        st.subheader("💊 Cure / Treatment")
                        st.write(disease_data[disease]["cure"])

                        st.subheader("⚠ Prevention Tips")
                        st.write(disease_data[disease]["prevention"])

                        st.subheader("🛒 Recommended Medicine")

                        st.markdown(
                         f"[🛒 Buy Recommended Medicine]({disease_data[disease]['medicine_link']})",
                          unsafe_allow_html=True
                        )

                    else:
                        st.warning("No additional information available for this disease.")