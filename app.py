import streamlit as st
from PIL import Image
import numpy as np

# ------------------------------
# PAGE CONFIG
# ------------------------------
st.set_page_config(
    page_title="Eco-Friendly Product Identifier",
    page_icon="🌿",
    layout="wide"
)

# ------------------------------
# HEADER SECTION
# ------------------------------
st.title("🌿 Eco-Friendly Product Identification")
st.markdown("""
This is a **Streamlit-based web application** where you can upload a product image 
and classify whether it is *eco-friendly* using a **CNN model**.
""")

st.write("---")

# ------------------------------
# SIDEBAR
# ------------------------------
with st.sidebar:
    st.header("📌 Navigation")
    page = st.radio("Go to:", ["Home", "Upload & Predict", "About Project"])

# ------------------------------
# LOAD MODEL (Dummy loader – replace with your model)
# ------------------------------
@st.cache_resource
def load_model():
    # Example: from tensorflow.keras.models import load_model
    # model = load_model("model.h5")
    return None

model = load_model()

# ------------------------------
# PREDICTION FUNCTION
# ------------------------------
def predict_image(img):
    # Replace with actual prediction logic

    # Example preprocessing
    img = img.resize((150, 150))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Example fake output (Replace with: model.predict(img_array))
    prediction = np.random.rand()

    if prediction > 0.5:
        return "Eco-Friendly 🌿", prediction
    else:
        return "Not Eco-Friendly ❌", prediction

# ------------------------------
# PAGES
# ------------------------------

if page == "Home":
    st.header("🏠 Home")
    st.write("""
    Welcome to the Eco-Friendly Product Identifier Web App.  
    Use the navigation menu to upload a product image and classify it.
    """)
    st.image("https://images.unsplash.com/photo-1524593119774-0f0654e20314", use_column_width=True)

elif page == "Upload & Predict":
    st.header("📤 Upload Product Image & Predict")

    uploaded_file = st.file_uploader("Upload an image (JPG/PNG)", type=["jpg", "png", "jpeg"])

    if uploaded_file:
        img = Image.open(uploaded_file)
        st.image(img, caption="Uploaded Image", width=300)

        if st.button("🔍 Predict"):
            label, confidence = predict_image(img)
            st.success(f"Prediction: **{label}**")
            st.info(f"Confidence: {confidence:.2f}")

elif page == "About Project":
    st.header("ℹ About the Project")
    st.markdown("""
    **Project Name:** Eco-Friendly Product Identifier  
    **Technology Used:**  
    - Python  
    - Streamlit  
    - TensorFlow/Keras CNN  
    - Image processing (PIL, NumPy)  
    - Google Colab for training  

    **Goal:**  
    To detect whether a product is eco-friendly based on its visual features.
    """)

