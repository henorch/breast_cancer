import streamlit as st
import numpy as np
import joblib

# Page configuration
st.set_page_config(
    page_title="Breast Cancer Early Detection",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load pipeline model
model = joblib.load("break_model.pkl")

# Title
st.title("ML Breast Cancer Early Detection and Support Decision")

st.markdown("Adjust the sliders below to input patient data and click **Predict** to see the result.")

# Create 4 columns with proportional widths
col1, col2, col3, col4 = st.columns([1.2, 1, 1, 1])

with col1:
    mean_radius = st.slider("Mean Radius", 6.981, 28.110, 14.127, 0.01)
    mean_texture = st.slider("Mean Texture", 9.710, 39.280, 19.290, 0.01)
    mean_smoothness = st.slider("Mean Smoothness", 0.05263, 0.1634, 0.09636, 0.001)

with col2:
    mean_compactness = st.slider("Mean Compactness", 0.01938, 0.3454, 0.10434, 0.001)
    mean_symmetry = st.slider("Mean Symmetry", 0.1060, 0.3040, 0.18116, 0.001)
    mean_fractal_dimension = st.slider("Mean Fractal Dimension", 0.04996, 0.09744, 0.06280, 0.001)

with col3:
    radius_error = st.slider("Radius Error", 0.1115, 2.8730, 0.40517, 0.01)
    texture_error = st.slider("Texture Error", 0.3602, 4.8850, 1.21685, 0.01)
    smoothness_error = st.slider("Smoothness Error", 0.001713, 0.03113, 0.007041, 0.001)

with col4:
    compactness_error = st.slider("Compactness Error", 0.002252, 0.1354, 0.025478, 0.001)
    concave_points_error = st.slider("Concave Points Error", 0.0000, 0.05279, 0.011796, 0.001)
    symmetry_error = st.slider("Symmetry Error", 0.007882, 0.07895, 0.020542, 0.001)
    worst_symmetry = st.slider("Worst Symmetry", 0.1565, 0.6638, 0.290076, 0.001)

# Combine features in correct order
features = np.array([
    mean_radius, mean_texture, mean_smoothness, mean_compactness,
    mean_symmetry, mean_fractal_dimension, radius_error,
    texture_error, smoothness_error, compactness_error,
    concave_points_error, symmetry_error, worst_symmetry
]).reshape(1, -1)

# Messages
positive = "At 95% accuracy, the data suggests possible development of malignant disease. Please see a doctor."
negative = "At 95% accuracy, the data suggests no malignant development. However, consider further testing."

# Predict
if st.button("Predict"):
    predicted = model.predict(features)[0]
    proba = model.predict_proba(features)[0]

    result = positive if predicted == 1 else negative
    st.success(result)
    st.info(f"Probability Benign: {proba[0]:.2f}, Malignant: {proba[1]:.2f}")
