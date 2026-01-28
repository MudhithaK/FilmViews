import streamlit as st
import numpy as np
import pandas as pd
import pickle
import joblib
import base64

# =========================
# Page config
# =========================
st.set_page_config(page_title="Views Predictor", layout="wide")

# =========================
# Helper: Load video as Base64
# =========================
def get_base64_video(video_path):
    with open(video_path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

# Make sure video.mp4 is in the SAME folder as this script
video_base64 = get_base64_video("video.mp4")

# =========================
# Background Video + Styling
# =========================
video_html = f"""
<style>
#bgvideo {{
    position: fixed;
    right: 0;
    bottom: 0;
    min-width: 100%;
    min-height: 100%;
    z-index: -1;
    object-fit: cover;
}}

.stApp {{
    background: transparent;
}}

h2, h3, label, span, div {{
    color: white !important;
    font-size: 20px !important;
}}
h1{{
    color: white !important;
    font-size: 40px !important;
}}
.subheader{{
color: white !important;
    font-size: 50px !important;

}}

[data-testid="stAppViewContainer"] {{
    background-color: rgba(0, 0, 0, 0.45);
}}
</style>

<video autoplay muted loop id="bgvideo">
    <source src="data:video/mp4;base64,{video_base64}" type="video/mp4">
</video>
"""

st.markdown(video_html, unsafe_allow_html=True)

# =========================
# Load data & model
# =========================
df = pickle.load(open('notebook/MudhithK/df.pkl', 'rb'))
pipe = joblib.load('notebook/MudhithK/predict.pkl')

# =========================
# App UI (Inputs Centered)
# =========================
#st.title("🎬 Views Predictor")

# Center inputs using 3 columns
left, center, right = st.columns([1, 2, 1])

with center:
    st.title("🎬 Views Predictor")

    FilmName = st.text_input("Enter Film Name")
    category = st.selectbox("Category", df['Category'].unique())
    language = st.selectbox("Language", df['Language'].unique())
    releaseY = st.number_input("Enter Release Year", min_value=1900, max_value=2030, step=1)
    type1 = st.selectbox("Release Month", list(range(1, 13)))
    viewY = st.number_input("Enter View Year", min_value=1900, max_value=2030, step=1)
    type2 = st.selectbox("Viewing Month", list(range(1, 13)))

    predict_btn = st.button("Predict Views")

# =========================
# Prediction Logic
# =========================
if predict_btn:
    query = pd.DataFrame({
        "Category": [category],
        "Language": [language],
        "Viewer_Rate": [0],
        "Release_Year": [releaseY],
        "Release_Month": [type1],
        "Viewing_Year": [viewY],
        "Viewing_Month_Num": [type2]
    })

    prediction = pipe.predict(query)[0]

    # Show prediction below the inputs
    left, center, right = st.columns([1, 0.5, 1])
    with center:
        st.markdown(
            "<h1 style='text-align:center; font-size:60px; color:white;'>📊 Predicted Number of Views</h1>",
            unsafe_allow_html=True)
        st.title(f"{int(prediction):,}")
