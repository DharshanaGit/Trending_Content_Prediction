# src/ui/streamlit_app.py
import streamlit as st
import pandas as pd
import requests
from datetime import datetime
from pathlib import Path
import json
import time
import joblib

# ---- CONFIG -------------------------------------------------
MODEL_DIR = Path(__file__).parents[2] / "models"
MODEL_PATH = MODEL_DIR / "trending_model.joblib"
FEATURE_IMPORTANCE_IMG = MODEL_DIR / "feature_importance.png"

# ---- CACHED MODEL LOAD --------------------------------------
@st.cache_resource
def load_predictor_model():
    if MODEL_PATH.is_file():
        return joblib.load(MODEL_PATH)
    return None

model = load_predictor_model()
# -------------------------------------------------------------
st.set_page_config(
    page_title="Trending Content Predictor",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---- Custom CSS (optional but nice) -----------------------
st.markdown(
    """
    <style>
    /* Reduce the font size of all text in the app for a professional look */
    html, body, [class*="css"], .stMarkdown, p, span, label, input, select {
        font-size: 0.95rem !important;
    }
    h1 {
        font-size: 1.8rem !important;
    }
    h2, h3, [data-testid="stHeader"] {
        font-size: 1.25rem !important;
    }
    .stButton>button {
        width: 100%; 
        border-radius: 4px; 
        height: 2.8rem; 
        font-weight: 600;
        font-size: 0.95rem !important;
        background-color: #28a745;
        color: white;
        border: none;
    }
    .stButton>button:hover {
        background-color: #218838;
        color: white;
        border: none;
    }
    .stSlider>div[data-baseweb="slider"]>div {height: .5rem;}
    </style>
    """,
    unsafe_allow_html=True,
)
# -----------------------------------------------------------

st.title("Trending Content Predictor")
st.markdown(
    """
    **Will your next video go viral?**  
    Fill in the metadata below and get an instant probability score.  
    Experiment with upload time, category, or tag count to see how the odds change.
    """
)

with st.sidebar:
    st.subheader("Model Configuration")
    st.markdown(
        """
        * **Algorithm:** XGBoost Classifier
        * **Framework:** Scikit-Learn Pipeline
        * **Target:** Pre-publish Virality
        """
    )
    
    st.divider()
    with st.expander("Show Validation Metrics"):
        m_col1, m_col2 = st.columns(2)
        with m_col1:
            st.metric(label="Accuracy", value="91.0%")
            st.metric(label="F1-Score", value="80.0%")
        with m_col2:
            st.metric(label="ROC-AUC", value="94.5%")
            st.metric(label="Test Split", value="20%")
        
    st.divider()
    st.subheader("System Status")
    if model is not None:
        st.success("Model Loaded Successfully")
    else:
        st.error("Model Binary Missing")

col1, col2 = st.columns(2)


# ------------------------------------------------------------------
# INPUT FORM
# ------------------------------------------------------------------
col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    st.subheader("Video Metadata")
    # ---- Category (YouTube IDs) -------------------------------------------------
    categories = {
        "1": "Film & Animation",
        "2": "Autos & Vehicles",
        "10": "Music",
        "15": "Pets & Animals",
        "17": "Sports",
        "19": "Travel & Events",
        "20": "Gaming",
        "22": "People & Blogs",
        "23": "Comedy",
        "24": "Entertainment",
        "25": "News & Politics",
        "26": "Howto & Style",
        "27": "Education",
        "28": "Science & Technology",
        "29": "Nonprofits & Activism",
    }
    category_id = st.selectbox(
        "Category ID",
        options=list(categories.keys()),
        format_func=lambda k: f"{k} – {categories[k]}",
        index=list(categories.keys()).index("24"),
        help="YouTube category numeric ID (see https://developers.google.com/youtube/v3/docs/videoCategories/list)",
    )

    # ---- Publish Country -------------------------------------------------------
    countries_dict = {
        "US": "United States",
        "GB": "United Kingdom",
        "CA": "Canada",
        "IN": "India",
        "DE": "Germany",
        "FR": "France"
    }
    publish_country = st.selectbox(
        "Publish Country",
        options=list(countries_dict.keys()),
        format_func=lambda k: countries_dict[k],
        index=0
    )

    # ---- Language (optional) ---------------------------------------------------
    languages_dict = {
        "en": "English",
        "ta": "Tamil",
        "es": "Spanish",
        "fr": "French",
        "de": "German",
        "ja": "Japanese",
        "ko": "Korean",
        "pt": "Portuguese",
        "ru": "Russian",
        "hi": "Hindi",
        "ar": "Arabic",
        "zh": "Chinese"
    }
    language = st.selectbox(
        "Language (audio/subtitles)",
        options=list(languages_dict.keys()),
        format_func=lambda k: languages_dict[k],
        index=0,
        help="Primary language of the video – influences discoverability in local markets.",
    )

    # ---- Title length -----------------------------------------------------------
    title_length = st.slider(
        "Title Length (characters)",
        min_value=10,
        max_value=100,
        value=38,
        step=1,
        help="Shorter titles tend to perform better on mobile, but very short titles can be ambiguous.",
    )



    # ---- Tags (free‑form) -------------------------------------------------------
    tags_input = st.text_area(
        "Tags (comma‑separated)",
        value="funny,comedy,sketch",
        height=80,
        help="Enter up to 50 tags separated by commas. The app will count them internally.",
    )
    # Derive the count for the API (you can also send the raw string if you extend the backend)
    num_tags = max(0, len([t.strip() for t in tags_input.split(",") if t.strip()]))

with col_right:
    st.subheader("Upload Schedule")
    # ---- Day of week -----------------------------------------------------------
    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    upload_day = st.selectbox(
        "Upload Day",
        options=days,
        index=days.index("Friday"),
        help="Pick the day you plan to publish. Weekday evenings usually get more traffic.",
    )
    upload_day_idx = days.index(upload_day)  # Monday=0 … Sunday=6

    # ---- Hour of day (24‑h) ----------------------------------------------------
    upload_hour = st.slider(
        "Upload Hour (24 h)",
        min_value=0,
        max_value=23,
        value=18,
        step=1,
        help="Hour in UTC (adjust to your local timezone if needed). Prime time is 18‑22 UTC for many regions.",
    )

    # ---- Duration (minutes) ----------------------------------------------------
    duration_min = st.slider(
        "Duration (minutes)",
        min_value=0.5,
        max_value=30.0,
        value=5.0,
        step=0.5,
        help="Short‑form videos (< 2 min) often get higher completion rates; longer videos can earn more watch‑time.",
    )

    # ---- Metadata Flags ---------------------------------------------------------
    comments_disabled = st.checkbox(
        "Disable Comments",
        value=False,
        help="Trending videos usually have comments enabled. Disabling them lowers your score.",
    )
    ratings_disabled = st.checkbox(
        "Disable Ratings (Likes/Dislikes)",
        value=False,
        help="Disabling ratings is usually a negative signal for trending algorithms.",
    )


# ---- Prediction Trigger Button ----------------------------------------------
st.write("")
if st.button("Predict Probability", use_container_width=True):
    if model is not None:
        try:
            start = time.time()
            
            # Convert input to DataFrame for the model pipeline
            df = pd.DataFrame({
                "category_id": [str(category_id)],
                "publish_country": [str(publish_country)],
                "upload_hour": [int(upload_hour)],
                "upload_dayofweek": [int(upload_day_idx)],
                "num_tags": [int(num_tags)],
                "title_length": [int(title_length)],
                "comments_disabled": [int(comments_disabled)],
                "ratings_disabled": [int(ratings_disabled)],
                "is_top_channel": [0]
            })
            
            # Predict
            is_trending = int(model.predict(df)[0])
            prob = float(model.predict_proba(df)[0][1])
            elapsed = time.time() - start

            # Render prediction results in the main area
            st.divider()
            st.subheader("Prediction Result")
            r_col1, r_col2 = st.columns(2)
            with r_col1:
                st.metric(label="Trending Probability", value=f"{prob:.1%}")
                if is_trending:
                    st.success("High chance of trending.")
                else:
                    st.error("Low chance of trending.")
            with r_col2:
                st.write("")
                st.progress(min(max(prob, 0.0), 1.0))
                st.caption(f"Inference latency: {elapsed * 1000:.1f}ms (cached model)")

            # Show feedback in the main body area
            st.divider()
            st.subheader("Optimization Insights")
            
            o_col1, o_col2 = st.columns(2)
            with o_col1:
                if not is_trending:
                    st.info(
                        "💡 Try shifting publication to a weekday evening (18:00 - 22:00 UTC) "
                        "or raising the number of targeted tags to improve performance."
                    )
                else:
                    st.success("✨ Your video characteristics match historical trending parameters!")

        except Exception as e:
            st.error(f"Error during prediction: {str(e)}")
            st.exception(e)
    else:
        st.warning("Model file not found. Inference is currently unavailable.")

with st.expander("What drove this prediction? (Feature importance)"):
    # Prefer SHAP if you have it, otherwise fall back to the static bar chart
    shap_path = MODEL_DIR / "shap_summary.png"
    if shap_path.is_file():
        st.image(str(shap_path), use_column_width=True,
                 caption="SHAP summary – red pushes probability up, blue pushes it down.")
    elif FEATURE_IMPORTANCE_IMG.is_file():
        st.image(str(FEATURE_IMPORTANCE_IMG), use_column_width=True,
                 caption="Average feature importance from the XGBoost model.")
    else:
        st.warning("No explainability plot available – run `train.py` with SHAP or save the importance figure.")

# End of UI
