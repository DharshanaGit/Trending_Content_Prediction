# src/ui/streamlit_app.py
import streamlit as st
import pandas as pd
import requests
from datetime import datetime
from pathlib import Path
import json
import time

# ---- CONFIG -------------------------------------------------
API_URL = "http://127.0.0.1:8000/predict"          # change if you deploy elsewhere
HEALTH_URL = "http://127.0.0.1:8000/docs"         # FastAPI docs endpoint – cheap health check
MODEL_DIR = Path(__file__).parents[2] / "models"
FEATURE_IMPORTANCE_IMG = MODEL_DIR / "feature_importance.png"
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
    .stButton>button {
        width: 100%; 
        border-radius: 4px; 
        height: 3rem; 
        font-weight: 600;
        font-size: 1.05rem;
        background-color: #2e66ff;
        color: white;
        border: none;
    }
    .stButton>button:hover {
        background-color: #1a4fdb;
        color: white;
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
    st.header("Model Information")
    st.caption("Trained on YouTube‑Trending metadata (XGBoost).")
    # Show a tiny version of the feature‑importance plot if it exists
    if FEATURE_IMPORTANCE_IMG.is_file():
        st.image(str(FEATURE_IMPORTANCE_IMG), use_column_width=True)
    else:
        st.info("Feature‑importance plot not found – run `train.py` first.")

    st.divider()
    st.subheader("API Status")
    try:
        health_resp = requests.get(HEALTH_URL, timeout=2)
        if health_resp.status_code == 200:
            st.success("API reachable")
        else:
            st.warning("API returned non‑200")
    except requests.exceptions.RequestException:
        st.error("Cannot reach API – check that the FastAPI server is running.")

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
    countries = ["US", "GB", "CA", "IN", "DE", "FR"]
    publish_country = st.selectbox("Publish Country", countries)

    # ---- Language (optional) ---------------------------------------------------
    languages = ["en", "es", "fr", "de", "ja", "ko", "pt", "ru", "hi", "ar", "zh"]
    language = st.selectbox(
        "Language (audio/subtitles)",
        options=languages,
        index=0,
        help="Primary language of the video – influences discoverability in local markets.",
    )

    # ---- Video Title -----------------------------------------------------------
    title = st.text_input(
        "Video Title",
        value="OMG! I CANNOT BELIEVE THIS HAPPENED!!",
        help="Try an ALL CAPS clickbait title with exclamation marks to see how it affects the score!",
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


if st.button("Predict", use_container_width=True):
    # ---- Compute text features on the fly --------------------------------------
    t_len = len(title)
    t_caps_ratio = sum(1 for c in title if c.isupper()) / (t_len + 1)
    has_exclamation = int("!" in title)
    
    # ---- Build payload that matches the API ------------------------------------
    payload = {
        "category_id": category_id,
        "publish_country": publish_country,
        "upload_hour": upload_hour,
        "upload_dayofweek": upload_day_idx,
        "num_tags": num_tags,
        "title_length": t_len,
        "comments_disabled": int(comments_disabled),
        "ratings_disabled": int(ratings_disabled),
        "title_caps_ratio": t_caps_ratio,
        "has_exclamation": has_exclamation,
    }

    # ---- Show a spinner while we wait -----------------------------------------
    with st.spinner("Querying the model..."):
        try:
            start = time.time()
            resp = requests.post(API_URL, json=payload, timeout=8)
            elapsed = time.time() - start

            if resp.status_code == 200:
                result = resp.json()
                # Expected keys from your FastAPI: "is_trending", "trending_probability"
                is_trending = result.get("is_trending", 0)
                prob = float(result.get("trending_probability", 0.0))

                st.divider()
                # ---- Simple metric ------------------------------------------------
                st.metric(label="Trending Probability", value=f"{prob:.1%}")
                if is_trending:
                    st.success("High chance of trending.")
                else:
                    st.error("Low chance of trending.")

                # ---- Probability gauge (Streamlit's progress bar works fine) -------
                st.progress(min(max(prob, 0.0), 1.0), text=f"{prob:.1%}")

                # ---- Feedback tip ---------------------------------------------------
                if not is_trending:
                    st.info(
                        "Try moving the upload to a weekday evening (18‑22 UTC) "
                        "or increasing the number of relevant tags – historically these boost odds."
                    )
                else:
                    st.success("Your video looks primed for a viral push.")

                # ---- Optional: Show latency (nice for judges) -----------------------
                st.caption(f"API latency: {elapsed:.2f}s")

            else:
                # API returned an error (e.g., 400/500)
                st.error(f"API error {resp.status_code}: {resp.text}")

        except requests.exceptions.ConnectionError:
            st.error("Could not reach the API. Is the FastAPI server running?")
        except requests.exceptions.Timeout:
            st.error("Request timed out – the backend might be overloaded.")
        except Exception as e:
            st.exception(e)   # shows full traceback in dev; remove in prod if you prefer

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
