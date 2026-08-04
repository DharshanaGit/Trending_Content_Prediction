# Problem Statement and Solution Design Document

**Project Name:** Trending Content Predictor  
**Author/Team:** Hackathon Submission Team  
**Date:** August 2026  

---

## 1. Introduction & Context
With over 500 hours of content uploaded to YouTube every minute, and corresponding growth across OTT platforms like Netflix, Prime Video, and regional VOD platforms, content creators and media houses face a high-stakes challenge. Deciding how to phrase titles, tag videos, categorize them, and schedule uploads represents a complex, multi-dimensional optimization problem. 

Currently, content success is largely treated as a guessing game or reactive process (analyzing performance *after* publication). This project introduces a predictive solution that allows creators and platforms to estimate the trending probability of a video **pre-publication**, using only the metadata available before the video goes live.

---

## 2. Problem Statement
The central challenge is to predict whether a newly created video will become **"Trending"** based strictly on its initial metadata, without introducing **data leakage** from post-publish metrics.

### Key Pain Points
1. **The Creator Guesswork Loop:** Creators optimize elements (like title length or tags) arbitrarily, leading to unpredictable reach.
2. **Platform Recommendation Cold-Start:** Video platforms rely heavily on engagement metrics (clicks, view duration, likes) to feed recommendation engines. For a newly uploaded video, these values are zero. The engine is blind during the critical first hours, leading to discoverability bottlenecks.
3. **Imbalanced Success Criteria:** A fraction of uploaded content drives the majority of platform traffic. Identifying this high-impact fraction requires robust, highly discriminative classification.

### Scope & Constraints
- **Target Variable (Trending):** Defined by analyzing the view counts in a real YouTube trending dataset. A video is classified as `Trending` (`is_trending = 1`) if its views reach or exceed the **75th percentile** threshold of the dataset. Otherwise, it is classified as `Not Trending` (`is_trending = 0`).
- **Feature Restrictions:** No post-publish variables (e.g., views, likes, dislikes, comment text, or comment counts) can be used during model inference, as these are unavailable at the moment of upload.

---

## 3. The Engineered Solution

Our solution addresses these pain points with a **3-tier modular system** that integrates high-accuracy machine learning, standard microservice backend processing, and an interactive frontend.

### A. The Feature Engineering Engine
To unlock high performance, we look past raw indicators (like title length) and analyze psychological clickbait markers and system configuration variables:
- **Title Shouting Ratio (`title_caps_ratio`):** We calculate the proportion of uppercase characters relative to total title length. Viral videos often leverage ALL CAPS phrases (e.g. "CAN'T BELIEVE THIS...").
- **Exclamation Flag (`has_exclamation`):** A binary indicator (`0` or `1`) showing if the title contains an exclamation mark, which serves as a proxy for urgency or clickbait.
- **Engagement Restrictions (`comments_disabled` & `ratings_disabled`):** Binary flags representing whether user feedback is blocked. Disabling these signals historically degrades platform indexing scores.
- **Timezone Scheduling (`upload_hour` & `upload_dayofweek`):** Extracting temporal features from publication timestamps to capture prime-time windows relative to target geographic audiences.

### B. Machine Learning Modeling (XGBoost)
We implemented a tuned **XGBoost (Extreme Gradient Boosting) Classifier** pipeline:
- **Categorical Processing:** Categorical features (`category_id` and `publish_country`) pass through a Scikit-Learn `OneHotEncoder`.
- **Numerical Scaling:** Numerical metadata features are standardized using `StandardScaler`.
- **Optimized Tree Structure:** Deployed a deeper ensemble of decision trees (`n_estimators=500`, `max_depth=10`, `learning_rate=0.03`, and subsampling rates of `0.9` to prevent overfitting).
- **Inference Pipeline:** Combined the preprocessors and tree ensemble into a single joblib-serialized pipeline (`trending_model.joblib`) for clean, single-call predictions.

### C. The Decoupled/Monolithic Architecture
The application offers dual deployment flexibility:
1. **API Microservice (FastAPI):** A fast REST service that loads the model pipeline on startup and listens for JSON payloads via a POST request at `/predict`. Validates input data structure natively using Pydantic.
2. **Direct Monolithic Client (Streamlit):** Bypasses HTTP request latency for cloud-native single-container platforms (e.g., Streamlit Community Cloud). The Streamlit app loads the joblib model directly in-memory, performing sub-second local inferences.

---

## 4. Evaluation and Validation Results

The model was trained and evaluated on a randomized train/test split (80% train, 20% validation) using real YouTube metadata.

### Model Accuracy Metrics
- **Overall Accuracy:** **91.0%**
- **ROC-AUC Score:** **0.9455** (Demonstrates a high probability of scoring a true trending video higher than a non-trending video).
- **F1-Score (Class 1 - Trending):** **0.80** (Precision `0.87`, Recall `0.73`), showing strong performance despite the inherent class imbalance of the trending target.

### Sample Prediction Matrix (Inference Results)
- **Normal Title:** *"A simple day in the life vlog"* -> Probability: ~14.2% (Not Trending)
- **Clickbait Title:** *"OMG! WE CANNOT BELIEVE THIS HAPPENED!!"* -> Probability: ~84.9% (Trending)

---

## 5. Technical Stack Breakdown
- **Language:** Python 3.9+
- **Numerical & ML Computations:** XGBoost, Scikit-Learn, Pandas, NumPy, Joblib
- **API Server & Routing:** FastAPI, Uvicorn, Pydantic
- **Client Interface:** Streamlit, CSS
- **Code Repository & CI/CD:** Git, GitHub, Streamlit Cloud Auto-reboot pipeline

---

## 6. Business Value & Future Vision
1. **Dynamic Optimization API:** Platforms can expose this API directly to creators in their creator studio. As they write their titles, they receive real-time structural guidance to increase score probability before submitting.
2. **Auto-retraining Pipelines:** A cron job can pull daily YouTube trending statistics to update the model, keeping feature weights aligned with shifting consumer preferences and platform algorithm updates.
3. **Multi-Modal Upgrades:** Future versions can process thumbnail image embeddings (using simple CNNs) and title semantic meanings (using lightweight LLMs/Transformers) to refine predictions.
