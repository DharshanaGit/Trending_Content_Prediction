# PLAN: Trending Content Prediction MVP

## 1. Project Goal
Build a working Machine Learning MVP that predicts whether a video will trend (or its probability of trending) based on metadata features. The focus is on rapid execution, demonstrating business value for creators (SEO/optimization) and platforms (recommendation cold-start), and providing an interactive UI.

## 2. Tech Stack
- **Language**: Python 3.9+
- **Data/ML**: Pandas, NumPy, Scikit-Learn, XGBoost/LightGBM, Joblib
- **Backend API**: FastAPI (with Uvicorn)
- **Frontend UI**: Streamlit
- **Deployment (Optional)**: Hugging Face Spaces / Streamlit Community Cloud (Frontend) & Render / Railway (Backend)

## 3. Implementation Phases

### Phase 1: Data Acquisition & Preprocessing
- **Agent**: `data-scientist` / `ml-engineer`
- **Tasks**:
  - [ ] Source dataset (Kaggle: YouTube Trending or TMDB).
  - [ ] Engineer target variable (`Trending`: 1 if views > 1M in 7 days, else 0).
  - [ ] Feature Engineering: Extract `Hour`, `Day of Week` from timestamps.
  - [ ] Feature Engineering: Count tags/keywords.
  - [ ] Categorical Encoding: One-Hot Encode `Genres`, `Language`, etc.

### Phase 2: Model Development
- **Agent**: `ml-engineer`
- **Tasks**:
  - [ ] Train baseline model (Logistic Regression or Random Forest).
  - [ ] Train champion model (XGBoost or LightGBM).
  - [ ] Evaluate model using F1-Score / ROC-AUC.
  - [ ] Extract feature importance to display in the UI.
  - [ ] Serialize the best model using `joblib` (`.pkl`).

### Phase 3: Backend API (FastAPI)
- **Agent**: `backend-specialist`
- **Tasks**:
  - [ ] Set up FastAPI app structure (`src/api`).
  - [ ] Create Pydantic schemas for input request data (video metadata).
  - [ ] Load the serialized model into the app on startup.
  - [ ] Create a `/predict` POST endpoint to return predictions.

### Phase 4: Frontend UI (Streamlit)
- **Agent**: `frontend-specialist` (or Fullstack)
- **Tasks**:
  - [ ] Initialize Streamlit app (`src/ui`).
  - [ ] Build input forms (sliders for duration, date/time pickers, dropdowns for genres).
  - [ ] Connect Streamlit to the FastAPI backend (or bundle locally if API proves problematic).
  - [ ] Display prediction results visually.
  - [ ] Add a chart showing "Feature Importance" (What drives trending?).

## 4. Open Questions & Socratic Gate
1. **Dataset**: Have you already downloaded a specific dataset for this, or should we write a script to download one from Kaggle via API?
2. **Architecture**: Do you want the Streamlit frontend and FastAPI backend decoupled as two separate services, or would you prefer a simplified monolithic approach (Streamlit directly importing and running the model without a REST API) to save time for the hackathon?
3. **Deployment**: Are we aiming to deploy this live during the hackathon, or is a local demo sufficient?

## 5. Verification Checklist
- [ ] Data preprocessing pipeline runs without errors.
- [ ] Model achieves baseline F1-Score/ROC-AUC better than random chance.
- [ ] Model successfully serialized and loaded.
- [ ] FastAPI endpoint returns 200 OK with expected JSON format.
- [ ] Streamlit UI successfully accepts inputs and displays a prediction.
