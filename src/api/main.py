from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import uvicorn
import os

app = FastAPI(title="Trending Content Predictor API")

# Load model on startup
MODEL_PATH = "models/trending_model.joblib"
model = None

@app.on_event("startup")
def load_model():
    global model
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
        print(f"Model loaded from {MODEL_PATH}")
    else:
        print(f"Warning: Model not found at {MODEL_PATH}")

class VideoRequest(BaseModel):
    category_id: str
    publish_country: str
    upload_hour: int
    upload_dayofweek: int
    num_tags: int
    title_length: int
    comments_disabled: int
    ratings_disabled: int
    title_caps_ratio: float
    has_exclamation: int

@app.get("/")
def read_root():
    return {"message": "Welcome to the Trending Content Predictor API"}

@app.post("/predict")
def predict_trend(video: VideoRequest):
    if model is None:
        return {"error": "Model not loaded"}
    
    # Convert input to DataFrame (since Pipeline expects 2D array/df)
    data = {
        "category_id": [video.category_id],
        "publish_country": [video.publish_country],
        "upload_hour": [video.upload_hour],
        "upload_dayofweek": [video.upload_dayofweek],
        "num_tags": [video.num_tags],
        "title_length": [video.title_length],
        "comments_disabled": [video.comments_disabled],
        "ratings_disabled": [video.ratings_disabled],
        "title_caps_ratio": [video.title_caps_ratio],
        "has_exclamation": [video.has_exclamation]
    }
    df = pd.DataFrame(data)
    
    # Predict probability
    prob = model.predict_proba(df)[0][1]
    is_trending = model.predict(df)[0]
    
    return {
        "is_trending": int(is_trending),
        "trending_probability": float(prob)
    }

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
