import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score, f1_score
import joblib
import os

def load_data(filepath="youtube.csv", sample_size=50000):
    print(f"Loading {filepath}...")
    # Load and optionally sample to keep training fast for the hackathon MVP
    df = pd.read_csv(filepath)
    if len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=42)
    return df

def feature_engineering(df):
    """Create target variable and extract features from youtube.csv."""
    df = df.copy()
    
    # 1. Target Variable Creation (Labeling)
    # Define "Trending" as the top 25% of views in this dataset
    threshold = df['views'].quantile(0.75)
    df['is_trending'] = (df['views'] >= threshold).astype(int)
    
    # 2. Feature Extraction
    # time_frame e.g. "17:00 to 17:59" -> extract 17
    # Use regex to extract the first number before a colon
    df['upload_hour'] = df['time_frame'].astype(str).str.extract(r'^(\d{1,2}):')[0].fillna(12).astype(int)
    
    # Map published_day_of_week to numeric 0-6
    day_map = {
        'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3,
        'Friday': 4, 'Saturday': 5, 'Sunday': 6
    }
    df['upload_dayofweek'] = df['published_day_of_week'].map(day_map).fillna(0).astype(int)
    
    # Calculate number of tags
    # Handle NaN by filling with empty string, then count '|' separator
    df['num_tags'] = df['tags'].fillna('').apply(lambda x: len(x.split('|')) if x else 0)
    
    # Calculate title length
    df['title_length'] = df['title'].fillna('').str.len()
    
    # Calculate advanced text features
    df['title_caps_ratio'] = df['title'].fillna('').apply(lambda x: sum(1 for c in x if c.isupper()) / (len(x) + 1))
    df['has_exclamation'] = df['title'].fillna('').str.contains('!').astype(int)
    
    # Metadata flags
    df['comments_disabled'] = df['comments_disabled'].astype(int)
    df['ratings_disabled'] = df['ratings_disabled'].astype(int)
    
    # Ensure category_id is treated as categorical (string)
    df['category_id'] = df['category_id'].astype(str)
    
    # Handle missing values in publish_country
    df['publish_country'] = df['publish_country'].fillna('Unknown').astype(str)
    
    return df

def train_model():
    df = load_data()
    
    print("Applying feature engineering...")
    df = feature_engineering(df)
    
    # Define features (X) and target (y)
    features = ['category_id', 'publish_country', 'upload_hour', 'upload_dayofweek', 
                'num_tags', 'title_length', 'comments_disabled', 'ratings_disabled', 
                'title_caps_ratio', 'has_exclamation']
    X = df[features]
    y = df['is_trending']
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Preprocessing Pipeline
    categorical_cols = ['category_id', 'publish_country']
    numeric_cols = ['upload_hour', 'upload_dayofweek', 'num_tags', 'title_length',
                    'comments_disabled', 'ratings_disabled', 'title_caps_ratio', 'has_exclamation']
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
        ])
    
    # Model Pipeline
    model = XGBClassifier(
        n_estimators=500, 
        learning_rate=0.03, 
        max_depth=10,
        min_child_weight=1,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
        eval_metric='logloss'
    )
    
    pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                               ('classifier', model)])
    
    print("Training XGBoost model on real youtube data...")
    pipeline.fit(X_train, y_train)
    
    print("Evaluating model...")
    y_pred = pipeline.predict(X_test)
    y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
    
    print("\n--- Classification Report ---")
    print(classification_report(y_test, y_pred))
    print(f"ROC-AUC Score: {roc_auc_score(y_test, y_pred_proba):.4f}")
    print(f"F1-Score: {f1_score(y_test, y_pred):.4f}")
    
    # Save the model
    os.makedirs('models', exist_ok=True)
    model_path = 'models/trending_model.joblib'
    joblib.dump(pipeline, model_path)
    print(f"\nModel saved to {model_path}")

if __name__ == '__main__':
    train_model()
