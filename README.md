# Trending Content Prediction 

## Problem Statement

Predict whether a newly uploaded movie or short film will become **Trending** (1M+ views within the first 7 days) using metadata such as genre, duration, upload time, language, tags, and other available features.

## Objective

Build a Machine Learning classification model that helps content creators and OTT platforms estimate the popularity of content before publishing.

## Overview

The application follows a simple 3-tier architecture:

**Streamlit UI → FastAPI Backend → Machine Learning Model → Prediction**

Users enter video details, and the model predicts whether the content is likely to trend along with a confidence score.

## Features

* Trending / Not Trending prediction
* Prediction probability
* Interactive Streamlit interface
* FastAPI REST API
* Data preprocessing and feature engineering
* Feature importance visualization
* Swagger API documentation

## Technology Stack

* Python
* Pandas
* NumPy
* Scikit-learn
* XGBoost
* FastAPI
* Streamlit
* Joblib
* Matplotlib
* Seaborn

## Project Structure

```text
Trending_Content_Prediction/
│
├── data/
├── models/
├── src/
│   ├── api/
│   ├── model/
│   └── ui/
├── requirements.txt
└── README.md
```

## Installation

```bash
git clone https://github.com/your-username/Trending_Content_Prediction.git

cd Trending_Content_Prediction

python -m venv venv

# Windows
venv\Scripts\activate

# Linux/macOS
source venv/bin/activate

pip install -r requirements.txt
```

## Dataset

Place the YouTube Trending CSV files inside:

```text
data/raw/
```

## Machine Learning Workflow

1. Load dataset
2. Preprocess data
3. Engineer features
4. Train the model
5. Evaluate performance
6. Predict trending content

## Algorithm

**XGBoost Classifier**

Chosen because it performs well on structured data, handles mixed feature types, and provides high classification accuracy.

## Evaluation Metrics

* Accuracy
* Precision
* Recall
* F1 Score
* ROC-AUC
* Confusion Matrix

## Train the Model

```bash
python src/model/train.py
```

## Run Backend

```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

Swagger Docs:

```text
http://localhost:8000/docs
```

## Run Frontend

```bash
streamlit run src/ui/app.py
```

Open:

```text
http://localhost:8501
```

## Future Enhancements

* Real-time prediction
* Cloud deployment
* User authentication
* Explainable AI
* Automated model retraining

