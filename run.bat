@echo off
cd /d "%~dp0"

echo Starting FastAPI Backend...
start "FastAPI" cmd /k "call .\venv\Scripts\activate.bat && python -m uvicorn src.api.main:app --reload"

echo Starting Streamlit Frontend...
start "Streamlit" cmd /k "call .\venv\Scripts\activate.bat && python -m streamlit run src/ui/app.py"

echo Both services are starting up! 
echo FastAPI Swagger UI: http://127.0.0.1:8000/docs
echo Streamlit App: http://localhost:8501
