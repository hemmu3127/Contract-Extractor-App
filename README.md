# Contract-Extractor-App

To make sure to run the app

1. Change to the current directory
2. create a virtual environment
3. pip install -r requirements.txt in that environment
4. after that run uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
5. then in browser check localhost:8000/docs
6. use the /extract keypoint to access the fastapi
