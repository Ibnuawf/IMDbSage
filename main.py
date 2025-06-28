from fastapi import FastAPI
from pydantic import BaseModel
import joblib

app = FastAPI(
    title="Sentiment Analysis API",
    description="A simple API to predict the sentiment of a movie review.",
    version="1.0.0"
)



# Load the pre-trained sentiment analysis model at application startup
model = joblib.load('models/sentiment_model.joblib')


# Pydantic model for incoming review data
class Review(BaseModel):
    text: str


# Prediction endpoint
@app.post('/predict/')
def predict_sentiment(review: Review):
    prediction = model.predict([review.text])
    probabilities = model.predict_proba([review.text])
    return {
        "review": review.text,
        "sentiment": prediction[0],
        "confidence_positive": probabilities[0][1]
    }


# Root endpoint for API health check
@app.get('/')
def read_root():
    return {"message": "Welcome to the Sentiment Analysis API. Go to /docs for more info."}