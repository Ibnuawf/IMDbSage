import sys
import joblib


# Load the pre-trained sentiment analysis model
model = joblib.load('models/sentiment_model.joblib')


# Parse command-line arguments
if len(sys.argv) < 2:
    print("Usage: python predict.py <review_text>")
    sys.exit(1)


review_text = " ".join(sys.argv[1:])


# Perform sentiment prediction
prediction = model.predict([review_text])
probabilities = model.predict_proba([review_text])


# Output the result
print(f'Review: "{review_text}"')
print(f"Sentiment: {prediction[0]}")
print(f"Confidence (Positive): {probabilities[0][1]:.4f}")