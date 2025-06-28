import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import joblib
import os


# Load the IMDb movie review dataset
print("Loading and preparing data...")
df = pd.read_csv('data/imdb_reviews.csv')
X = df['review']
y = df['sentiment']


# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Define the machine learning pipeline
sentiment_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', LogisticRegression())
])


# Train the model
print("Training the model...")
sentiment_pipeline.fit(X_train, y_train)
print("Model training complete.")


# Evaluate the model
accuracy = sentiment_pipeline.score(X_test, y_test)
print(f"Model accuracy: {accuracy:.4f}")


# Save the trained model
if not os.path.exists('models'):
    os.makedirs('models')

print("Saving the model...")
joblib.dump(sentiment_pipeline, 'models/sentiment_model.joblib')
print("Model saved successfully in the 'models' directory.")