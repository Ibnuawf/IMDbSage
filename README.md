# Sentiment Analysis for Movie Reviews

This project provides both a command-line tool and a RESTful API for predicting the sentiment (positive or negative) of movie reviews using a machine learning model trained on IMDb data.

## Features

- Train a sentiment analysis model using TF-IDF and Logistic Regression
- Predict sentiment from the command line or via a FastAPI web API
- Easily extensible and well-documented codebase

## Installation

1. **Clone the repository or download the project files.**
2. **(Recommended) Create a virtual environment:**
   ```bash
   python -m venv myenv
   myenv\Scripts\activate  # On Windows
   # source myenv/bin/activate  # On macOS/Linux
   ```
3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Training the Model

To train the sentiment analysis model on the IMDb dataset, run:

```bash
python train.py
```

This will process the data and save the trained model to the `models/` directory as `sentiment_model.joblib`.

## Making Predictions (Command Line)

To predict the sentiment of a new review from the command line:

```bash
python predict.py "This movie was fantastic! The acting was superb and the plot was gripping."
```

Example output:

```
Review: "This movie was fantastic! The acting was superb and the plot was gripping."
Sentiment: positive
Confidence (Positive): 0.9876
```

## Running the API

To start the FastAPI server:

```bash
uvicorn main:app --reload
```

The API will be available at [http://127.0.0.1:8000](http://127.0.0.1:8000).

### API Usage

- **POST** `/predict/` with JSON body:

  ```json
  {
    "text": "This movie was great!"
  }
  ```

  Response example:

  ```json
  {
    "review": "This movie was great!",
    "sentiment": "positive",
    "confidence_positive": 0.95
  }
  ```

- Visit [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs) for interactive API documentation (Swagger UI).

## Project Structure

- `data/imdb_reviews.csv` — Raw movie review data
- `models/sentiment_model.joblib` — Trained machine learning model
- `train.py` — Script to train the model
- `predict.py` — Script to make predictions from the command line
- `main.py` — FastAPI application for the web API
- `requirements.txt` — Python dependencies
- `README.md` — Project documentation

## Requirements

- Python 3.8+
- See `requirements.txt` for all required packages
