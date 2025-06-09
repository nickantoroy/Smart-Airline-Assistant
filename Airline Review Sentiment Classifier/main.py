import pandas as pd
import numpy as np
import re
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from typing import List, Tuple, Any
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk

nltk.download('stopwords')
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text) 
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return ' '.join(words)

def train_sentiment_model(training_data: List[Tuple[str, str]]) -> Any:
  
    texts, labels = zip(*training_data)

    processed_texts = [preprocess_text(text) for text in texts]

    vectorizer = TfidfVectorizer(max_features=5000)
    X = vectorizer.fit_transform(processed_texts)

    model = LogisticRegression(max_iter=1000)
    model.fit(X, labels)

    return {'model': model, 'vectorizer': vectorizer}

def predict_sentiment(model_dict: Any, new_text: str) -> str:

    processed_text = preprocess_text(new_text)
    X_new = model_dict['vectorizer'].transform([processed_text])
    return model_dict['model'].predict(X_new)[0]

def save_model(model_dict: dict, filepath: str) -> None:
    with open(filepath, 'wb') as f:
        pickle.dump(model_dict, f)

def load_model(filepath: str) -> dict:
    with open(filepath, 'rb') as f:
        return pickle.load(f)

def evaluate_model(model_dict, X_test, y_test):
    predictions = model_dict['model'].predict(X_test)
    print("Accuracy:", accuracy_score(y_test, predictions))
    print("Confusion Matrix:\n", confusion_matrix(y_test, predictions))
    print("Classification Report:\n", classification_report(y_test, predictions))

if __name__ == "__main__":
    try:
        df = pd.read_csv('AirlineReviews.csv', encoding='latin1', on_bad_lines='skip')

        df.dropna(subset=['Review', 'OverallScore'], inplace=True)

        def clean_text(text):
            """Cleans and trims long reviews."""
            if pd.isna(text):
                return ""
            text = str(text)
            return text.strip() if len(text) <= 500 else text[:500]

        df['Review'] = df['Review'].apply(clean_text)
        def get_sentiment(score):
            try:
                score = float(score)
                return "positive" if score >= 5 else "negative"
            except:
                return None

        df['sentiment'] = df['OverallScore'].apply(get_sentiment)
        df.dropna(subset=['sentiment'], inplace=True)

        print(f"Total cleaned reviews: {len(df)}")

        training_data = list(zip(df['Review'].tolist(), df['sentiment'].tolist()))

        model_dict = train_sentiment_model(training_data)

        save_model(model_dict, 'sentiment_model.pkl')

        test_cases = [
            "The flight was on time, and the staff was friendly.",
            "I had to wait 3 hours due to a delay. Terrible!",
            "Great legroom and comfortable seats.",
            "Lost my luggage, extremely upset about this.",
            "Check-in was smooth, no issues at all."
        ]

        expected_sentiments = [
            "positive",
            "negative",
            "positive",
            "negative",
            "positive"
        ]

        print("\nRunning test cases:")
        print("-" * 50)

        correct_predictions = 0
        for test_text, expected in zip(test_cases, expected_sentiments):
            prediction = predict_sentiment(model_dict, test_text)
            correct_predictions += int(prediction == expected)

            print(f"\nTest case: {test_text}")
            print(f"Expected sentiment: {expected}")
            print(f"Predicted sentiment: {prediction}")
            print(f"Correct: {'✓' if prediction == expected else '✗'}")

        accuracy = (correct_predictions / len(test_cases)) * 100
        print(f"\nTest case accuracy: {accuracy:.1f}%")

        test_texts = [
            "The flight was comfortable and the staff was friendly",
            "Terrible service and delayed flight",
            "Average experience, nothing special"
        ]

        print("\nExample predictions:")
        for text in test_texts:
            print(f"\nText: {text}")
            print(f"Predicted sentiment: {predict_sentiment(model_dict, text)}")

    except FileNotFoundError:
        print("Error: AirlineReviews.csv file not found. Please ensure the file exists in the current directory.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")
