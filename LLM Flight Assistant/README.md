
# ✈️ Airline Review Sentiment Analysis

This project implements a sentiment analysis model using Logistic Regression to classify airline customer reviews as positive or negative. The model is trained using TF-IDF vectorization for feature extraction and applies text preprocessing using stopword removal and lemmatization.

## 🚀 Features

- Text preprocessing (lowercasing, punctuation removal, stopword removal, and lemmatization)
- TF-IDF vectorization for text representation
- Logistic Regression model for sentiment classification
- Model saving and loading using pickle
- Test cases for validation
- Error handling for missing data or file issues

## 🛠 Project Structure

```bash
Airline Review Sentiment Analysis/
├── main.py                  # Main script with model training and prediction functions
├── AirlineReviews.csv       # Dataset containing airline reviews
├── sentiment_model.pkl      # Saved model for future predictions
├── requirements.txt         # List of dependencies
└── README.md                # Project documentation
```

## 📦 Requirements

Ensure you have **Python 3.7+** installed and run the following command to install dependencies:

```bash
pip install -r requirements.txt
```

Make sure you also download the necessary NLTK resources:

```python
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
```

## 🧑‍💻 How to Run

### 1. Train the Model
Run the main script to train the sentiment analysis model:

```bash
python main.py
```

The script will:
- Load and preprocess the dataset
- Train a Logistic Regression classifier
- Save the model to `sentiment_model.pkl`
- Perform test case validation and display the results

### 2. Make Predictions
You can load the trained model and predict sentiment using the following code:

```python
from main import load_model, predict_sentiment

# Load the saved model
model_dict = load_model('sentiment_model.pkl')

# Example Review
review = "The flight was fantastic and the service was excellent!"
prediction = predict_sentiment(model_dict, review)
print(f"Sentiment: {prediction}")
```

**Output:**
```bash
Sentiment: positive
```

## 🧪 Test Cases

The model has been validated using the following test cases:

| **Review Text**                                           | **Expected Sentiment** |
|-----------------------------------------------------------|------------------------|
| The flight was on time, and the staff was friendly.       | Positive               |
| I had to wait 3 hours due to a delay. Terrible!           | Negative               |
| Great legroom and comfortable seats.                      | Positive               |
| Lost my luggage, extremely upset about this.              | Negative               |
| Check-in was smooth, no issues at all.                    | Positive               |

The accuracy of the model on these test cases will be displayed during execution.

## 🧑‍🔬 Model Evaluation

The model is evaluated using:
- **Accuracy Score**
- **Confusion Matrix**
- **Classification Report**

The results will be displayed at the end of model training.

## 🛡 Troubleshooting

- **FileNotFoundError:** Ensure `AirlineReviews.csv` is available in the directory.
- **NLTK Errors:** Run the following to download missing NLTK data:

```python
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
```

- **Pickle Error:** Ensure `sentiment_model.pkl` is generated after training.

## 📧 Contact

For any issues or questions, please feel free to contact me at **nickanto555@gmail.com**.

Happy Coding! 😊
