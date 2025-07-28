
import pandas as pd

df = pd.read_csv("Career QA Dataset.csv")

df.head()

df.info()

import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
import joblib

df['cleaned_question'] = df['question'].apply(lambda text: re.sub(f'[{re.escape(string.punctuation)}]', '', str(text).lower()))
print("Text preprocessing completed. A 'cleaned_question' column has been added.")
print("Sample of cleaned questions:")
print(df[['question', 'cleaned_question']].head())

print("\nStep 4: Starting TF-IDF vectorization...")
tfidf_vectorizer = TfidfVectorizer(max_features=5000)
X = tfidf_vectorizer.fit_transform(df['cleaned_question'])
# The target variable is 'role' (the career role).
y = df['role']

print("TF-IDF vectorization completed. Shape of features (X):", X.shape)
print("Number of unique career roles (y):", len(y.unique()))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print("\nStep 5: Data split into training and testing sets.")
print(f"Training set size: {X_train.shape[0]} samples")
print(f"Testing set size: {X_test.shape[0]} samples")

model = LogisticRegression(max_iter=1000)
print("\nStep 6: Starting model training (Logistic Regression)...")
model.fit(X_train, y_train)
print("Model training completed.")

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='weighted')

print("\nStep 7: Model Evaluation Results:")
print(f"Model Accuracy: {accuracy:.4f}")
print(f"Model F1-Score (weighted): {f1:.4f}")

model_filename = 'intent_model.pkl'
vectorizer_filename = 'vectorizer.pkl'

joblib.dump(model, model_filename)
joblib.dump(tfidf_vectorizer, vectorizer_filename)

print(f"\nStep 8: Model saved as '{model_filename}' and vectorizer saved as '{vectorizer_filename}'.")
print("\n'train_model.py' script finished successfully.")

"""# **SVM**"""

from sklearn.svm import SVC

model = SVC(kernel='linear', probability=True, random_state=42) # Changed: Initialize SVC model. [cite_start]Added probability=True for potential future use (e.g., confidence scores) [cite: 4]
print("\nStep 6: Starting model training (Support Vector Machine)...")
model.fit(X_train, y_train)
print("Model training completed.")

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='weighted') # Use weighted for multiclass problems

print("\nStep 7: Model Evaluation Results:")
print(f"Model Accuracy: {accuracy:.4f}")
print(f"Model F1-Score (weighted): {f1:.4f}")

model_filename = 'intent_model_svm.pkl'
vectorizer_filename = 'vectorizer_svm.pkl'

joblib.dump(model, model_filename)
joblib.dump(tfidf_vectorizer, vectorizer_filename)

print(f"\nStep 8: Model saved as '{model_filename}' and vectorizer saved as '{vectorizer_filename}'.")
print("\n'train_model.py' script finished successfully using SVM.")

"""# **Naive Bayes**"""

from sklearn.naive_bayes import MultinomialNB

model = MultinomialNB() # Changed: Initialize Multinomial Naive Bayes model
print("\nStep 6: Starting model training (Multinomial Naive Bayes)...")
model.fit(X_train, y_train)
print("Model training completed.")

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='weighted')

print("\nStep 7: Model Evaluation Results:")
print(f"Model Accuracy: {accuracy:.4f}")
print(f"Model F1-Score (weighted): {f1:.4f}")

model_filename = 'intent_model_naive_bayes.pkl'
vectorizer_filename = 'vectorizer_naive_bayes.pkl'

joblib.dump(model, model_filename)
joblib.dump(tfidf_vectorizer, vectorizer_filename)

print(f"\nStep 8: Model saved as '{model_filename}' and vectorizer saved as '{vectorizer_filename}'.")
print("\n'train_model.py' script finished successfully using Naive Bayes.")

import joblib
import re
import string

# --- Step 1: Define the text preprocessing function ---
# This function must be identical to the one used during model training.
def preprocess_text(text):
    text = str(text).lower()  # Convert to string and lowercase
    text = re.sub(f'[{re.escape(string.punctuation)}]', '', text)  # Remove punctuation
    return text

# --- Step 2: Load the saved model and vectorizer ---
# Adjust filenames if you saved them with different names (e.g., for Logistic Regression or SVM)
try:
    model = joblib.load('intent_model_naive_bayes.pkl')
    tfidf_vectorizer = joblib.load('vectorizer_naive_bayes.pkl')
    print("Model and vectorizer loaded successfully.")
except FileNotFoundError:
    print("Error: Model or vectorizer file not found. Make sure 'intent_model_naive_bayes.pkl' and 'vectorizer_naive_bayes.pkl' are in the same directory.")
    print("If you trained a different model (e.g., Logistic Regression or SVM), adjust the filenames in the code.")
    exit()
except Exception as e:
    print(f"An error occurred while loading the files: {e}")
    exit()

# --- Step 3: Test with a sample user input ---
sample_question = "What does a data scientist do?"
# You can change this to any question you want to test
# sample_question = "How to become a marketing manager?"
# sample_question = "What are the responsibilities of a financial analyst?"

print(f"\nSample Question: '{sample_question}'")

# --- Step 4: Preprocess the sample question ---
cleaned_question = preprocess_text(sample_question)
print(f"Cleaned Question: '{cleaned_question}'")

# --- Step 5: Transform the cleaned question using the loaded TF-IDF vectorizer ---
# The vectorizer expects an iterable (e.g., a list), so put the single question in a list.
question_vector = tfidf_vectorizer.transform([cleaned_question])
print(f"Question Vector Shape: {question_vector.shape}")

# --- Step 6: Make a prediction using the loaded model ---
predicted_role = model.predict(question_vector)[0]

print(f"\nPredicted Career Role: {predicted_role}")

# Optional: If your model supports probability estimates (like Logistic Regression or SVC with probability=True)
# You can get probability scores for each class
# if hasattr(model, 'predict_proba'):
#     probabilities = model.predict_proba(question_vector)
#     # Get class names from the model if available (might vary by sklearn version/model)
#     # For LogisticRegression and SVC, model.classes_ contains the labels
#     class_probabilities = dict(zip(model.classes_, probabilities[0]))
#     print("\nClass Probabilities:")
#     for role, prob in sorted(class_probabilities.items(), key=lambda item: item[1], reverse=True)[:5]:
#         print(f"- {role}: {prob:.4f}")

