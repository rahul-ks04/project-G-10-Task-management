#app.py - Main

from flask import Flask, render_template, request, jsonify
import joblib  # Import joblib for loading models
import pandas as pd
import numpy as np
import re
import string
from datetime import datetime

# --- NLTK setup ---
# These imports and downloads are crucial for the preprocessing function.
# Ensure 'stopwords' and 'punkt' (for word_tokenize) are downloaded once
# by running `nltk.download('stopwords')` and `nltk.download('punkt')`
# in your environment (e.g., a separate Python script or temporary Jupyter cell).
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize  # Ensure word_tokenize is imported

# Initialize NLTK components for preprocessing
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()


# --- Preprocessing function (from Week 1) ---
def preprocess_text(text):
    """
    Applies NLP preprocessing steps to a given text.
    Includes lowercasing, punctuation/number removal, tokenization,
    stopword removal, and stemming.
    """
    if not isinstance(text, str):  # Handle non-string inputs gracefully
        return ""
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(text)
    cleaned = [stemmer.stem(word) for word in tokens if word not in stop_words]
    return " ".join(cleaned)


# --- Load the saved model and vectorizer ---
# IMPORTANT:
# 1. Ensure your .pkl files are in the 'src/models/' directory relative to app.py.
# 2. Update the filename 'final_priority_prediction_model_random_forest.pkl'
#    to match the actual best model saved from your Week 4 notebook
#    (e.g., final_priority_prediction_model_xgboost.pkl, _svm.pkl, _naive_bayes.pkl).
try:
    final_model = joblib.load('models/final_priority_prediction_model_xgboost.pkl')  # <--- ADJUST THIS FILENAME!
    vectorizer = joblib.load('models/tfidf_vectorizer.pkl')
    print("Model and vectorizer loaded successfully.")
except FileNotFoundError:
    print("Error: Model or vectorizer .pkl files not found.")
    print("Please ensure they are in the 'project-G-10-Task-management/models/' directory relative to app.py")
    print("and that the model filename in app.py matches your saved best model.")
    exit()  # Exit the application if models aren't found

# --- Priority Mapping (from Week 2) ---
priority_mapping = {'Low': 0, 'Medium': 1, 'High': 2, 'Critical': 3}
# Reverse mapping for displaying results to the user
reverse_priority_mapping = {v: k for k, v in priority_mapping.items()}

# --- Workload Balancing Logic (from Week 3 - simplified for in-memory demo) ---
# This dictionary simulates user workloads. It will reset every time the Flask app restarts.
# For a production application, this data would be stored in a persistent database.
users = {
    'user_A': {'workload': 5, 'skills': ['development', 'design']},
    'user_B': {'workload': 2, 'skills': ['development']},
    'user_C': {'workload': 8, 'skills': ['design', 'testing']},
    'user_D': {'workload': 3, 'skills': ['testing']}
}


def assign_task_heuristically(predicted_priority_label):
    """
    Heuristically assigns a task to the user with the lowest current workload.
    Increments the assigned user's workload.
    """
    if not users:
        return None, "No users available for assignment."

    min_workload = float('inf')
    assigned_user = None

    for user_id, user_info in users.items():
        if user_info['workload'] < min_workload:
            min_workload = user_info['workload']
            assigned_user = user_id

    if assigned_user:
        users[assigned_user]['workload'] += 1  # Simulate task assignment
    return assigned_user


# --- Flask App Setup ---
app = Flask(__name__)


@app.route('/')
def index():
    """
    Renders the main HTML page and passes current user workloads to the template.
    """
    current_workloads = {user: info['workload'] for user, info in users.items()}
    return render_template('index.html', current_workloads=current_workloads)


@app.route('/predict', methods=['POST'])
def predict():
    """
    Handles task prediction and assignment requests from the web form.
    Processes input, makes prediction, applies workload balancing, and returns results as JSON.
    """
    if request.method == 'POST':
        task_description = request.form.get('task_description', '')
        due_date_str = request.form.get('due_date', '')

        # --- Preprocess and Feature Engineer for new input ---
        clean_description = preprocess_text(task_description)

        # TF-IDF vectorization for the new task description
        # vectorizer.transform expects an iterable (list of strings)
        task_tfidf = vectorizer.transform([clean_description])

        # Calculate days_until_due and has_due_date for the new task
        has_due_date = 0
        days_until_due_non_negative = 0
        if due_date_str:
            try:
                due_date = datetime.strptime(due_date_str, '%Y-%m-%d')
                creation_date = datetime.now()  # Assume creation date is current time
                days_until_due = (due_date - creation_date).days
                days_until_due_non_negative = max(0, days_until_due)  # Ensure non-negative
                has_due_date = 1
            except ValueError:
                print(f"Warning: Invalid due date format '{due_date_str}'. Using default values (no due date).")
                # days_until_due_non_negative and has_due_date remain 0

        # --- Prepare the final input feature vector for the model ---
        # This is the most crucial part: the input features MUST be in the same order
        # and format (TF-IDF features first, then numerical features) as X_train
        # that your model was trained on.

        # 1. Get TF-IDF features as a dense array
        task_tfidf_dense = task_tfidf.toarray()

        # 2. Create a list of all feature values in the correct order
        #    This assumes the numerical features were appended AFTER TF-IDF features during training.
        input_features_list = task_tfidf_dense[0].tolist()  # Start with the TF-IDF features
        input_features_list.append(days_until_due_non_negative)  # Add the first numerical feature
        input_features_list.append(has_due_date)  # Add the second numerical feature

        # 3. Convert to a 2D numpy array (1 sample, N features) for prediction
        final_input_features = np.array([input_features_list])

        # --- Make Prediction ---
        predicted_priority_encoded = final_model.predict(final_input_features)[0]
        predicted_priority_label = reverse_priority_mapping.get(predicted_priority_encoded, "Unknown")

        # --- Apply Workload Balancing ---
        assigned_user = assign_task_heuristically(predicted_priority_label)

        # Prepare response for the frontend
        response = {
            'task_description': task_description,
            'predicted_priority': predicted_priority_label,
            'assigned_user': assigned_user,
            'current_workloads': {user: info['workload'] for user, info in users.items()}  # Send updated workloads
        }
        return jsonify(response)


# --- Run the Flask app ---
if __name__ == '__main__':
    # Set host to '0.0.0.0' to make it accessible from other devices on the network
    # (useful for testing on mobile, but be cautious in production)
    app.run(debug=True, host='127.0.0.1', port=5000)
