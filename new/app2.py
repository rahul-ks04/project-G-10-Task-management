import streamlit as st
import joblib
import re
import string
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize
from scipy.sparse import hstack
from datetime import date

# --- Load models and vectorizer ---
svm_model = joblib.load("models/svm_model.pkl")
rf_priority_model = joblib.load("models/rf_priority_model.pkl")
vectorizer = joblib.load("models/vectorizer.pkl")

# --- Load dataset for user profiles ---
df = pd.read_csv("synthetic_task_dataset_realistic.csv")

# --- Text Preprocessing ---
def preprocess_text(text):
    text = text.lower()
    text = re.sub(f"[{string.punctuation}]", " ", text)
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in stopwords.words("english")]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    return " ".join(tokens)

# --- Predict Task Category ---
def classify_task(description):
    processed = preprocess_text(description)
    vectorized = vectorizer.transform([processed])
    return svm_model.predict(vectorized)[0]

# --- Recommend Best User for a Category ---
def assign_user(task_category):
    preferred_users = df[df['user_preferred_categories'].str.contains(task_category, na=False)]

    if preferred_users.empty:
        assigned_user = df.sort_values(by='user_current_workload').iloc[0]['assigned_to']
    else:
        assigned_user = preferred_users.sort_values(by='user_current_workload').iloc[0]['assigned_to']
    
    return assigned_user

# --- Predict Task Priority ---
def predict_priority(description, effort, days_due, has_due, workload):
    processed = preprocess_text(description)
    desc_vector = vectorizer.transform([processed])
    numeric = np.array([[effort, days_due, has_due, workload]])
    final_input = hstack([desc_vector, numeric])
    return rf_priority_model.predict(final_input)[0]

# --- Streamlit App Layout ---
st.set_page_config(page_title="AI Task Management Assistant")
st.title("AI Task Management Assistant")
st.markdown("Automatically classify tasks, predict priority, and assign the best-fit user.")

with st.form("task_form"):
    task_desc = st.text_area("Enter Task Description", height=100)
    effort = st.slider("Estimated Effort (hours)", 1, 24, 4)

    has_due = st.checkbox("Task has a due date?", value=True)
    days_due = -1
    if has_due:
        due_date = st.date_input("Select Due Date", min_value=date.today())
        days_due = (due_date - date.today()).days

    submitted = st.form_submit_button("Predict Task Insights")

if submitted:
    if not task_desc.strip():
        st.warning("Please enter a valid task description.")
    else:
        category = classify_task(task_desc)
        user = assign_user(category)
        user_workload = df[df['assigned_to'] == user]['user_current_workload'].values[0]
        priority = predict_priority(task_desc, effort, days_due, int(has_due), user_workload)
        
        priority_label_map = {0: "Low", 1: "Medium", 2: "High", 3: "Critical"}
        predicted_priority_label = priority_label_map.get(priority, "Unknown")

        st.subheader("Predicted Results")
        st.markdown(f"**Task Category:** {category}")
        st.markdown(f"**Assigned To:** {user}")
        st.markdown(f"**Predicted Priority:** {predicted_priority_label}")
