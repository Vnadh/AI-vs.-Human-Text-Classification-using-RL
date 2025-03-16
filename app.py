import streamlit as st
import numpy as np
import pandas as pd
import torch
from stable_baselines3 import PPO
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

vectorizer_filename='model/tfidf_vectorizer.joblib'
vectorizer = joblib.load(vectorizer_filename)

# Load the trained PPO model
model_path = "model/text_classifier_rl.zip"
model = PPO.load(model_path)

def classify_text(text):
    """Classifies input text as AI-generated or Human-written."""
    obs = vectorizer.transform([text]).toarray()
    predicted_action, _ = model.predict(obs)
    return "AI-generated" if predicted_action[0] == 1 else "Human-written"

# Streamlit UI
st.title("AI vs Human Text Classification using RL")
st.write("Enter a text to classify whether it is AI-generated or Human-written.")

user_input = st.text_area("Enter text:", "")

if st.button("Classify"):
    if user_input.strip():
        prediction = classify_text(user_input)
        st.success(f"Prediction: {prediction}")
    else:
        st.warning("Please enter some text.")
