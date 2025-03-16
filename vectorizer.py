import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib  # For saving and loading models

# 1. Load and preprocess the data
df = pd.read_csv('AI_Human.csv')
df = df.dropna()

# 2. Create and fit the vectorizer
vectorizer = TfidfVectorizer(max_features=5000)
vectorizer.fit(df['text'])

# 3. Save the vectorizer
vectorizer_filename = 'model/tfidf_vectorizer.joblib'
joblib.dump(vectorizer, vectorizer_filename)

print(f"Vectorizer saved to {vectorizer_filename}")