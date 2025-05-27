import pandas as pd
import re  # <-- ADD THIS IMPORT
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Load datasets with explicit dtype to handle mixed types
true_df = pd.read_csv('True.csv', dtype=str, low_memory=False)
fake_df = pd.read_csv('Fake.csv', dtype=str, low_memory=False)

# Label and combine
true_df['label'] = 0  # 0 for true news
fake_df['label'] = 1  # 1 for fake news
df = pd.concat([true_df, fake_df])

# Clean text function
def clean_text(text):
    if not isinstance(text, str):  # Handle non-string values
        return ""
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)  # Keep only words and spaces
    return ' '.join(text.split())  # Remove extra spaces

# Apply cleaning
df['clean_text'] = df['text'].apply(clean_text)

# Remove empty texts
df = df[df['clean_text'].str.strip().astype(bool)]

# Vectorize
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df['clean_text'])
y = df['label']

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X, y)

# Save files
with open('vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)
    
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Model and vectorizer saved successfully!")