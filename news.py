import pandas as pd
import numpy as np
import re
import joblib
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Load the datasets
df1 = pd.read_csv("gossipcop_fake.csv")
df2 = pd.read_csv("gossipcop_real.csv")
df3 = pd.read_csv("politifact_fake.csv")
df4 = pd.read_csv("politifact_real.csv")

# Add label columns
df1['label'] = 'FAKE'
df2['label'] = 'REAL'
df3['label'] = 'FAKE'
df4['label'] = 'REAL'

# Combine all
df = pd.concat([df1, df2, df3, df4], ignore_index=True)
df = df[['title', 'label']].dropna()

# Shuffle
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Preprocessing function
def preprocess_text(text):
    text = re.sub(r'[^a-zA-Z]', ' ', str(text)).lower()
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    clean_tokens = [
        lemmatizer.lemmatize(word)
        for word in tokens
        if word not in stopwords.words('english') and len(word) > 2
    ]
    return " ".join(clean_tokens)

# Clean text
df['clean_text'] = df['title'].apply(preprocess_text)

# Vectorization
vectorizer = TfidfVectorizer(max_features=10000, stop_words='english', ngram_range=(1, 2))
X = vectorizer.fit_transform(df['clean_text'])
y = df['label'].apply(lambda x: 1 if x == 'REAL' else 0)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Train model
model = RandomForestClassifier(
    n_estimators=300,
    max_depth=50,
    class_weight='balanced',
    random_state=42
)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Save
joblib.dump(model, 'news_classifier_model.pkl')
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')
