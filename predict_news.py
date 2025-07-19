import streamlit as st
from joblib import load
import re
import nltk
import os

# Set custom path for nltk_data
NLTK_DATA_DIR = "nltk_data"
nltk.data.path.append(NLTK_DATA_DIR)

# Download required nltk packages if not already present
nltk_resources = {
    "punkt": "tokenizers/punkt",
    "stopwords": "corpora/stopwords",
    "wordnet": "corpora/wordnet"
}

for resource, path in nltk_resources.items():
    try:
        nltk.data.find(path)
    except LookupError:
        nltk.download(resource, download_dir=NLTK_DATA_DIR)

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Load model and vectorizer
model = load('news_classifier_model.pkl')
vectorizer = load('tfidf_vectorizer.pkl')

# Preprocessing function
def preprocess_text(text):
    text = re.sub(r'[^a-zA-Z]', ' ', text).lower()
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    clean_tokens = [
        lemmatizer.lemmatize(word)
        for word in tokens
        if word not in stopwords.words('english') and len(word) > 2
    ]
    return " ".join(clean_tokens)

# Page setup
st.set_page_config(page_title="Fake News Detector", layout="wide")

# Custom CSS styling
st.markdown(
    """
    <style>
    .main {
        background-color: rgba(0, 0, 0, 0.6);
        padding: 2rem;
        border-radius: 12px;
        max-width: 800px;
        margin: 3rem auto;
        color: white;
        box-shadow: 0px 0px 10px rgba(255, 255, 255, 0.2);
    }

    .center-title {
        text-align: center;
        font-size: 2.8rem;
        font-weight: bold;
        margin-bottom: 1rem;
        color: white;
    }

    .stTextInput>div>div>input, .stTextArea>div>textarea {
        color: black;
        background-color: white;
    }

    .stButton>button {
        color: white;
        background-color: #4CAF50;
        border: none;
        padding: 10px 20px;
        border-radius: 8px;
        font-weight: bold;
        font-size: 1rem;
    }

    .stButton>button:hover {
        background-color: #45a049;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# App layout
st.markdown('<div class="main">', unsafe_allow_html=True)

st.markdown('<div class="center-title">📰 Fake News Detector</div>', unsafe_allow_html=True)
st.write("Enter a news article below and the model will predict whether it's **REAL** or **FAKE**.")

user_input = st.text_area("Paste your news article here:")

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter a news article.")
    else:
        clean_article = preprocess_text(user_input)
        vectorized_article = vectorizer.transform([clean_article])
        prediction = model.predict(vectorized_article)[0]
        label = "REAL" if prediction == 1 else "FAKE"
        st.success(f"Prediction: {label}")

st.markdown('</div>', unsafe_allow_html=True)
