import streamlit as st
import joblib
import numpy as np
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag

# Make sure you have downloaded the required NLTK data files:
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('averaged_perceptron_tagger')
# nltk.download('wordnet')

# Initialize stopwords and lemmatizer
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    """
    Basic cleaning: lowercases text, removes URLs, mentions, punctuation, extra whitespace, and numbers.
    """
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+|\b\w+\.com\b', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r"[^\w\s']", '', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\d+', '', text)
    return text

def remove_stopwords(tokens):
    """
    Removes common English stopwords from a list of tokens.
    """
    return [token for token in tokens if token not in stop_words]

def lemmatize_tokens_pos(tokens):
    """
    Applies POS-based lemmatization to a list of tokens.
    """
    lemmatized_tokens = []
    pos_tags = pos_tag(tokens)
    for token, tag in pos_tags:
        # Use verb lemmatization for tokens starting with 'VB'
        if tag.startswith('VB'):
            lemma = lemmatizer.lemmatize(token, pos='v')
        else:
            lemma = lemmatizer.lemmatize(token)
        lemmatized_tokens.append(lemma)
    return lemmatized_tokens

def join_tokens(tokens):
    """
    Joins a list of tokens into a single string.
    """
    return " ".join(tokens)

def preprocess_text(text):
    """
    Complete preprocessing pipeline that applies:
      1. Basic cleaning (lowercasing, URL/mention/punctuation/number removal)
      2. Tokenization
      3. Stopword removal
      4. POS-based lemmatization
      5. Joining tokens back into a cleaned string.
    
    Returns:
        A fully preprocessed string.
    """
    # Step 1: Clean the text
    cleaned = clean_text(text)
    # Step 2: Tokenize the cleaned text
    tokens = word_tokenize(cleaned)
    # Step 3: Remove stopwords
    tokens_no_stop = remove_stopwords(tokens)
    # Step 4: Apply POS-based lemmatization
    tokens_lemmatized = lemmatize_tokens_pos(tokens_no_stop)
    # Step 5: Join tokens into a string
    final_text = join_tokens(tokens_lemmatized)
    return final_text


# --- 1. Load the trained model and vectorizers (once at the start) ---
@st.cache_resource # Use st.cache_resource for loading models/vectorizers
def load_model_components():
    """Loads the trained Random Forest model and TF-IDF vectorizers."""
    try:
        loaded_rf_model = joblib.load('fake_news_rf_model.joblib')
        loaded_vectorizer_text = joblib.load('tfidf_vectorizer_text.joblib')
        loaded_vectorizer_title = joblib.load('tfidf_vectorizer_title.joblib')
        return loaded_rf_model, loaded_vectorizer_text, loaded_vectorizer_title
    except FileNotFoundError:
        st.error("Model or vectorizer files not found. Please ensure they are in the same directory as the script.")
        return None, None, None

rf_model, vectorizer_text, vectorizer_title = load_model_components()

# --- 2. Prediction Function (same as before) ---
def predict_fake_news(title_content, text_content, model, text_vectorizer, title_vectorizer):
    """
    Predicts if a news article is fake or real using the loaded Random Forest model.

    Args:
        text_content (str): The main text content of the news article.
        title_content (str): The title of the news article.
        model: Loaded Random Forest model.
        text_vectorizer: Loaded TF-IDF vectorizer for text.
        title_vectorizer: Loaded TF-IDF vectorizer for title.

    Returns:
        str: "Fake News" or "Real News" prediction.
    """
    
    try:
        processed_title = preprocess_text(title_content)
        processed_text = preprocess_text(text_content)
        text_features = text_vectorizer.transform([processed_text])
        title_features = title_vectorizer.transform([processed_title])
        combined_features = np.hstack([title_features.toarray(), text_features.toarray()])
        prediction_rf = model.predict(combined_features)
        if prediction_rf[0] == 1:
            prediction_label = "Real News"
        else:
            prediction_label = "Fake News"
        return prediction_label
    except Exception as e:
        st.error(f"An error occurred during prediction: {str(e)}")
        return None

# --- 3. Streamlit App Interface ---
st.title("Fake News Detection App") # App title

st.write("Enter the news article text and title below to check if it's likely to be fake news or real news.") # Description

title_input = st.text_input("Article Title:") # Text input for title
text_input = st.text_area("Article Text:", height=200) # Text area for article text

predict_button = st.button("Predict") # Prediction button

# --- 4. Prediction and Output ---
if predict_button:
    if text_input and title_input: # Check if both inputs are provided
        prediction = predict_fake_news(title_input, text_input, rf_model, vectorizer_text, vectorizer_title) # Get prediction
        if prediction is not None:
            st.write("## Prediction:") # Display prediction heading
            if prediction == "Fake News":
                st.error(f"⚠️ **Fake News** ⚠️") # Highlight Fake News prediction in red
            else:
                st.success(f"✅ **Real News** ✅") # Highlight Real News prediction in green
    else:
        st.warning("Please enter both article title and text to get a prediction.") # Warning if inputs are missing