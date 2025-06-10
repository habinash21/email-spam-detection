import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Title
st.title("📧 Email Spam Detection App")
st.write("Enter your message below to check if it's **Spam** or **Ham**.")

# Load and prepare the dataset (same Kaggle dataset)
@st.cache_data
def load_model():
    df = pd.read_csv("spam.csv", encoding='latin-1')[['v1', 'v2']]
    df.columns = ['label', 'message']
    
    # Label encode
    le = LabelEncoder()
    df['label_num'] = le.fit_transform(df['label'])  # spam=1, ham=0
    
    # Vectorize
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(df['message'])
    y = df['label_num']
    
    # Train
    model = MultinomialNB()
    model.fit(X, y)
    
    return model, vectorizer, le

model, vectorizer, le = load_model()

# User input
user_input = st.text_area("✉️ Paste your email/message here:", height=150)

if st.button("Check Spam"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        # Vectorize and predict
        user_vec = vectorizer.transform([user_input])
        prediction = model.predict(user_vec)[0]
        label = le.inverse_transform([prediction])[0]
        
        # Show result
        if label == "spam":
            st.error("🚫 This message is likely **SPAM**.")
        else:
            st.success("✅ This message is **HAM** (Not Spam).")
