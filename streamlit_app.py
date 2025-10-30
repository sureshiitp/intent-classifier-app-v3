
import streamlit as st, joblib

st.set_page_config(page_title="Intent Classifier (TF-IDF Baseline)", layout="wide")

@st.cache_resource
def load_artifacts():
    model = joblib.load("tfidf_model.joblib")
    vectorizer = joblib.load("tfidf_vectorizer.joblib")
    return model, vectorizer

model, vectorizer = load_artifacts()

st.title("🤖 Intent Classifier — TF-IDF + Logistic Regression (Baseline)")
msg = st.text_input("Enter a customer message:")

if st.button("Predict") and msg:
    X = vectorizer.transform([msg])
    intent = model.predict(X)[0]
    st.success(f"Predicted intent: **{intent}**")
