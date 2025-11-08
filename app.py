# === FINAL STREAMLIT APP ===
# Models: TF-IDF, BiLSTM + Attention, TinyBERT Transformer (ONNX)

import streamlit as st
import joblib
import numpy as np
import onnxruntime as ort
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import backend as K

st.set_page_config(page_title="Intent Classifier — TF-IDF, BiLSTM, TinyBERT", layout="wide")
st.title("🤖 Intent Classifier — TF-IDF | BiLSTM | TinyBERT")

# ========== 1. LOAD TF-IDF ========== 
@st.cache_resource
def load_tfidf_model():
    model = joblib.load("tfidf_model.joblib")
    vectorizer = joblib.load("tfidf_vectorizer.joblib")
    return model, vectorizer

# ========== 2. LOAD BiLSTM + ATTENTION ========== 
@st.cache_resource
def load_bilstm_model():
    class AttentionLayer(tf.keras.layers.Layer):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)

        def build(self, input_shape):
            self.W = self.add_weight(name="att_weight", shape=(input_shape[-1], 1), initializer="normal")
            self.b = self.add_weight(name="att_bias", shape=(input_shape[1], 1), initializer="zeros")
            super().build(input_shape)

        def call(self, x):
            e = K.tanh(K.dot(x, self.W) + self.b)
            a = K.softmax(e, axis=1)
            output = x * a
            return K.sum(output, axis=1)

    model = tf.keras.models.load_model("bilstm_model.keras", custom_objects={"AttentionLayer": AttentionLayer}, compile=False)
    tokenizer = joblib.load("tokenizer.joblib")
    labels = joblib.load("labels_meta.joblib")["classes"]
    return model, tokenizer, labels

# ========== 3. LOAD TinyBERT (Transformer — ONNX) ==========
@st.cache_resource
def load_tinybert():
    ort_session = ort.InferenceSession("tinybert_github/tinybert_quant.onnx")
    tokenizer = joblib.load("tinybert_github/tokenizer.json") if "tokenizer.json" else None
    label_encoder = joblib.load("tinybert_github/label_encoder.joblib")
    return ort_session, tokenizer, label_encoder

# ========== SIDEBAR ==========
st.sidebar.header("⚙️ Select Model")
model_choice = st.sidebar.radio(
    "Choose a model:",
    ["TF-IDF + Logistic Regression", "BiLSTM + Attention", "TinyBERT (Transformer)"]
)

msg = st.text_area("💬 Enter customer message here:", height=100)

# ========== PREDICTION ==========
if st.button("🔍 Predict Intent"):
    if not msg.strip():
        st.warning("Please enter a message!")
    else:
        if model_choice == "TF-IDF + Logistic Regression":
            model, vectorizer = load_tfidf_model()
            X = vectorizer.transform([msg])
            intent = model.predict(X)[0]
            st.success(f"✅ Predicted Intent: **{intent}**")

        elif model_choice == "BiLSTM + Attention":
            model, tokenizer, labels = load_bilstm_model()
            seq = tokenizer.texts_to_sequences([msg])
            padded = pad_sequences(seq, maxlen=40)
            probs = model.predict(padded)[0]
            intent = labels[np.argmax(probs)]
            st.success(f"✅ Predicted Intent: **{intent}**")

        else:  # TinyBERT Transformer
            ort_session, tokenizer, label_encoder = load_tinybert()
            inputs = tokenizer(msg, return_tensors="np", padding=True, truncation=True, max_length=64)
            outputs = ort_session.run(None, {
                "input_ids": inputs["input_ids"],
                "attention_mask": inputs["attention_mask"]
            })
            pred_id = np.argmax(outputs[0])
            intent = label_encoder.inverse_transform([pred_id])[0]
            st.success(f"✅ Predicted Intent (TinyBERT): **{intent}**")

# ========== OPTIONAL: ANALYTICS ==========
st.sidebar.subheader("📊 Analytics")
if st.sidebar.checkbox("Show Dataset Insights"):
    st.header("📊 Dataset Insights & Accuracy")
    try:
        df = pd.read_csv("customer_intent_dataset_100k.csv")
        intent_counts = df["intent"].value_counts().head(10)
        st.subheader("🔹 Top 10 Intent Classes")
        fig, ax = plt.subplots()
        ax.bar(intent_counts.index, intent_counts.values, color="skyblue")
        plt.xticks(rotation=45)
        st.pyplot(fig)
    except:
        st.warning("Dataset file missing!")

    accuracy_data = pd.DataFrame({
        "Model": ["TF-IDF", "BiLSTM", "TinyBERT"],
        "Accuracy (%)": [71.1, 75.8, 78.5]  # Edit if needed
    })
    st.subheader("📈 Model Accuracy Comparison")
    st.bar_chart(accuracy_data.set_index("Model"))




