# === STEP 4: Combined Streamlit App — Baseline + Deep Model Switch (Fixed) ===

import streamlit as st
import joblib
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

# === Load Baseline Model (TF-IDF + Logistic Regression) ===
@st.cache_resource
def load_tfidf_model():
    model = joblib.load("tfidf_model.joblib")
    vectorizer = joblib.load("tfidf_vectorizer.joblib")
    return model, vectorizer


# === Load Deep Model (BiLSTM + Attention) ===
@st.cache_resource
def load_bilstm_model():
    # --- Recreate the custom AttentionLayer so Keras can reload it ---
    from tensorflow.keras import backend as K

    class AttentionLayer(tf.keras.layers.Layer):
        def __init__(self, **kwargs):
            super(AttentionLayer, self).__init__(**kwargs)

        def build(self, input_shape):
            self.W = self.add_weight(
                name="att_weight",
                shape=(input_shape[-1], 1),
                initializer="normal"
            )
            self.b = self.add_weight(
                name="att_bias",
                shape=(input_shape[1], 1),
                initializer="zeros"
            )
            super(AttentionLayer, self).build(input_shape)

        def call(self, x):
            e = K.tanh(K.dot(x, self.W) + self.b)
            a = K.softmax(e, axis=1)
            output = x * a
            return K.sum(output, axis=1)

    # --- Load model with custom layer ---
    model = tf.keras.models.load_model(
        "bilstm_model.keras",
        custom_objects={"AttentionLayer": AttentionLayer},
        compile=False
    )
    tokenizer = joblib.load("tokenizer.joblib")
    label_meta = joblib.load("labels_meta.joblib")
    labels = label_meta["classes"]
    return model, tokenizer, labels


# === Streamlit UI ===
st.set_page_config(page_title="Intent Classifier — Compare TF-IDF vs BiLSTM", layout="wide")
st.title("🤖 Intent Classifier — Compare TF-IDF vs BiLSTM")

st.sidebar.header("⚙️ Select Model")
model_choice = st.sidebar.radio("Choose Model:", ["TF-IDF + Logistic Regression", "BiLSTM + Attention"])

msg = st.text_area("💬 Enter Customer Message:", height=100)

if st.button("Predict Intent"):
    if not msg.strip():
        st.warning("Please enter a message before predicting.")
    else:
        if model_choice == "TF-IDF + Logistic Regression":
            model, vectorizer = load_tfidf_model()
            x = vectorizer.transform([msg])
            probs = model.predict_proba(x)[0]
            classes = model.classes_
            top_preds = sorted(zip(classes, probs), key=lambda x: x[1], reverse=True)[:5]
            st.success(f"**Predicted Intent:** {model.predict(x)[0]}")
            st.write("Top Predictions:")
            st.table({
                "Intent": [p[0] for p in top_preds],
                "Probability (%)": [round(p[1]*100, 2) for p in top_preds]
            })

        else:
            model, tokenizer, labels = load_bilstm_model()
            seq = tokenizer.texts_to_sequences([msg])
            padded = pad_sequences(seq, maxlen=40, padding='post', truncating='post')
            probs = model.predict(padded)[0]
            top_idx = np.argmax(probs)
            pred_label = labels[top_idx]
            top_preds = sorted(zip(labels, probs), key=lambda x: x[1], reverse=True)[:5]
            st.success(f"**Predicted Intent:** {pred_label}")
            st.write("Top Predictions:")
            st.table({
                "Intent": [p[0] for p in top_preds],
                "Probability (%)": [round(float(p[1])*100, 2) for p in top_preds]
            })

