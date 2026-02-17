import streamlit as st
import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ------------------ PAGE CONFIG ------------------
st.set_page_config(
    page_title="Emotion Detection AI",
    page_icon="🧠",
    layout="centered",
)

# ------------------ CUSTOM CSS ------------------
st.markdown("""
    <style>
    .main {
        background-color: #0E1117;
    }
    .stTextInput>div>div>input {
        background-color: #262730;
        color: white;
    }
    .big-title {
        font-size: 40px;
        font-weight: bold;
        text-align: center;
        color: #00FFFF;
    }
    .subtitle {
        text-align: center;
        font-size: 18px;
        color: #AAAAAA;
        margin-bottom: 30px;
    }
    .result-box {
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        font-size: 22px;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# ------------------ LOAD MODEL ------------------
@st.cache_resource
def load_resources():
    model = load_model("emotion_rnn_model.h5")

    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)

    with open("label_encoder.pkl", "rb") as f:
        encoder = pickle.load(f)

    return model, tokenizer, encoder


model, tokenizer, encoder = load_resources()

# ------------------ PREDICTION FUNCTION ------------------
def predict_emotion(text):
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=20, padding="post")
    pred = model.predict(padded)
    label = encoder.inverse_transform([np.argmax(pred)])
    confidence = np.max(pred)
    return label[0], confidence


# ------------------ UI ------------------

st.markdown('<div class="big-title">🧠 Emotion Detection AI</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Enter a sentence and let AI detect the emotion</div>', unsafe_allow_html=True)

user_input = st.text_input("✍️ Type your sentence here")

col1, col2 = st.columns([1,1])

with col1:
    predict_button = st.button("🔍 Analyze Emotion")

with col2:
    clear_button = st.button("🧹 Clear")

if clear_button:
    st.rerun()

if predict_button and user_input:

    with st.spinner("Analyzing emotion..."):
        emotion, confidence = predict_emotion(user_input)

    # Emotion Color Mapping
    color_map = {
        "happy": "#00FFAA",
        "sad": "#3399FF",
        "angry": "#FF4B4B",
        "stressed": "#FFA500",
        "romantic": "#FF69B4",
        "neutral": "#AAAAAA"
    }

    color = color_map.get(emotion.lower(), "#FFFFFF")

    st.markdown(
        f"""
        <div class="result-box" style="background-color:{color};">
            Detected Emotion: {emotion.upper()} <br><br>
            Confidence: {round(confidence*100, 2)}%
        </div>
        """,
        unsafe_allow_html=True
    )

    st.progress(float(confidence))

# ------------------ FOOTER ------------------
st.markdown("---")
st.markdown(
    "<center>Built with ❤️ using TensorFlow & Streamlit</center>",
    unsafe_allow_html=True
)
