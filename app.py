import streamlit as st
import numpy as np
import pandas as pd
import pickle
import os
import time
import json
from collections import Counter
import plotly.graph_objects as go
import plotly.express as px

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="EmoSense AI · Emotion Intelligence",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────
# ADVANCED CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=JetBrains+Mono:wght@300;400;600&display=swap');

:root {
    --bg: #06090f;
    --surface: #0d1117;
    --card: #111827;
    --border: #1f2937;
    --accent: #6ee7f7;
    --accent2: #a78bfa;
    --accent3: #fb7185;
    --text: #f1f5f9;
    --muted: #64748b;
    --happy: #22d3ee;
    --sad: #818cf8;
    --angry: #f87171;
    --stressed: #fb923c;
    --romantic: #f472b6;
    --neutral: #94a3b8;
}

* { box-sizing: border-box; }

html, body, [data-testid="stAppViewContainer"] {
    background: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'Syne', sans-serif;
}

[data-testid="stAppViewContainer"] {
    background: radial-gradient(ellipse 80% 60% at 50% -20%, #0e1f3b 0%, var(--bg) 70%) !important;
}

[data-testid="stHeader"] { display: none; }
[data-testid="stSidebar"] { background: var(--surface) !important; border-right: 1px solid var(--border); }
.block-container { padding: 1.5rem 2rem 3rem 2rem !important; max-width: 1200px; }

/* Tabs */
[data-testid="stTabs"] button {
    font-family: 'Syne', sans-serif !important;
    font-weight: 600;
    color: var(--muted) !important;
    border-bottom: 2px solid transparent;
    transition: all 0.2s;
    font-size: 0.9rem;
    letter-spacing: 0.03em;
}
[data-testid="stTabs"] button[aria-selected="true"] {
    color: var(--accent) !important;
    border-bottom: 2px solid var(--accent) !important;
}
[data-testid="stTabs"] [data-baseweb="tab-list"] {
    background: transparent !important;
    border-bottom: 1px solid var(--border);
    gap: 0.5rem;
}

/* Input */
textarea, input[type="text"] {
    background: var(--card) !important;
    border: 1px solid var(--border) !important;
    color: var(--text) !important;
    font-family: 'JetBrains Mono', monospace !important;
    border-radius: 12px !important;
    font-size: 1rem !important;
    transition: border-color 0.2s !important;
}
textarea:focus, input[type="text"]:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 2px rgba(110,231,247,0.15) !important;
}

/* Buttons */
[data-testid="stButton"] button {
    background: linear-gradient(135deg, #1a3a5c, #0e2a4a) !important;
    border: 1px solid var(--accent) !important;
    color: var(--accent) !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    border-radius: 10px !important;
    letter-spacing: 0.05em;
    transition: all 0.2s !important;
    text-transform: uppercase;
    font-size: 0.8rem !important;
}
[data-testid="stButton"] button:hover {
    background: linear-gradient(135deg, #1e4a7a, #0e3560) !important;
    box-shadow: 0 0 20px rgba(110,231,247,0.25) !important;
    transform: translateY(-1px);
}

/* Slider */
[data-testid="stSlider"] [class*="thumb"] { background: var(--accent) !important; }
[data-testid="stSlider"] [class*="track"]:first-child { background: var(--accent) !important; }

/* Metrics */
[data-testid="metric-container"] {
    background: var(--card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px;
    padding: 1rem !important;
}
[data-testid="stMetricValue"] { color: var(--accent) !important; font-family: 'JetBrains Mono'; }

/* Plotly charts */
.js-plotly-plot { border-radius: 12px; overflow: hidden; }

/* Progress bars */
[data-testid="stProgress"] > div > div {
    background: linear-gradient(90deg, var(--accent), var(--accent2)) !important;
    border-radius: 99px;
}
[data-testid="stProgress"] > div {
    background: var(--border) !important;
    border-radius: 99px;
}

/* Spinner */
[data-testid="stSpinner"] { color: var(--accent) !important; }

/* Dataframes */
[data-testid="stDataFrame"] { border-radius: 12px; overflow: hidden; border: 1px solid var(--border); }

/* Expander */
details { background: var(--card) !important; border: 1px solid var(--border) !important; border-radius: 12px !important; padding: 0.5rem; }
summary { color: var(--muted) !important; font-family: 'Syne', sans-serif; }

div[data-testid="stSelectbox"] > div { background: var(--card) !important; border: 1px solid var(--border) !important; border-radius: 10px !important; }

/* scrollbar */
::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 99px; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# EMOTION CONFIG
# ─────────────────────────────────────────────
EMOTIONS = {
    "happy":    {"emoji": "😄", "color": "#22d3ee", "glow": "#22d3ee44", "desc": "Joy & Positivity"},
    "sad":      {"emoji": "😢", "color": "#818cf8", "glow": "#818cf844", "desc": "Sorrow & Melancholy"},
    "angry":    {"emoji": "😠", "color": "#f87171", "glow": "#f8717144", "desc": "Anger & Frustration"},
    "stressed": {"emoji": "😰", "color": "#fb923c", "glow": "#fb923c44", "desc": "Stress & Anxiety"},
    "romantic": {"emoji": "💕", "color": "#f472b6", "glow": "#f472b644", "desc": "Love & Romance"},
    "neutral":  {"emoji": "😐", "color": "#94a3b8", "glow": "#94a3b844", "desc": "Calm & Balanced"},
}

# ─────────────────────────────────────────────
# MODEL TRAINING (Self-Contained)
# ─────────────────────────────────────────────
MODEL_PATH = "/tmp/emosense_model.h5"
TOK_PATH   = "/tmp/emosense_tokenizer.pkl"
ENC_PATH   = "/tmp/emosense_encoder.pkl"
HIST_PATH  = "/tmp/emosense_history.json"

@st.cache_resource(show_spinner=False)
def load_or_train_model():
    """Train a full RNN model with LSTM + Bidirectional layers."""
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.layers import (
        Embedding, LSTM, Bidirectional, Dense, Dropout,
        GlobalAveragePooling1D, Conv1D, MaxPooling1D, BatchNormalization
    )
    from tensorflow.keras.preprocessing.text import Tokenizer
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from sklearn.preprocessing import LabelEncoder
    from sklearn.model_selection import train_test_split

    # Load dataset
    df = pd.read_csv("emotion_dataset.csv")
    df.columns = df.columns.str.strip()
    df = df.dropna()

    texts  = df["text"].tolist()
    labels = df["emotion"].tolist()

    # Tokenizer
    MAX_VOCAB = 5000
    MAX_LEN   = 30
    tokenizer = Tokenizer(num_words=MAX_VOCAB, oov_token="<OOV>")
    tokenizer.fit_on_texts(texts)
    seqs    = tokenizer.texts_to_sequences(texts)
    padded  = pad_sequences(seqs, maxlen=MAX_LEN, padding="post", truncating="post")

    # Label Encoder
    encoder = LabelEncoder()
    y = encoder.fit_transform(labels)
    num_classes = len(encoder.classes_)

    X_train, X_val, y_train, y_val = train_test_split(
        padded, y, test_size=0.15, random_state=42, stratify=y
    )

    # ── ADVANCED RNN ARCHITECTURE ──────────────────────────────────────────────
    # Conv1D feature extraction → Bidirectional LSTM → LSTM stacking
    model = Sequential([
        Embedding(MAX_VOCAB, 128, input_length=MAX_LEN),

        # Convolutional feature extraction (n-gram patterns)
        Conv1D(128, 3, activation="relu", padding="same"),
        BatchNormalization(),
        MaxPooling1D(2),
        Dropout(0.2),

        # Bidirectional LSTM (captures context from both directions)
        Bidirectional(LSTM(128, return_sequences=True, dropout=0.2, recurrent_dropout=0.1)),
        BatchNormalization(),

        # Stacked LSTM
        LSTM(64, return_sequences=True, dropout=0.2, recurrent_dropout=0.1),
        BatchNormalization(),

        # Final LSTM
        LSTM(32, dropout=0.2),

        # Dense head
        Dense(64, activation="relu"),
        Dropout(0.3),
        Dense(32, activation="relu"),
        Dense(num_classes, activation="softmax"),
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    callbacks = [
        EarlyStopping(monitor="val_accuracy", patience=8, restore_best_weights=True, verbose=0),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, verbose=0)
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=60,
        batch_size=32,
        callbacks=callbacks,
        verbose=0
    )

    # Save artifacts
    model.save(MODEL_PATH)
    with open(TOK_PATH, "wb") as f: pickle.dump(tokenizer, f)
    with open(ENC_PATH, "wb") as f: pickle.dump(encoder, f)
    with open(HIST_PATH, "w") as f:
        json.dump({
            "accuracy": history.history["accuracy"],
            "val_accuracy": history.history["val_accuracy"],
            "loss": history.history["loss"],
            "val_loss": history.history["val_loss"],
        }, f)

    val_acc = max(history.history["val_accuracy"])
    return model, tokenizer, encoder, history.history, MAX_LEN, val_acc


# ─────────────────────────────────────────────
# PREDICTION
# ─────────────────────────────────────────────
def predict_emotion(text, model, tokenizer, encoder, max_len):
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    seq    = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=max_len, padding="post", truncating="post")
    probs  = model.predict(padded, verbose=0)[0]
    classes = encoder.classes_
    pred_idx = np.argmax(probs)
    return classes[pred_idx], float(probs[pred_idx]), dict(zip(classes, probs.tolist()))


# ─────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []   # list of {text, emotion, confidence, probs}
if "batch_results" not in st.session_state:
    st.session_state.batch_results = []

# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.markdown("""
<div style="text-align:center; padding: 2.5rem 0 1.5rem 0;">
    <div style="font-size:0.8rem; letter-spacing:0.3em; color:#6ee7f7; text-transform:uppercase; margin-bottom:0.5rem; font-family:'JetBrains Mono';">
        ◈ Neural Language Intelligence ◈
    </div>
    <h1 style="font-size:3.2rem; font-weight:800; margin:0; background: linear-gradient(135deg, #6ee7f7, #a78bfa, #fb7185); -webkit-background-clip:text; -webkit-text-fill-color:transparent; letter-spacing:-0.02em;">
        EmoSense AI
    </h1>
    <p style="color:#64748b; font-size:1rem; margin-top:0.75rem; font-family:'JetBrains Mono'; font-weight:300;">
        Bidirectional LSTM · Conv1D Feature Extraction · Stacked RNN Architecture
    </p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# TRAINING STATUS
# ─────────────────────────────────────────────
with st.spinner("🧬 Initializing neural network — training RNN on emotion corpus..."):
    model, tokenizer, encoder, train_history, MAX_LEN, val_acc = load_or_train_model()

st.markdown(f"""
<div style="display:flex; justify-content:center; margin-bottom:1.5rem;">
    <div style="background:#0d2a1a; border:1px solid #16a34a; border-radius:99px; padding:0.35rem 1.2rem; font-family:'JetBrains Mono'; font-size:0.8rem; color:#4ade80;">
        ✓ Model Ready · Val Accuracy: {val_acc*100:.1f}% · Architecture: Conv1D + BiLSTM + Stacked LSTM
    </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🔍  Analyze", 
    "📋  Batch Mode", 
    "📊  Session Analytics",
    "🧠  Model Insights",
    "📖  About RNN"
])

# ══════════════════════════════════════════════
# TAB 1 — SINGLE ANALYZE
# ══════════════════════════════════════════════
with tab1:
    st.markdown("<br>", unsafe_allow_html=True)
    
    col_left, col_right = st.columns([1.1, 0.9], gap="large")

    with col_left:
        st.markdown("""
        <div style="font-size:0.75rem; letter-spacing:0.15em; color:#6ee7f7; text-transform:uppercase; font-family:'JetBrains Mono'; margin-bottom:0.5rem;">
            Input Terminal
        </div>
        """, unsafe_allow_html=True)

        user_text = st.text_area(
            label="",
            placeholder="Type or paste any sentence here...\ne.g. 'I can't stop thinking about her smile'\n     'Everything is falling apart today'\n     'I am absolutely thrilled about this!'",
            height=155,
            key="main_input",
            label_visibility="collapsed"
        )

        c1, c2, c3 = st.columns(3)
        with c1: analyze_btn = st.button("⚡ ANALYZE", use_container_width=True)
        with c2: clear_btn   = st.button("✕ CLEAR", use_container_width=True)
        with c3: save_btn    = st.button("＋ SAVE TO LOG", use_container_width=True)

        # Quick demo sentences
        st.markdown("""<div style="font-size:0.72rem; color:#475569; margin-top:0.75rem; font-family:'JetBrains Mono';">Quick demos →</div>""", unsafe_allow_html=True)
        demo_cols = st.columns(3)
        demos = [
            ("😄 Joy", "I feel absolutely wonderful and full of energy today!"),
            ("😢 Sad", "I miss them so much, everything feels empty now."),
            ("😠 Anger", "This is completely unfair and I am furious about it!"),
        ]
        demo_cols2 = st.columns(3)
        demos2 = [
            ("😰 Stress", "I have so many deadlines and I can't handle all of this."),
            ("💕 Love", "I am deeply in love and my heart is so full."),
            ("😐 Calm", "The weather is pleasant and I feel at ease today."),
        ]
        for col, (label, sentence) in zip(demo_cols, demos):
            with col:
                if st.button(label, use_container_width=True, key=f"demo_{label}"):
                    st.session_state["demo_text"] = sentence
                    st.rerun()
        for col, (label, sentence) in zip(demo_cols2, demos2):
            with col:
                if st.button(label, use_container_width=True, key=f"demo2_{label}"):
                    st.session_state["demo_text"] = sentence
                    st.rerun()

    with col_right:
        st.markdown("""
        <div style="font-size:0.75rem; letter-spacing:0.15em; color:#6ee7f7; text-transform:uppercase; font-family:'JetBrains Mono'; margin-bottom:0.5rem;">
            Neural Output
        </div>
        """, unsafe_allow_html=True)

        result_placeholder = st.empty()

    # Handle demo injection
    active_text = user_text
    if "demo_text" in st.session_state and not user_text:
        active_text = st.session_state.pop("demo_text")

    if clear_btn:
        st.rerun()

    if analyze_btn or (active_text and active_text != user_text):
        text_to_analyze = active_text or user_text
        if text_to_analyze.strip():
            emotion, confidence, all_probs = predict_emotion(
                text_to_analyze, model, tokenizer, encoder, MAX_LEN
            )
            ecfg = EMOTIONS.get(emotion.lower(), EMOTIONS["neutral"])

            # Main result card
            with col_right:
                st.markdown(f"""
                <div style="
                    background: linear-gradient(135deg, {ecfg['glow']}, #0d1117 60%);
                    border: 1px solid {ecfg['color']}66;
                    border-radius: 16px; padding: 1.5rem; text-align: center;
                    box-shadow: 0 0 30px {ecfg['glow']};
                ">
                    <div style="font-size:3.5rem; line-height:1; margin-bottom:0.5rem;">{ecfg['emoji']}</div>
                    <div style="font-size:2rem; font-weight:800; color:{ecfg['color']}; letter-spacing:0.05em;">
                        {emotion.upper()}
                    </div>
                    <div style="color:#94a3b8; font-size:0.85rem; margin:0.25rem 0 1rem 0; font-family:'JetBrains Mono';">
                        {ecfg['desc']}
                    </div>
                    <div style="background:#ffffff10; border-radius:99px; height:6px; margin-bottom:0.5rem; overflow:hidden;">
                        <div style="width:{confidence*100:.1f}%; height:100%; background:linear-gradient(90deg,{ecfg['color']},{ecfg['color']}88); border-radius:99px;"></div>
                    </div>
                    <div style="font-family:'JetBrains Mono'; font-size:0.9rem; color:{ecfg['color']}; font-weight:600;">
                        {confidence*100:.1f}% confidence
                    </div>
                </div>
                """, unsafe_allow_html=True)

            # Probability chart
            st.markdown("<br>", unsafe_allow_html=True)
            sorted_probs = sorted(all_probs.items(), key=lambda x: x[1], reverse=True)
            labels = [f"{EMOTIONS[e]['emoji']} {e.capitalize()}" for e, _ in sorted_probs]
            values = [v * 100 for _, v in sorted_probs]
            colors = [EMOTIONS[e]["color"] for e, _ in sorted_probs]

            fig = go.Figure(go.Bar(
                x=values, y=labels, orientation="h",
                marker=dict(color=colors, line=dict(width=0)),
                text=[f"{v:.1f}%" for v in values],
                textposition="inside",
                textfont=dict(family="JetBrains Mono", size=12, color="#ffffff"),
            ))
            fig.update_layout(
                plot_bgcolor="#0d1117", paper_bgcolor="#0d1117",
                font=dict(family="Syne", color="#94a3b8", size=12),
                xaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
                yaxis=dict(showgrid=False, tickfont=dict(size=13)),
                margin=dict(l=10, r=20, t=10, b=10),
                height=220, showlegend=False,
            )
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

            # Radar chart
            radar_vals = [all_probs.get(e, 0) * 100 for e in EMOTIONS.keys()]
            fig_radar = go.Figure(go.Scatterpolar(
                r=radar_vals + [radar_vals[0]],
                theta=[f"{v['emoji']} {k.capitalize()}" for k, v in EMOTIONS.items()] + [f"{list(EMOTIONS.values())[0]['emoji']} {list(EMOTIONS.keys())[0].capitalize()}"],
                fill="toself",
                fillcolor="rgba(110,231,247,0.12)",
                line=dict(color="#6ee7f7", width=2),
                marker=dict(size=5, color="#6ee7f7"),
            ))
            fig_radar.update_layout(
                polar=dict(
                    bgcolor="#0d1117",
                    radialaxis=dict(visible=True, range=[0, 100], showticklabels=False, gridcolor="#1f2937", linecolor="#1f2937"),
                    angularaxis=dict(tickfont=dict(size=11, family="Syne"), gridcolor="#1f2937", linecolor="#1f2937"),
                ),
                paper_bgcolor="#111827", margin=dict(l=40, r=40, t=30, b=30),
                height=270, showlegend=False,
                font=dict(color="#94a3b8"),
            )
            st.markdown("**Emotion Radar Map**")
            st.plotly_chart(fig_radar, use_container_width=True, config={"displayModeBar": False})

            # Save to session history
            entry = {
                "text": text_to_analyze[:80] + ("…" if len(text_to_analyze) > 80 else ""),
                "emotion": emotion,
                "confidence": confidence,
                "probs": all_probs,
            }
            if save_btn or analyze_btn:
                st.session_state.history.append(entry)

        else:
            with col_right:
                st.markdown("""
                <div style="border:1px dashed #1f2937; border-radius:16px; padding:3rem; text-align:center; color:#374151;">
                    <div style="font-size:2.5rem; margin-bottom:0.5rem;">🧬</div>
                    <div style="font-family:'JetBrains Mono'; font-size:0.85rem;">Awaiting input...</div>
                </div>
                """, unsafe_allow_html=True)


# ══════════════════════════════════════════════
# TAB 2 — BATCH MODE
# ══════════════════════════════════════════════
with tab2:
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size:0.75rem; letter-spacing:0.15em; color:#6ee7f7; text-transform:uppercase; font-family:'JetBrains Mono'; margin-bottom:0.5rem;">
        Batch Emotion Analyzer — one sentence per line
    </div>
    """, unsafe_allow_html=True)

    batch_text = st.text_area(
        label="",
        placeholder="I am feeling great today!\nThis situation is making me really anxious.\nI love spending time with family.\nWhy is everything so wrong?\nI just feel okay, nothing special.",
        height=200,
        label_visibility="collapsed",
        key="batch_input"
    )

    bc1, bc2 = st.columns([1, 4])
    with bc1:
        batch_btn = st.button("⚡ ANALYZE ALL", use_container_width=True)

    if batch_btn and batch_text.strip():
        sentences = [s.strip() for s in batch_text.strip().split("\n") if s.strip()]
        results = []
        prog = st.progress(0)
        for i, sent in enumerate(sentences):
            emotion, conf, probs = predict_emotion(sent, model, tokenizer, encoder, MAX_LEN)
            results.append({"Sentence": sent, "Emotion": emotion.capitalize(), "Confidence %": round(conf * 100, 1)})
            prog.progress((i + 1) / len(sentences))
        prog.empty()
        st.session_state.batch_results = results

    if st.session_state.batch_results:
        df_res = pd.DataFrame(st.session_state.batch_results)

        # Color rows by emotion
        def highlight_emotion(row):
            color = EMOTIONS.get(row["Emotion"].lower(), {}).get("color", "#ffffff")
            return [f"color: {color}"] * len(row)

        st.dataframe(
            df_res.style.apply(highlight_emotion, axis=1),
            use_container_width=True,
            hide_index=True,
        )

        # Distribution pie
        counts = Counter(df_res["Emotion"].str.lower())
        fig_pie = go.Figure(go.Pie(
            labels=[f"{EMOTIONS[e]['emoji']} {e.capitalize()}" for e in counts.keys()],
            values=list(counts.values()),
            marker=dict(colors=[EMOTIONS[e]["color"] for e in counts.keys()], line=dict(width=2, color="#06090f")),
            textfont=dict(family="Syne", size=13),
            hole=0.45,
        ))
        fig_pie.update_layout(
            paper_bgcolor="#111827", font=dict(color="#94a3b8"),
            margin=dict(l=10, r=10, t=30, b=10), height=300,
            legend=dict(font=dict(size=12, family="Syne")),
        )
        st.markdown("**Batch Emotion Distribution**")
        st.plotly_chart(fig_pie, use_container_width=True, config={"displayModeBar": False})

        # Download CSV
        csv = df_res.to_csv(index=False)
        st.download_button(
            label="⬇ Download CSV", data=csv,
            file_name="emosense_batch_results.csv", mime="text/csv"
        )


# ══════════════════════════════════════════════
# TAB 3 — SESSION ANALYTICS
# ══════════════════════════════════════════════
with tab3:
    st.markdown("<br>", unsafe_allow_html=True)

    if not st.session_state.history:
        st.markdown("""
        <div style="text-align:center; padding:4rem 0; color:#374151;">
            <div style="font-size:3rem; margin-bottom:1rem;">📊</div>
            <div style="font-family:'JetBrains Mono'; font-size:0.9rem;">No analyses saved yet. Use the Analyze tab and click Save.</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        hist = st.session_state.history
        emotions_detected = [h["emotion"] for h in hist]
        counts = Counter(emotions_detected)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Analyses", len(hist))
        m2.metric("Dominant Emotion", max(counts, key=counts.get).capitalize())
        m3.metric("Avg Confidence", f"{np.mean([h['confidence'] for h in hist])*100:.1f}%")
        m4.metric("Emotions Detected", len(counts))

        st.markdown("<br>", unsafe_allow_html=True)

        # Timeline chart
        fig_line = go.Figure()
        for i, h in enumerate(hist):
            ecfg = EMOTIONS.get(h["emotion"].lower(), EMOTIONS["neutral"])
            fig_line.add_trace(go.Scatter(
                x=[i], y=[h["confidence"] * 100],
                mode="markers",
                marker=dict(size=16, color=ecfg["color"], line=dict(width=2, color="#0d1117")),
                name=h["emotion"].capitalize(),
                hovertemplate=f"<b>{h['emotion'].upper()}</b><br>{h['text']}<br>Confidence: {h['confidence']*100:.1f}%<extra></extra>",
                showlegend=False,
            ))
        fig_line.update_layout(
            plot_bgcolor="#0d1117", paper_bgcolor="#111827",
            font=dict(family="Syne", color="#94a3b8"),
            xaxis=dict(title="Analysis #", gridcolor="#1f2937", zeroline=False),
            yaxis=dict(title="Confidence %", range=[0, 105], gridcolor="#1f2937"),
            margin=dict(l=10, r=10, t=30, b=10), height=260,
            title=dict(text="Confidence Timeline", font=dict(size=14)),
        )
        st.plotly_chart(fig_line, use_container_width=True, config={"displayModeBar": False})

        # Emotion frequency bar
        sorted_counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)
        fig_bar = go.Figure(go.Bar(
            x=[EMOTIONS[e]["emoji"] + " " + e.capitalize() for e, _ in sorted_counts],
            y=[v for _, v in sorted_counts],
            marker=dict(color=[EMOTIONS[e]["color"] for e, _ in sorted_counts], line=dict(width=0)),
            text=[v for _, v in sorted_counts],
            textposition="outside",
            textfont=dict(family="JetBrains Mono", size=14, color="#ffffff"),
        ))
        fig_bar.update_layout(
            plot_bgcolor="#0d1117", paper_bgcolor="#111827",
            font=dict(family="Syne", color="#94a3b8"),
            xaxis=dict(showgrid=False), yaxis=dict(showgrid=False, showticklabels=False),
            margin=dict(l=10, r=10, t=30, b=10), height=250,
            title=dict(text="Session Emotion Frequency", font=dict(size=14)),
        )
        st.plotly_chart(fig_bar, use_container_width=True, config={"displayModeBar": False})

        # History table
        st.markdown("**Full Session Log**")
        df_hist = pd.DataFrame([{
            "Text": h["text"],
            "Emotion": EMOTIONS[h["emotion"].lower()]["emoji"] + " " + h["emotion"].capitalize(),
            "Confidence": f"{h['confidence']*100:.1f}%"
        } for h in hist])
        st.dataframe(df_hist, use_container_width=True, hide_index=True)

        if st.button("🗑 Clear History"):
            st.session_state.history = []
            st.rerun()


# ══════════════════════════════════════════════
# TAB 4 — MODEL INSIGHTS
# ══════════════════════════════════════════════
with tab4:
    st.markdown("<br>", unsafe_allow_html=True)

    col_a, col_b = st.columns(2, gap="large")

    with col_a:
        st.markdown("#### 📈 Training Curves")
        fig_acc = go.Figure()
        fig_acc.add_trace(go.Scatter(
            y=train_history["accuracy"], name="Train Accuracy",
            line=dict(color="#6ee7f7", width=2),
            fill="tozeroy", fillcolor="rgba(110,231,247,0.06)"
        ))
        fig_acc.add_trace(go.Scatter(
            y=train_history["val_accuracy"], name="Val Accuracy",
            line=dict(color="#a78bfa", width=2, dash="dot"),
        ))
        fig_acc.update_layout(
            plot_bgcolor="#0d1117", paper_bgcolor="#111827",
            font=dict(family="Syne", color="#94a3b8"),
            xaxis=dict(title="Epoch", gridcolor="#1f2937"),
            yaxis=dict(title="Accuracy", gridcolor="#1f2937", range=[0, 1.05]),
            legend=dict(bgcolor="transparent"),
            margin=dict(l=10, r=10, t=10, b=10), height=250,
        )
        st.plotly_chart(fig_acc, use_container_width=True, config={"displayModeBar": False})

        fig_loss = go.Figure()
        fig_loss.add_trace(go.Scatter(
            y=train_history["loss"], name="Train Loss",
            line=dict(color="#fb7185", width=2),
        ))
        fig_loss.add_trace(go.Scatter(
            y=train_history["val_loss"], name="Val Loss",
            line=dict(color="#fb923c", width=2, dash="dot"),
        ))
        fig_loss.update_layout(
            plot_bgcolor="#0d1117", paper_bgcolor="#111827",
            font=dict(family="Syne", color="#94a3b8"),
            xaxis=dict(title="Epoch", gridcolor="#1f2937"),
            yaxis=dict(title="Loss", gridcolor="#1f2937"),
            legend=dict(bgcolor="transparent"),
            margin=dict(l=10, r=10, t=10, b=10), height=220,
        )
        st.plotly_chart(fig_loss, use_container_width=True, config={"displayModeBar": False})

    with col_b:
        st.markdown("#### 🏗 Architecture")
        st.markdown("""
        <div style="background:#0d1117; border:1px solid #1f2937; border-radius:12px; padding:1.2rem; font-family:'JetBrains Mono'; font-size:0.8rem; line-height:2;">
            <div style="color:#6ee7f7;">┌─────────────────────────────┐</div>
            <div style="color:#94a3b8;">│  <span style="color:#a78bfa;">Embedding</span>  vocab=5000, dim=128 │</div>
            <div style="color:#94a3b8;">│  <span style="color:#22d3ee;">Conv1D</span>     128 filters, k=3    │</div>
            <div style="color:#94a3b8;">│  BatchNorm + MaxPool(2)      │</div>
            <div style="color:#94a3b8;">│  Dropout(0.2)                │</div>
            <div style="color:#94a3b8;">│  <span style="color:#f472b6;">BiLSTM</span>     128 units ←→       │</div>
            <div style="color:#94a3b8;">│  BatchNorm                   │</div>
            <div style="color:#94a3b8;">│  <span style="color:#f472b6;">LSTM</span>       64 units           │</div>
            <div style="color:#94a3b8;">│  BatchNorm                   │</div>
            <div style="color:#94a3b8;">│  <span style="color:#f472b6;">LSTM</span>       32 units           │</div>
            <div style="color:#94a3b8;">│  Dense(64) → Dense(32)       │</div>
            <div style="color:#94a3b8;">│  <span style="color:#fb923c;">Softmax</span>    6 classes          │</div>
            <div style="color:#6ee7f7;">└─────────────────────────────┘</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("#### ⚙️ Hyperparameters")
        params = {
            "Max Vocab Size": "5,000 tokens",
            "Sequence Length": "30 tokens",
            "Embedding Dim": "128",
            "Optimizer": "Adam (lr=0.001)",
            "Batch Size": "32",
            "Max Epochs": "60",
            "Early Stopping": "patience=8",
            "LR Scheduler": "ReduceLROnPlateau",
            "Dropout": "0.2–0.3",
            "Classes": "6 emotions",
        }
        for k, v in params.items():
            st.markdown(f"""
            <div style="display:flex; justify-content:space-between; padding:0.35rem 0; border-bottom:1px solid #1f2937; font-family:'JetBrains Mono'; font-size:0.8rem;">
                <span style="color:#64748b;">{k}</span>
                <span style="color:#6ee7f7;">{v}</span>
            </div>
            """, unsafe_allow_html=True)

        # Dataset stats
        st.markdown("<br>#### 📁 Dataset")
        df_data = pd.read_csv("emotion_dataset.csv")
        ec = df_data["emotion"].value_counts()
        fig_ec = go.Figure(go.Bar(
            x=[EMOTIONS[e]["emoji"]+" "+e.capitalize() for e in ec.index],
            y=ec.values,
            marker=dict(color=[EMOTIONS[e]["color"] for e in ec.index], line=dict(width=0)),
            text=ec.values, textposition="outside",
            textfont=dict(family="JetBrains Mono", color="#ffffff", size=12),
        ))
        fig_ec.update_layout(
            plot_bgcolor="#0d1117", paper_bgcolor="#111827",
            font=dict(family="Syne", color="#94a3b8"),
            xaxis=dict(showgrid=False), yaxis=dict(showgrid=False, showticklabels=False),
            margin=dict(l=0, r=0, t=10, b=0), height=220,
        )
        st.plotly_chart(fig_ec, use_container_width=True, config={"displayModeBar": False})


# ══════════════════════════════════════════════
# TAB 5 — ABOUT RNN
# ══════════════════════════════════════════════
with tab5:
    st.markdown("<br>", unsafe_allow_html=True)

    concepts = [
        {
            "title": "RNN (Recurrent Neural Network)",
            "icon": "🔄",
            "color": "#6ee7f7",
            "body": "RNNs process sequences step-by-step, maintaining a hidden state that carries memory of previous tokens. Unlike feedforward networks, they model temporal dependencies — perfect for language where word order matters deeply.",
        },
        {
            "title": "LSTM (Long Short-Term Memory)",
            "icon": "🧠",
            "color": "#a78bfa",
            "body": "LSTMs solve the vanishing gradient problem with 3 gates: Input (what to remember), Forget (what to discard), Output (what to expose). This lets the model capture dependencies across long sequences without degradation.",
        },
        {
            "title": "Bidirectional LSTM",
            "icon": "↔️",
            "color": "#fb7185",
            "body": "Standard LSTMs only see past context. Bidirectional LSTMs run two LSTMs — one forward, one backward — and concatenate their outputs. This means every token is contextualized by both what came before AND after it.",
        },
        {
            "title": "Conv1D Feature Extraction",
            "icon": "🔎",
            "color": "#fb923c",
            "body": "1D Convolutions act as n-gram detectors, scanning sliding windows over the token sequence. They extract local patterns (phrases, negations, emotional keywords) before feeding into the RNN layers for temporal modeling.",
        },
        {
            "title": "Stacked RNN",
            "icon": "📚",
            "color": "#f472b6",
            "body": "Stacking multiple LSTM layers creates a deep temporal hierarchy. Lower layers learn surface patterns (syntax), higher layers learn abstract representations (sentiment, intent). Each layer's output becomes the next layer's input sequence.",
        },
        {
            "title": "BatchNormalization",
            "icon": "⚖️",
            "color": "#4ade80",
            "body": "Normalizes activations at each layer to stabilize and accelerate training. Reduces internal covariate shift — the problem where each layer must adapt to a changing distribution from the previous layer.",
        },
    ]

    for i in range(0, len(concepts), 2):
        cols = st.columns(2, gap="large")
        for j, col in enumerate(cols):
            if i + j < len(concepts):
                c = concepts[i + j]
                with col:
                    st.markdown(f"""
                    <div style="background:#0d1117; border:1px solid {c['color']}33; border-left:3px solid {c['color']}; border-radius:12px; padding:1.2rem; margin-bottom:1rem;">
                        <div style="font-size:1.5rem; margin-bottom:0.5rem;">{c['icon']}</div>
                        <div style="font-size:0.95rem; font-weight:700; color:{c['color']}; margin-bottom:0.5rem;">{c['title']}</div>
                        <div style="font-size:0.83rem; color:#94a3b8; line-height:1.7;">{c['body']}</div>
                    </div>
                    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="background:#0d1117; border:1px solid #1f2937; border-radius:12px; padding:1.5rem; margin-top:0.5rem;">
        <div style="font-size:0.85rem; font-weight:700; color:#6ee7f7; margin-bottom:0.75rem;">📐 How EmoSense Works — End to End</div>
        <div style="font-family:'JetBrains Mono'; font-size:0.75rem; color:#64748b; line-height:2.2;">
            Raw Text → Tokenizer → Integer Sequences → Padding (len=30)<br>
            → Embedding (128d) → Conv1D (local patterns) → MaxPool<br>
            → BiLSTM (bidirectional context) → LSTM × 2 (deep temporal)<br>
            → Dense(64) → Dense(32) → Softmax(6) → Emotion Probabilities
        </div>
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("""
<div style="text-align:center; border-top:1px solid #1f2937; padding-top:1.5rem;">
    <span style="font-family:'JetBrains Mono'; font-size:0.75rem; color:#374151;">
        EmoSense AI · Built with TensorFlow & Streamlit · 
        Conv1D + Bidirectional LSTM + Stacked RNN Architecture · 6 Emotion Classes
    </span>
</div>
""", unsafe_allow_html=True)
