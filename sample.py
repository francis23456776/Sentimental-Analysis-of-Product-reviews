import streamlit as st
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import plotly.express as px

nltk.download('stopwords', quiet=True)
nltk.download('vader_lexicon', quiet=True)

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Sentiment Predictor",
    page_icon="⚡",
    layout="centered"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;800&family=Plus+Jakarta+Sans:wght@300;400;600&display=swap');

/* Main background */
.stApp {
    background: #05070a;
    color: #ffffff;
    font-family: 'Plus Jakarta Sans', sans-serif;
}

/* Main heading */
.main-title {
    font-family: 'Orbitron', sans-serif;
    font-size: 2.8rem;
    font-weight: 800;
    text-align: center;
    background: linear-gradient(90deg, #00f2fe, #7062f5, #fc00ff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 5px;
}

/* Subtitle */
.sub {
    text-align: center;
    color: #a0a5b5;
    font-family: 'Orbitron', sans-serif;
    font-size: 0.85rem;
    margin-bottom: 30px;
}

/* Neon divider */
.neon-divider {
    height: 2px;
    background: linear-gradient(
        90deg,
        transparent,
        #7062f5,
        #00f2fe,
        transparent
    );
    margin: 30px 0;
}

/* Sentiment result box */
.res-box {
    padding: 25px;
    border-radius: 15px;
    text-align: center;
    font-family: 'Orbitron', sans-serif;
    font-size: 1.6rem;
    border-left: 8px solid;
    margin-top: 15px;
}

/* Badges */
.badge {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 0.75rem;
    font-family: 'Orbitron', sans-serif;
    background: #1a1a2e;
    border: 1px solid #7062f5;
    color: #00f2fe;
    margin-bottom: 10px;
}

/* Hide sidebar */
section[data-testid="stSidebar"] {
    display: none;
}

/* Text input and text area */
.stApp input,
.stApp textarea {
    background-color: #151827 !important;
    color: #ffffff !important;
    -webkit-text-fill-color: #ffffff !important;
    border: 1px solid #7062f5 !important;
    border-radius: 8px !important;
    caret-color: #00f2fe !important;
}

/* Placeholder text */
.stApp input::placeholder,
.stApp textarea::placeholder {
    color: #a0a5b5 !important;
    -webkit-text-fill-color: #a0a5b5 !important;
    opacity: 1 !important;
}

/* Select boxes */
.stApp [data-baseweb="select"] > div {
    background-color: #151827 !important;
    color: #ffffff !important;
    border-color: #7062f5 !important;
}

/* Dropdown text */
.stApp [role="option"] {
    background-color: #151827 !important;
    color: #ffffff !important;
}

/* Labels and general text */
.stApp label,
.stApp [data-testid="stWidgetLabel"],
.stApp [data-testid="stWidgetLabel"] p {
    color: #ffffff !important;
}

/* File uploader */
.stApp [data-testid="stFileUploader"] {
    color: #ffffff !important;
}

.stApp [data-testid="stFileUploaderDropzone"] {
    background-color: #151827 !important;
    border: 1px dashed #7062f5 !important;
}

.stApp [data-testid="stFileUploaderDropzone"] * {
    color: #ffffff !important;
}

/* Buttons */
.stApp .stButton > button,
.stApp .stDownloadButton > button {
    background-color: #111426 !important;
    color: #ffffff !important;
    border: 1px solid #7062f5 !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    transition: all 0.2s ease;
}

/* Button hover */
.stApp .stButton > button:hover,
.stApp .stDownloadButton > button:hover {
    border-color: #00f2fe !important;
    color: #00f2fe !important;
    box-shadow: 0 0 12px rgba(0, 242, 254, 0.2);
}

/* Metrics */
[data-testid="stMetric"] {
    background-color: #101321;
    padding: 15px;
    border: 1px solid #252943;
    border-radius: 12px;
}

[data-testid="stMetricLabel"],
[data-testid="stMetricValue"] {
    color: #ffffff !important;
}

/* Dataframes and tables */
.stApp [data-testid="stDataFrame"] {
    border: 1px solid #252943;
    border-radius: 8px;
}

/* Captions */
.stApp .stCaption,
.stApp [data-testid="stCaptionContainer"] {
    color: #a0a5b5 !important;
}

/* Responsive heading for smaller screens */
@media (max-width: 600px) {
    .main-title {
        font-size: 1.8rem;
    }

    .sub {
        font-size: 0.7rem;
    }
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">SENTIMENT PREDICTOR</div>', unsafe_allow_html=True)
st.markdown('<div class="sub">⚡ Type anything · Predict sentiment instantly · Upload CSV for custom model</div>', unsafe_allow_html=True)
st.markdown('<div class="neon-divider"></div>', unsafe_allow_html=True)

# --- SECTION 1: INSTANT TEXT PREDICTION ---
st.markdown("### � Predict Sentiment")

# Model selector
use_custom = 'model' in st.session_state
model_label = "🟢 Custom Model (CSV)" if use_custom else "🔵 Built-in Model (VADER)"
st.markdown(f'<span class="badge">Active: {model_label}</span>', unsafe_allow_html=True)

input_text = st.text_area("Type or paste any text:", placeholder="e.g. I love this product! It's absolutely amazing.", height=130)

if st.button("⚡ PREDICT SENTIMENT", use_container_width=True):
    if input_text.strip():
        if use_custom:
            # Use custom trained model
            tfidf = st.session_state['tfidf']
            model = st.session_state['model']
            vec = tfidf.transform([input_text.lower()])
            result = model.predict(vec)[0]
            confidence = np.max(model.predict_proba(vec)) * 100
        else:
            # Use VADER (works instantly, no training needed)
            sia = SentimentIntensityAnalyzer()
            scores = sia.polarity_scores(input_text)
            compound = scores['compound']
            if compound >= 0.05:
                result, confidence = "Positive", round((compound + 1) / 2 * 100, 1)
            elif compound <= -0.05:
                result, confidence = "Negative", round((1 - (compound + 1) / 2) * 100, 1)
            else:
                result, confidence = "Neutral", round((1 - abs(compound)) * 100, 1)

        color_map = {
            "Positive": "#00ff7f",
            "Negative": "#ff4b4b",
            "Neutral":  "#ffd300",
            "Irrelevant": "#808080"
        }
        color = color_map.get(result, "#7062f5")

        st.markdown(
            f"<div class='res-box' style='border-color:{color}; background:rgba(255,255,255,0.03); color:{color};'>"
            f"{result.upper()}"
            f"<br><span style='font-size:0.85rem; color:white;'>Confidence: {confidence:.1f}%</span>"
            f"</div>",
            unsafe_allow_html=True
        )
    else:
        st.warning("Please enter some text first.")

# --- SECTION 2: BULK CSV ANALYSIS ---
st.markdown('<div class="neon-divider"></div>', unsafe_allow_html=True)
st.markdown("### Bulk Analysis via CSV")
st.caption("Upload a CSV file with reviews or product titles to see sentiment distribution.")

bulk_file = st.file_uploader("Upload CSV for Bulk Analysis", type=["csv"], key="bulk_upload")

if bulk_file:
    df_bulk = pd.read_csv(bulk_file)
    st.write("### Preview of uploaded data")
    st.dataframe(df_bulk.head(), use_container_width=True)

    # Column selection
    text_col = st.selectbox("Select the column containing text for analysis:", df_bulk.columns)

    if st.button(" RUN BULK ANALYSIS", use_container_width=True):
        with st.spinner("Processing..."):
            # Sentiment logic helper
            def get_sentiment(text):
                text_str = str(text).lower()
                if use_custom:
                    tfidf = st.session_state['tfidf']
                    model = st.session_state['model']
                    vec = tfidf.transform([text_str])
                    return model.predict(vec)[0]
                else:
                    sia = SentimentIntensityAnalyzer()
                    scores = sia.polarity_scores(text_str)
                    compound = scores['compound']
                    if compound >= 0.05: return "Positive"
                    elif compound <= -0.05: return "Negative"
                    else: return "Neutral"

            df_bulk['Sentiment'] = df_bulk[text_col].apply(get_sentiment)

            # Results
            st.success("Analysis Complete!")

            col1, col2 = st.columns([1, 2])

            with col1:
                st.write("####  Summary")
                counts = df_bulk['Sentiment'].value_counts().reset_index()
                counts.columns = ['Sentiment', 'Count']
                st.dataframe(counts, hide_index=True)

            with col2:
                st.write("####  Distribution")
                fig = px.pie(counts, values='Count', names='Sentiment',
                             color='Sentiment',
                             color_discrete_map={
                                 "Positive": "#00ff7f",
                                 "Negative": "#ff4b4b",
                                 "Neutral": "#ffd300",
                                 "Irrelevant": "#808080"
                             },
                             hole=0.4)
                fig.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font_color='white',
                    margin=dict(t=0, b=0, l=0, r=0),
                    height=250
                )
                st.plotly_chart(fig, use_container_width=True)

            st.write("###  Detailed Results")
            st.dataframe(df_bulk, use_container_width=True)

            # Download button
            csv = df_bulk.to_csv(index=False).encode('utf-8')
            st.download_button(
                " Download Results as CSV",
                csv,
                "sentiment_analysis_results.csv",
                "text/csv",
                key='download-csv',
                use_container_width=True
            )

# --- SECTION 3: OPTIONAL CSV UPLOAD (TRAINING) ---
st.markdown('<div class="neon-divider"></div>', unsafe_allow_html=True)
st.markdown("### 📂 (Optional) Train Custom Model with Your CSV")
st.caption("CSV must have columns: **id, entity, sentiment, text** — Overrides the built-in VADER model")

uploaded_file = st.file_uploader("Upload CSV", type=["csv"], label_visibility="collapsed")

if uploaded_file:
    with st.spinner(" Training custom model..."):
        try:
            cols = ['id', 'entity', 'sentiment', 'text']
            df = pd.read_csv(uploaded_file, names=cols).dropna()

            if len(df) < 10:
                st.error(" Dataset too small. Need at least 10 rows.")
            else:
                stop = set(stopwords.words('english'))
                df['cleaned'] = df['text'].apply(
                    lambda t: " ".join([w for w in re.sub(r'[^a-z\s]', '', str(t).lower()).split() if w not in stop])
                )
                sample_size = min(5000, len(df))
                df_s = df.sample(sample_size, random_state=42)

                tfidf = TfidfVectorizer(max_features=2000)
                X = tfidf.fit_transform(df_s['cleaned'])
                rf = RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42)
                rf.fit(X, df_s['sentiment'])

                st.session_state['model'] = rf
                st.session_state['tfidf'] = tfidf
                st.success(f" Custom model trained on **{sample_size}** records! Labels: {list(df['sentiment'].unique())}")
                st.rerun()

        except Exception as e:
            st.error(f" Error: {e}")

if use_custom:
    if st.button(" Remove Custom Model (revert to VADER)"):
        del st.session_state['model']
        del st.session_state['tfidf']
        st.rerun()

st.markdown('<div class="neon-divider"></div>', unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:#7062f5; font-family:Orbitron; font-size:0.75rem;'>● SENTIMENT ENGINE ONLINE ●</p>", unsafe_allow_html=True)