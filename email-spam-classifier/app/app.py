import os
import pickle
import sys
import streamlit as st

# Add the ML code folder to Python's search path
SRC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src")
sys.path.append(os.path.abspath(SRC_DIR))

from preprocess import data_transformation


# ---------- Page config ----------
st.set_page_config(
    page_title="Spam Shield | Email Classifier",
    page_icon="🛡️",
    layout="centered",
)


# ---------- Custom CSS ----------
st.markdown("""
<style>
    .main {
        background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
    }
    .stApp {
        background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
    }
    .title-text {
        font-size: 2.6rem;
        font-weight: 800;
        text-align: center;
        background: linear-gradient(90deg, #38bdf8, #818cf8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0rem;
    }
    .subtitle-text {
        text-align: center;
        color: #94a3b8;
        font-size: 1.05rem;
        margin-bottom: 2rem;
    }
    .stTextArea textarea {
        background-color: #1e293b;
        color: #f1f5f9;
        border: 1px solid #334155;
        border-radius: 12px;
        font-size: 1rem;
    }
    div.stButton > button {
        width: 100%;
        border-radius: 10px;
        height: 3em;
        font-weight: 700;
        font-size: 1rem;
        background: linear-gradient(90deg, #38bdf8, #6366f1);
        color: white;
        border: none;
        transition: transform 0.15s ease;
    }
    div.stButton > button:hover {
        transform: scale(1.02);
        color: white;
        border: none;
    }
    .result-card {
        padding: 1.6rem;
        border-radius: 16px;
        text-align: center;
        margin-top: 1.5rem;
        animation: fadeIn 0.4s ease-in;
    }
    .spam-card {
        background: linear-gradient(135deg, #7f1d1d, #991b1b);
        border: 1px solid #ef4444;
    }
    .ham-card {
        background: linear-gradient(135deg, #14532d, #166534);
        border: 1px solid #22c55e;
    }
    .result-label {
        font-size: 1.8rem;
        font-weight: 800;
        color: white;
        margin: 0;
    }
    .result-sub {
        color: #e2e8f0;
        font-size: 0.95rem;
        margin-top: 0.3rem;
    }
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(6px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .example-chip {
        font-size: 0.85rem;
        color: #94a3b8;
    }
</style>
""", unsafe_allow_html=True)


# ---------- Load artifacts (cached so it only runs once) ----------
@st.cache_resource
def load_artifacts():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(BASE_DIR, "..", "models", "model.pkl")
    vectorizer_path = os.path.join(BASE_DIR, "..", "models", "vectorizer.pkl")

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    with open(vectorizer_path, "rb") as f:
        tfidf = pickle.load(f)

    return model, tfidf


def predict(text, model, tfidf):
    cleaned_text = data_transformation(text)
    vector = tfidf.transform([cleaned_text]).toarray()

    prediction = model.predict(vector)[0]

    # LinearSVC has no predict_proba, but decision_function gives a
    # signed distance from the separating hyperplane — useful as a
    # rough "confidence" signal for the UI.
    confidence_score = model.decision_function(vector)[0]

    return prediction, confidence_score


# ---------- Header ----------
st.markdown('<p class="title-text">🛡️ Spam Shield</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="subtitle-text">Paste an email or SMS below and let the model decide</p>',
    unsafe_allow_html=True,
)

try:
    model, tfidf = load_artifacts()
    artifacts_loaded = True
except FileNotFoundError:
    artifacts_loaded = False
    st.error(
        "Model artifacts not found. Run `python train.py` first to generate "
        "`models/model.pkl` and `models/vectorizer.pkl`."
    )

# ---------- Example buttons ----------
st.markdown("**Try an example:**")
col1, col2, col3 = st.columns(3)

example_text = None
with col1:
    if st.button("💰 Spam example"):
        example_text = "WINNER!! You have been selected to receive a $1000 Walmart gift card. Click here to claim now!"
with col2:
    if st.button("✉️ Ham example"):
        example_text = "Hey, are we still on for lunch tomorrow at 1pm?"
with col3:
    if st.button("🧾 Tricky example"):
        example_text = "Your Amazon order #4471 has shipped and will arrive Friday."

if "email_text" not in st.session_state:
    st.session_state.email_text = ""

if example_text:
    st.session_state.email_text = example_text

# ---------- Input area ----------
email_text = st.text_area(
    "Email / SMS text",
    value=st.session_state.email_text,
    height=180,
    placeholder="Paste the email or message text here...",
    label_visibility="collapsed",
    key="email_text",
)

# ---------- Predict button ----------
analyze_clicked = st.button("🔍 Analyze Message", disabled=not artifacts_loaded)

if analyze_clicked:
    if not email_text.strip():
        st.warning("Please enter some text to analyze.")
    else:
        with st.spinner("Analyzing message..."):
            prediction, score = predict(email_text, model, tfidf)

        if prediction == 1:
            st.markdown(f"""
            <div class="result-card spam-card">
                <p class="result-label">🚨 SPAM</p>
                <p class="result-sub">This message shows signs of spam/phishing content</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-card ham-card">
                <p class="result-label">✅ NOT SPAM</p>
                <p class="result-sub">This message looks legitimate</p>
            </div>
            """, unsafe_allow_html=True)

        with st.expander("See details"):
            st.write("**Cleaned/transformed text used for prediction:**")
            st.code(data_transformation(email_text))
            st.write(f"**Raw decision score:** `{score:.4f}`")
            st.caption(
                "Score is the distance from the model's decision boundary "
                "(negative → not spam, positive → spam). It is not a true "
                "probability since LinearSVC doesn't support predict_proba."
            )

# ---------- Footer ----------
st.markdown("---")
st.caption("Built with TF-IDF + LinearSVC · Trained on the SMS Spam Collection dataset")