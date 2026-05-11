import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import streamlit as st
from vectorize import vectorize
from recommend import recommend_movie

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CineMatch",
    page_icon="🎬",
    layout="centered"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=DM+Sans:wght@300;400;500&display=swap');

/* Background */
.stApp {
    background-color: #0a0a0a;
    background-image:
        radial-gradient(ellipse 80% 50% at 50% -10%, rgba(220, 38, 38, 0.15), transparent),
        radial-gradient(ellipse 60% 40% at 80% 80%, rgba(120, 20, 20, 0.1), transparent);
}

/* Hide streamlit branding */
#MainMenu, footer, header { visibility: hidden; }

/* Main container */
.block-container {
    padding-top: 3rem;
    max-width: 680px;
}

/* Hero title */
.hero-title {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 5rem;
    letter-spacing: 0.08em;
    line-height: 1;
    color: #ffffff;
    margin: 0;
}

.hero-accent {
    color: #dc2626;
}

.hero-sub {
    font-family: 'DM Sans', sans-serif;
    font-weight: 300;
    font-size: 0.95rem;
    color: #6b6b6b;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    margin-top: 0.5rem;
    margin-bottom: 2.5rem;
}

/* Divider */
.red-line {
    width: 48px;
    height: 3px;
    background: #dc2626;
    margin: 1rem 0 2rem 0;
}

/* Selectbox label */
.stSelectbox label {
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.75rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.2em !important;
    text-transform: uppercase !important;
    color: #6b6b6b !important;
}

/* Selectbox input */
.stSelectbox > div > div {
    background-color: #141414 !important;
    border: 1px solid #2a2a2a !important;
    border-radius: 4px !important;
    color: #ffffff !important;
    font-family: 'DM Sans', sans-serif !important;
    transition: border-color 0.2s;
}

.stSelectbox > div > div:hover {
    border-color: #dc2626 !important;
}

/* Button */
.stButton > button {
    background-color: #dc2626 !important;
    color: #ffffff !important;
    font-family: 'Bebas Neue', sans-serif !important;
    font-size: 1.1rem !important;
    letter-spacing: 0.15em !important;
    border: none !important;
    border-radius: 4px !important;
    padding: 0.6rem 2.5rem !important;
    width: 100% !important;
    margin-top: 0.5rem !important;
    transition: background-color 0.2s, transform 0.1s !important;
    cursor: pointer !important;
}

.stButton > button:hover {
    background-color: #b91c1c !important;
    transform: translateY(-1px) !important;
}

.stButton > button:active {
    transform: translateY(0px) !important;
}

/* Results section */
.results-label {
    font-family: 'DM Sans', sans-serif;
    font-size: 0.75rem;
    font-weight: 500;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: #6b6b6b;
    margin-top: 2.5rem;
    margin-bottom: 1rem;
}

/* Movie card */
.movie-card {
    display: flex;
    align-items: center;
    gap: 1rem;
    background: #141414;
    border: 1px solid #1f1f1f;
    border-radius: 4px;
    padding: 1rem 1.25rem;
    margin-bottom: 0.6rem;
    transition: border-color 0.2s, transform 0.15s;
    animation: slideIn 0.3s ease forwards;
    opacity: 0;
}

.movie-card:hover {
    border-color: #dc2626;
    transform: translateX(4px);
}

.movie-number {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 1.4rem;
    color: #dc2626;
    min-width: 28px;
    line-height: 1;
}

.movie-title {
    font-family: 'DM Sans', sans-serif;
    font-size: 1rem;
    font-weight: 400;
    color: #e5e5e5;
    letter-spacing: 0.01em;
}

/* Staggered animation */
.movie-card:nth-child(1) { animation-delay: 0.05s; }
.movie-card:nth-child(2) { animation-delay: 0.10s; }
.movie-card:nth-child(3) { animation-delay: 0.15s; }
.movie-card:nth-child(4) { animation-delay: 0.20s; }
.movie-card:nth-child(5) { animation-delay: 0.25s; }

@keyframes slideIn {
    from { opacity: 0; transform: translateX(-12px); }
    to   { opacity: 1; transform: translateX(0); }
}

/* Spinner color */
.stSpinner > div { border-top-color: #dc2626 !important; }
</style>
""", unsafe_allow_html=True)


# ── Load data (cached) ────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    return vectorize()

movies, similarity = load_data()


# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div>
    <p class="hero-title">CINE<span class="hero-accent">MATCH</span></p>
    <div class="red-line"></div>
    <p class="hero-sub">Discover your next favourite film</p>
</div>
""", unsafe_allow_html=True)


# ── Input ─────────────────────────────────────────────────────────────────────
option = st.selectbox("Select a movie", movies['title'].values, index=None, placeholder="Start typing a movie name...")

st.button("Find Similar Movies", key="recommend_btn")

# ── Results ───────────────────────────────────────────────────────────────────
if st.session_state.get("recommend_btn") and option:
    with st.spinner("Finding matches..."):
        recommendations = recommend_movie(option)

    st.markdown('<p class="results-label">Because you liked ' + option + '</p>', unsafe_allow_html=True)

    cards_html = ""
    for i, movie in enumerate(recommendations, 1):
        cards_html += f"""
        <div class="movie-card">
            <span class="movie-number">0{i}</span>
            <span class="movie-title">{movie}</span>
        </div>
        """
    st.markdown(cards_html, unsafe_allow_html=True)

elif st.session_state.get("recommend_btn") and not option:
    st.markdown('<p style="color:#dc2626; font-family: DM Sans; font-size:0.85rem;">Please select a movie first.</p>', unsafe_allow_html=True)