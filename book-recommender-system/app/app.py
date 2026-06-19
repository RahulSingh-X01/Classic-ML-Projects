import streamlit as st
import pickle
import numpy as np
import os

st.set_page_config(
    page_title="BookShelf",
    page_icon="📚",
    layout="wide"
)

# Minimal CSS — only style custom HTML cards, never touch Streamlit widgets
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@500;600&family=Inter:wght@400;500;600&display=swap');

.block-container { padding-top: 1.5rem !important; max-width: 1200px !important; }
#MainMenu, footer, header { visibility: hidden; }

.hero-title {
    font-family: 'Playfair Display', serif;
    font-size: 2.4rem;
    font-weight: 600;
    color: #ffffff;
    margin: 0 0 0.3rem;
}
.hero-sub { font-size: 1rem; color: #6b7280; margin: 0 0 1rem; }

.sec-title {
    font-family: 'Playfair Display', serif;
    font-size: 1.35rem;
    font-weight: 500;
    color: #ffffff;
    margin: 0 0 0.15rem;
}
.sec-sub { font-size: 0.88rem; color: #6b7280; margin: 0 0 1.2rem; }

.bcard {
    background: #ffffff;
    border: 1px solid #e5e7eb;
    border-radius: 12px;
    padding: 0.85rem;
    text-align: center;
}
.bcard img {
    width: 100%; height: 155px;
    object-fit: cover;
    border-radius: 6px;
    margin-bottom: 0.55rem;
}
.bcard-title {
    font-size: 0.78rem; font-weight: 600;
    color: #111827; line-height: 1.3;
    margin: 0 0 0.18rem;
    display: -webkit-box;
    -webkit-line-clamp: 2;
    -webkit-box-orient: vertical;
    overflow: hidden;
}
.bcard-author { font-size: 0.7rem; color: #6b7280; margin: 0 0 0.35rem; }
.badge-amber {
    display: inline-block;
    background: #fef3c7; color: #92400e;
    font-size: 0.66rem; font-weight: 600;
    padding: 2px 7px; border-radius: 20px;
}

.rcard {
    background: #ffffff;
    border: 1px solid #e5e7eb;
    border-radius: 12px;
    padding: 1rem 1.15rem;
    display: flex; align-items: flex-start; gap: 0.9rem;
    margin-bottom: 0.75rem;
}
.rcard img {
    width: 64px; height: 90px;
    object-fit: cover; border-radius: 6px;
    flex-shrink: 0;
}
.rcard-num {
    font-family: 'Playfair Display', serif;
    font-size: 1.5rem; color: #d1d5db;
    font-weight: 600; min-width: 26px;
    line-height: 1; padding-top: 3px;
}
.rcard-title { font-size: 0.92rem; font-weight: 600; color: #111827; margin: 0 0 0.18rem; line-height: 1.3; }
.rcard-author { font-size: 0.8rem; color: #6b7280; margin: 0 0 0.45rem; }
.badge-indigo {
    display: inline-block;
    background: #ede9fe; color: #4c1d95;
    font-size: 0.66rem; font-weight: 600;
    padding: 2px 7px; border-radius: 20px;
}

.empty-box { text-align: center; padding: 2.5rem 1rem; color: #9ca3af; }
.empty-box .icon { font-size: 2.2rem; margin-bottom: 0.5rem; }
.empty-box p { color: #9ca3af; font-size: 0.9rem; }

hr.divider { border: none; border-top: 1px solid #e5e7eb; margin: 2rem 0 1rem; }
</style>
""", unsafe_allow_html=True)


# ── Load artifacts ────────────────────────────────────────────────────────
BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
ARTIFACTS_DIR = os.path.join(BASE_DIR, '..', 'artifacts')

@st.cache_resource
def load_artifacts():
    popular_df = pickle.load(open(os.path.join(ARTIFACTS_DIR, "popular.pkl"),    "rb"))
    pivot      = pickle.load(open(os.path.join(ARTIFACTS_DIR, "pivot.pkl"),      "rb"))
    similarity = pickle.load(open(os.path.join(ARTIFACTS_DIR, "similarity.pkl"), "rb"))
    books      = pickle.load(open(os.path.join(ARTIFACTS_DIR, "books.pkl"),      "rb"))
    return popular_df, pivot, similarity, books

popular_df, pivot, similarity, books = load_artifacts()


def recommend_book(book_name):
    if book_name not in pivot.index:
        return []
    index   = np.where(pivot.index == book_name)[0][0]
    similar = sorted(
        list(enumerate(similarity[index])), key=lambda x: x[1], reverse=True
    )[1:6]
    result = []
    for i in similar:
        title     = pivot.index[i[0]]
        book_info = books[books['Book-Title'] == title]
        if book_info.empty:
            continue
        row = book_info.iloc[0]
        result.append({
            'title' : row['Book-Title'],
            'author': row['Book-Author'],
            'image' : row['Image-URL-M'],
            'score' : round(i[1], 3)
        })
    return result


# ── Hero ─────────────────────────────────────────────────────────────────
st.markdown("""
<p class="hero-title">📚 BookShelf</p>
<p class="hero-sub">Discover books you'll love — powered by readers like you.</p>
""", unsafe_allow_html=True)

tab1, tab2 = st.tabs(["🔍  Find Similar Books", "🔥  Trending Books"])


# ══════════════════════════════════════════════════════════════
# TAB 1 — Recommendation Engine
# ══════════════════════════════════════════════════════════════
with tab1:
    st.markdown("<br>", unsafe_allow_html=True)

    # Label rendered by Streamlit natively — inherits light theme color (#111827)
    col_sel, col_btn = st.columns([4, 1], vertical_alignment="bottom")

    with col_sel:
        book_list     = sorted(pivot.index.tolist())
        selected_book = st.selectbox(
            "Select a book from the dataset",
            options=["— choose a book —"] + book_list,
        )

    with col_btn:
        clicked = st.button("Find Similar →", use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)

    if clicked:
        if selected_book == "— choose a book —":
            st.warning("Please select a book first.")
        else:
            recs = recommend_book(selected_book)
            if not recs:
                st.markdown("""
                <div class="empty-box">
                    <div class="icon">🔍</div>
                    <p>No recommendations found for this book.</p>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(
                    f'<p class="sec-title">Because you liked <em>{selected_book}</em></p>'
                    '<p class="sec-sub">5 books readers also loved</p>',
                    unsafe_allow_html=True
                )
                col_a, col_b = st.columns(2)
                fb = "https://via.placeholder.com/64x90?text=?"
                for idx, book in enumerate(recs):
                    html = f"""
                    <div class="rcard">
                        <span class="rcard-num">0{idx+1}</span>
                        <img src="{book['image']}" onerror="this.src='{fb}'" />
                        <div>
                            <p class="rcard-title">{book['title']}</p>
                            <p class="rcard-author">{book['author']}</p>
                            <span class="badge-indigo">Match {int(book['score']*100)}%</span>
                        </div>
                    </div>"""
                    (col_a if idx % 2 == 0 else col_b).markdown(html, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="empty-box">
            <div class="icon">📖</div>
            <p>Select a book above and click <b>Find Similar →</b> to get recommendations.</p>
        </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# TAB 2 — Trending Books
# ══════════════════════════════════════════════════════════════
with tab2:
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        '<p class="sec-title">Trending this season</p>'
        '<p class="sec-sub">Top 50 highest-rated books with 250+ reader reviews</p>',
        unsafe_allow_html=True
    )

    COLS = 6
    fb   = "https://via.placeholder.com/120x155?text=?"
    for i in range(0, len(popular_df), COLS):
        row_df = popular_df.iloc[i:i+COLS]
        cols   = st.columns(COLS)
        for col, (_, book) in zip(cols, row_df.iterrows()):
            col.markdown(f"""
            <div class="bcard">
                <img src="{book['Image-URL-M']}" onerror="this.src='{fb}'" />
                <p class="bcard-title">{book['Book-Title']}</p>
                <p class="bcard-author">{book['Book-Author']}</p>
                <span class="badge-amber">⭐ {round(book['avg-rating'],1)} · {int(book['num-rating']):,}</span>
            </div>""", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)


# ── Footer ────────────────────────────────────────────────────────────────
st.markdown("""
<hr class="divider">
<p style="text-align:center; color:#9ca3af; font-size:0.78rem; padding-bottom:1rem;">
    Built with collaborative filtering · Book-Crossing Dataset
</p>
""", unsafe_allow_html=True)