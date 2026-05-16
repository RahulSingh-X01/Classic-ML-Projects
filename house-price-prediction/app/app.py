import streamlit as st
import pandas as pd
import numpy as np
import joblib
import sys
import os

# ── Resolve project root & inject src/ into path ─────────────────────────────
# app.py lives in:  house-price-prediction/app/app.py
# models live in:   house-price-prediction/models/
# src scripts in:   house-price-prediction/src/
APP_DIR     = os.path.dirname(os.path.abspath(__file__))          # .../app/
PROJECT_DIR = os.path.dirname(APP_DIR)                            # .../house-price-prediction/
MODELS_DIR  = os.path.join(PROJECT_DIR, "models")
SRC_DIR     = os.path.join(PROJECT_DIR, "src")

if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Bengaluru House Price Predictor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;700;900&family=DM+Sans:wght@300;400;500;600&display=swap');

/* ── Reset & base ── */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

.stApp {
    background: #0d0d0d;
    color: #f0ece4;
}

/* Hide Streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 0 !important; max-width: 100% !important; }

/* ── Hero banner ── */
.hero {
    background: linear-gradient(135deg, #0d0d0d 0%, #1a1208 50%, #0d0d0d 100%);
    border-bottom: 1px solid #2a2010;
    padding: 52px 64px 40px;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 340px; height: 340px;
    background: radial-gradient(circle, rgba(212,175,55,0.12) 0%, transparent 70%);
    border-radius: 50%;
}
.hero::after {
    content: '';
    position: absolute;
    bottom: -80px; left: 10%;
    width: 220px; height: 220px;
    background: radial-gradient(circle, rgba(212,175,55,0.07) 0%, transparent 70%);
    border-radius: 50%;
}
.hero-label {
    font-family: 'DM Sans', sans-serif;
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 4px;
    color: #d4af37;
    text-transform: uppercase;
    margin-bottom: 14px;
}
.hero-title {
    font-family: 'Playfair Display', serif;
    font-size: clamp(36px, 5vw, 64px);
    font-weight: 900;
    line-height: 1.05;
    color: #f0ece4;
    margin: 0 0 16px;
}
.hero-title span { color: #d4af37; }
.hero-sub {
    font-size: 15px;
    color: #8a8070;
    font-weight: 300;
    max-width: 480px;
    line-height: 1.6;
}
.hero-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(212,175,55,0.1);
    border: 1px solid rgba(212,175,55,0.25);
    color: #d4af37;
    padding: 5px 14px;
    border-radius: 20px;
    font-size: 12px;
    font-weight: 500;
    margin-top: 20px;
}

/* ── Main layout ── */
.main-layout {
    display: grid;
    grid-template-columns: 1fr 420px;
    gap: 0;
    min-height: calc(100vh - 220px);
}

/* ── Form panel ── */
.form-panel {
    padding: 48px 64px;
    border-right: 1px solid #1e1a12;
}

/* ── Result panel ── */
.result-panel {
    background: #0a0a0a;
    padding: 48px 40px;
    position: sticky;
    top: 0;
    min-height: 100%;
}

/* ── Section labels ── */
.section-label {
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 3.5px;
    color: #d4af37;
    text-transform: uppercase;
    margin-bottom: 28px;
    display: flex;
    align-items: center;
    gap: 10px;
}
.section-label::after {
    content: '';
    flex: 1;
    height: 1px;
    background: linear-gradient(to right, #2a2010, transparent);
}

/* ── Streamlit widget overrides ── */
.stSelectbox > div > div,
.stNumberInput > div > div > input {
    background: #161209 !important;
    border: 1px solid #2a2010 !important;
    border-radius: 8px !important;
    color: #f0ece4 !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 14px !important;
    transition: border-color 0.2s !important;
}
.stSelectbox > div > div:hover,
.stNumberInput > div > div > input:focus {
    border-color: #d4af37 !important;
    box-shadow: 0 0 0 2px rgba(212,175,55,0.08) !important;
}

/* Dropdown arrow color */
.stSelectbox svg { color: #d4af37 !important; }

/* Labels */
.stSelectbox label,
.stNumberInput label,
.stSlider label {
    font-size: 12px !important;
    font-weight: 600 !important;
    letter-spacing: 0.5px !important;
    color: #8a8070 !important;
    text-transform: uppercase !important;
    margin-bottom: 6px !important;
}

/* Slider */
.stSlider > div > div > div > div {
    background: #d4af37 !important;
}
.stSlider > div > div > div {
    background: #2a2010 !important;
}

/* ── Predict button ── */
.stButton > button {
    width: 100%;
    background: linear-gradient(135deg, #d4af37 0%, #b8942a 100%) !important;
    color: #0d0d0d !important;
    border: none !important;
    padding: 16px 32px !important;
    border-radius: 10px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 14px !important;
    font-weight: 700 !important;
    letter-spacing: 1.5px !important;
    text-transform: uppercase !important;
    cursor: pointer !important;
    transition: all 0.2s !important;
    margin-top: 8px !important;
    box-shadow: 0 4px 24px rgba(212,175,55,0.2) !important;
}
.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 8px 32px rgba(212,175,55,0.35) !important;
}
.stButton > button:active {
    transform: translateY(0) !important;
}

/* ── Result card ── */
.result-idle {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    height: 100%;
    text-align: center;
    padding: 40px 0;
}
.result-icon {
    font-size: 56px;
    margin-bottom: 20px;
    opacity: 0.4;
}
.result-idle-text {
    color: #3a3020;
    font-size: 14px;
    font-weight: 400;
    line-height: 1.7;
    max-width: 260px;
}

.price-card {
    background: linear-gradient(145deg, #1a1208, #120e04);
    border: 1px solid #2a2010;
    border-radius: 16px;
    padding: 36px 32px;
    margin-bottom: 20px;
    text-align: center;
    position: relative;
    overflow: hidden;
}
.price-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(to right, transparent, #d4af37, transparent);
}
.price-tag {
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 3px;
    color: #d4af37;
    text-transform: uppercase;
    margin-bottom: 12px;
}
.price-value {
    font-family: 'Playfair Display', serif;
    font-size: clamp(32px, 4vw, 48px);
    font-weight: 900;
    color: #f0ece4;
    line-height: 1.1;
}
.price-value span {
    color: #d4af37;
}
.price-unit {
    font-size: 13px;
    color: #5a5040;
    margin-top: 8px;
    font-weight: 400;
}

/* ── Stat pills ── */
.stat-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 10px;
    margin-top: 16px;
}
.stat-pill {
    background: #0f0c06;
    border: 1px solid #1e1a12;
    border-radius: 10px;
    padding: 14px 16px;
}
.stat-pill-label {
    font-size: 10px;
    color: #5a5040;
    font-weight: 600;
    letter-spacing: 1px;
    text-transform: uppercase;
    margin-bottom: 4px;
}
.stat-pill-value {
    font-size: 16px;
    font-weight: 700;
    color: #f0ece4;
}

/* ── Feature grid (form) ── */
.feature-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 0 32px;
}

/* ── Divider ── */
.gold-divider {
    height: 1px;
    background: linear-gradient(to right, transparent, #2a2010, transparent);
    margin: 32px 0;
}

/* ── Tooltip hint ── */
.hint {
    font-size: 12px;
    color: #3a3020;
    margin-top: -8px;
    margin-bottom: 16px;
    font-style: italic;
}

/* ── Market insight card ── */
.insight-card {
    background: #0f0c06;
    border: 1px solid #1e1a12;
    border-radius: 12px;
    padding: 20px;
    margin-top: 16px;
}
.insight-title {
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 2.5px;
    color: #5a5040;
    text-transform: uppercase;
    margin-bottom: 14px;
}
.insight-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 8px 0;
    border-bottom: 1px solid #1a1508;
}
.insight-row:last-child { border-bottom: none; }
.insight-key { font-size: 12px; color: #6a6050; }
.insight-val { font-size: 13px; font-weight: 600; color: #c0a830; }
</style>
""", unsafe_allow_html=True)


# ── Load artifacts ────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_artifacts():
    model            = joblib.load(os.path.join(MODELS_DIR, "xgb_model.pkl"))
    ohe              = joblib.load(os.path.join(MODELS_DIR, "ohe.pkl"))
    location_encoder = joblib.load(os.path.join(MODELS_DIR, "location_encoder.pkl"))
    num_imputer      = joblib.load(os.path.join(MODELS_DIR, "num_imputer.pkl"))
    cat_imputer      = joblib.load(os.path.join(MODELS_DIR, "cat_imputer.pkl"))
    loc_ppsf         = joblib.load(os.path.join(MODELS_DIR, "loc_ppsf.pkl"))
    return model, ohe, location_encoder, num_imputer, cat_imputer, loc_ppsf

try:
    model, ohe, location_encoder, num_imputer, cat_imputer, loc_ppsf = load_artifacts()
    artifacts_ok = True
except Exception as e:
    artifacts_ok = False
    load_error = str(e)


# ── Prediction helper ─────────────────────────────────────────────────────────
from preprocess import preprocess_data
from features import feature_engineering

def predict(area_type, availability, location, size, total_sqft, bath, balcony):
    df = pd.DataFrame([{
        'area_type':   area_type,
        'availability': availability,
        'location':    location,
        'size':        size,
        'total_sqft':  total_sqft,
        'bath':        bath,
        'balcony':     balcony,
    }])
    df = preprocess_data(df, num_imputer=num_imputer, cat_imputer=cat_imputer, fit=False)
    df = feature_engineering(df, fit=False,
                             location_encoder=location_encoder,
                             ohe=ohe, loc_ppsf=loc_ppsf)
    y_pred = np.expm1(model.predict(df))
    return round(float(y_pred[0]), 2)


# ── Data for dropdowns ────────────────────────────────────────────────────────
AREA_TYPES = [
    "Super built-up  Area",
    "Built-up  Area",
    "Plot  Area",
    "Carpet  Area",
]

SIZES = [f"{i} BHK" for i in range(1, 11)] + [f"{i} Bedroom" for i in range(1, 6)]

LOCATIONS = sorted([
    "1st Block Jayanagar", "1st Phase JP Nagar", "2nd Phase Judicial Layout",
    "2nd Stage Nagarbhavi", "5th Block Hbr Layout", "5th Phase JP Nagar",
    "6th Phase JP Nagar", "7th Phase JP Nagar", "8th Phase JP Nagar",
    "9th Phase JP Nagar", "AECS Layout", "Abbigere", "Akshaya Nagar",
    "Ambedkar Nagar", "Amruthahalli", "Anandapura", "Ananth Nagar",
    "Anekal", "Anjanapura", "Ardendale", "Arekere", "Attibele",
    "BEML Layout", "BTM 2nd Stage", "BTM Layout", "Banaswadi",
    "Bannerghatta", "Bannerghatta Road", "Basavangudi", "Basaveshwara Nagar",
    "Battarahalli", "Begur", "Begur Road", "Bellandur", "Benson Town",
    "Bharathi Nagar", "Bhoganhalli", "Billekahalli", "Binny Pete",
    "Bisuvanahalli", "Bommanahalli", "Bommasandra", "Bommasandra Industrial Area",
    "Bommasandra Jigani Link Road", "Brookefield", "Budigere",
    "CV Raman Nagar", "Chamrajpet", "Chandapura", "Channasandra",
    "Chikka Tirupathi", "Chikkabanavar", "Chikkalasandra", "Choodasandra",
    "Cooke Town", "Cox Town", "Cunningham Road", "Dasanapura",
    "Dasarahalli", "Devanahalli", "Devarachikkanahalli", "Dodda Nekkundi",
    "Doddaballapur", "Doddakallasandra", "Doddathoguru", "Domlur",
    "Dommasandra", "EPIP Zone", "Electronic City", "Electronic City Phase II",
    "Electronics City Phase 1", "Elephanta Nagar", "Frazer Town",
    "GM Palaya", "Garudachar Palya", "Gollarapalya Hosahalli",
    "Gottigere", "Green Glen Layout", "Gubbalala", "Gunjur",
    "HAL 2nd Stage", "HBR Layout", "HRBR Layout", "HSR Layout",
    "Haralur Road", "Harlur", "Hebbal", "Hebbal Kempapura",
    "Hegde Nagar", "Hennur", "Hennur Road", "Hoodi",
    "Horamavu Agara", "Horamavu Banaswadi", "Hormavu", "Hosa Road",
    "Hoysala Nagar", "Hulimavu", "ISRO Layout", "ITPL",
    "Iblur Village", "Indira Nagar", "JP Nagar", "Jakkur",
    "Jalahalli", "Jalahalli East", "Jigani", "Judicial Layout",
    "KR Puram", "Kadubeesanahalli", "Kadugodi", "Kaggadasapura",
    "Kaggalipura", "Kaikondrahalli", "Kalena Agrahara", "Kalyan Nagar",
    "Kambipura", "Kammanahalli", "Kammasandra", "Kanakapura",
    "Kanakpura Road", "Kannamangala", "Karchkanahalli", "Kasavanhalli",
    "Kasturi Nagar", "Kathari Pura", "Kenchenahalli", "Kereguddadahalli",
    "Kodichikkanahalli", "Kodihalli", "Kogilu", "Koramangala",
    "Kothannur", "Kothanur", "Kudlu", "Kudlu Gate",
    "Kumaraswami Layout", "Kundalahalli", "LB Shastri Nagar",
    "Laggere", "Lakshminarayana Pura", "Lingadheeranahalli",
    "Magadi Road", "Mahadevpura", "Mahalakshmi Layout", "Mallasandra",
    "Malleshpalya", "Malleshwaram", "Marathahalli", "Margondanahalli",
    "Marsur", "Mico Layout", "Munnekollal", "Murugeshpalya",
    "Mysore Road", "NGR Layout", "NRI Layout", "Nagarbhavi",
    "Nagasandra", "Nagavara", "Nagavarapalya", "Narayanapura",
    "Neeladri Nagar", "Nehru Nagar", "Old Airport Road",
    "Old Madras Road", "Padmanabhanagar", "Pai Layout",
    "Panathur", "Parappana Agrahara", "Pattandur Agrahara",
    "Poorna Pragna Layout", "Prithvi Layout", "R.T. Nagar",
    "RMV 2nd Stage", "Raja Rajeshwari Nagar", "Rajaji Nagar",
    "Rajiv Nagar", "Ramagondanahalli", "Ramamurthy Nagar",
    "Rayasandra", "Sahakara Nagar", "Sahakar Nagar", "Sanjay Nagar",
    "Sarakki Nagar", "Sarjapur", "Sarjapur  Road", "Sarjapura - Attibele Road",
    "Sector 2 HSR Layout", "Sector 7 HSR Layout", "Seegehalli",
    "Shampura", "Shivaji Nagar", "Singasandra", "Somasundara Palya",
    "Sompura", "Sonnenahalli", "Subramanyapura", "Sultan Palaya",
    "TC Palaya", "Talaghattapura", "Thanisandra", "Thigalarapalya",
    "Thubarahalli", "Tindlu", "Tumkur Road", "Ulsoor",
    "Uttarahalli", "Varthur", "Varthur Road", "Venkatapura",
    "Vidyaranyapura", "Vijayanagar", "Vishveshwarya Layout",
    "Vishwapriya Layout", "Vittasandra", "Whitefield",
    "Yelachenahalli", "Yelahanka", "Yelahanka New Town",
    "Yelenahalli", "Yeshwanthpur", "other",
])

AVAILABILITY = ["Ready To Move", "Immediate Possession"] + \
               [f"Dec {y}" for y in [2025, 2026, 2027]] + \
               [f"Jun {y}" for y in [2025, 2026]] + \
               ["Oct 2025", "Mar 2026"]


# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <div class="hero-label">Real Estate Intelligence · Bengaluru</div>
  <h1 class="hero-title">House Price<br><span>Predictor</span></h1>
  <p class="hero-sub">Powered by XGBoost trained on Bengaluru's residential market. Configure property details to get an instant valuation.</p>
  <div class="hero-badge">🤖 XGBoost Model &nbsp;·&nbsp; Bengaluru Market</div>
</div>
""", unsafe_allow_html=True)


# ── Model load warning ────────────────────────────────────────────────────────
if not artifacts_ok:
    st.error(f"⚠️ Could not load model artifacts. Make sure the `models/` folder exists.\n\n`{load_error}`")
    st.stop()


# ── Two-column layout ─────────────────────────────────────────────────────────
col_form, col_result = st.columns([3, 2], gap="large")

with col_form:
    st.markdown('<div class="form-panel">', unsafe_allow_html=True)

    # ── Property Identity ──
    st.markdown('<div class="section-label">01 &nbsp; Property Identity</div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        area_type = st.selectbox("Area Type", AREA_TYPES, index=0)
    with c2:
        location = st.selectbox("Location", LOCATIONS,
                                index=LOCATIONS.index("Whitefield") if "Whitefield" in LOCATIONS else 0)

    c3, c4 = st.columns(2)
    with c3:
        size = st.selectbox("Size (BHK)", SIZES, index=1)
    with c4:
        availability = st.selectbox("Availability", AVAILABILITY, index=0)

    st.markdown('<div class="gold-divider"></div>', unsafe_allow_html=True)

    # ── Property Dimensions ──
    st.markdown('<div class="section-label">02 &nbsp; Dimensions & Layout</div>', unsafe_allow_html=True)

    total_sqft = st.slider(
        "Total Square Footage",
        min_value=300, max_value=10000,
        value=1200, step=50,
        help="Total built-up area in square feet"
    )

    c5, c6 = st.columns(2)
    with c5:
        bath = st.number_input("Bathrooms", min_value=1, max_value=10, value=2, step=1)
    with c6:
        balcony = st.number_input("Balconies", min_value=0, max_value=5, value=1, step=1)

    # Live metrics
    try:
        size_num = int(size.split()[0])
    except:
        size_num = 2
    sqft_per_bhk = total_sqft / size_num if size_num else total_sqft

    st.markdown(f"""
    <div class="insight-card">
      <div class="insight-title">Property Summary</div>
      <div class="insight-row">
        <span class="insight-key">Sqft / BHK</span>
        <span class="insight-val">{sqft_per_bhk:,.0f} sq.ft</span>
      </div>
      <div class="insight-row">
        <span class="insight-key">Bath / BHK ratio</span>
        <span class="insight-val">{bath/size_num:.2f}</span>
      </div>
      <div class="insight-row">
        <span class="insight-key">Luxury flag</span>
        <span class="insight-val">{"✦ Yes" if size_num >= 4 and total_sqft >= 2000 else "No"}</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="gold-divider"></div>', unsafe_allow_html=True)

    # ── Predict button ──
    predict_btn = st.button("✦ &nbsp; Predict Price", use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)


# ── Result panel ──────────────────────────────────────────────────────────────
with col_result:

    if "price" not in st.session_state:
        st.session_state.price = None
        st.session_state.inputs = {}

    if predict_btn:
        with st.spinner("Valuing property…"):
            try:
                price = predict(
                    area_type=area_type,
                    availability=availability,
                    location=location,
                    size=size,
                    total_sqft=float(total_sqft),
                    bath=float(bath),
                    balcony=float(balcony),
                )
                st.session_state.price = price
                st.session_state.inputs = {
                    "location": location,
                    "size": size,
                    "sqft": total_sqft,
                    "bath": bath,
                    "balcony": balcony,
                    "area_type": area_type,
                }
            except Exception as e:
                st.error(f"Prediction failed: {e}")

    if st.session_state.price is None:
        st.markdown("""
        <div style="display:flex;flex-direction:column;align-items:center;
                    justify-content:center;height:60vh;text-align:center;padding:40px;">
          <div style="font-size:64px;opacity:0.2;margin-bottom:20px;">🏛️</div>
          <div style="color:#3a3020;font-size:15px;line-height:1.8;max-width:240px;">
            Fill in the property details and press <strong style="color:#5a4a20;">Predict Price</strong>
            to get an instant valuation.
          </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        price = st.session_state.price
        inp   = st.session_state.inputs

        # Format price
        if price >= 100:
            display = f"₹ {price/100:.2f} Cr"
            unit    = f"({price:.2f} Lakhs)"
        else:
            display = f"₹ {price:.2f} L"
            unit    = "Lakhs"

        price_per_sqft = (price * 100000) / inp["sqft"] if inp["sqft"] else 0

        st.markdown(f"""
        <div style="padding:32px 32px 0;">
          <div class="section-label">Estimated Valuation</div>

          <div class="price-card">
            <div class="price-tag">Market Estimate</div>
            <div class="price-value"><span>{display.split()[0]}</span> {" ".join(display.split()[1:])}</div>
            <div class="price-unit">{unit}</div>
          </div>

          <div class="stat-grid">
            <div class="stat-pill">
              <div class="stat-pill-label">Price / sqft</div>
              <div class="stat-pill-value">₹ {price_per_sqft:,.0f}</div>
            </div>
            <div class="stat-pill">
              <div class="stat-pill-label">Size</div>
              <div class="stat-pill-value">{inp['size']}</div>
            </div>
            <div class="stat-pill">
              <div class="stat-pill-label">Area</div>
              <div class="stat-pill-value">{inp['sqft']:,} sqft</div>
            </div>
            <div class="stat-pill">
              <div class="stat-pill-label">Bath · Balcony</div>
              <div class="stat-pill-value">{inp['bath']} · {inp['balcony']}</div>
            </div>
          </div>

          <div class="insight-card" style="margin-top:20px;">
            <div class="insight-title">Configuration</div>
            <div class="insight-row">
              <span class="insight-key">Location</span>
              <span class="insight-val">{inp['location']}</span>
            </div>
            <div class="insight-row">
              <span class="insight-key">Area type</span>
              <span class="insight-val">{inp['area_type'].strip()}</span>
            </div>
            <div class="insight-row">
              <span class="insight-key">Price range</span>
              <span class="insight-val">₹ {price*0.92:.1f}L – ₹ {price*1.08:.1f}L</span>
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)