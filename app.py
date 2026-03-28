import streamlit as st
import pandas as pd
import nltk
from transformers import pipeline
import requests
from bs4 import BeautifulSoup

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
from nltk.tokenize import sent_tokenize

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="GreenTruth Auditor",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

*, *::before, *::after { box-sizing: border-box; }

html, body, [data-testid="stAppViewContainer"] {
    background: #0a0f0a !important;
    color: #e8f0e8 !important;
    font-family: 'DM Sans', sans-serif !important;
}

[data-testid="stAppViewContainer"] {
    background:
        radial-gradient(ellipse 80% 50% at 20% -10%, rgba(34,85,34,0.35) 0%, transparent 60%),
        radial-gradient(ellipse 60% 40% at 80% 110%, rgba(20,60,20,0.3) 0%, transparent 60%),
        #0a0f0a !important;
    min-height: 100vh;
}

[data-testid="stHeader"] { background: transparent !important; }
[data-testid="stSidebar"] { background: #0d130d !important; }
#MainMenu, footer, [data-testid="stDecoration"] { display: none !important; }

::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: #0a0f0a; }
::-webkit-scrollbar-thumb { background: #2d5a2d; border-radius: 4px; }

.main .block-container {
    max-width: 1200px !important;
    padding: 2rem 3rem 4rem !important;
}

.hero-wrap {
    text-align: center;
    padding: 3.5rem 0 2.5rem;
}

.hero-eyebrow {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    font-weight: 500;
    letter-spacing: 0.22em;
    color: #4caf50;
    text-transform: uppercase;
    margin-bottom: 1.2rem;
    display: block;
}

.hero-title {
    font-family: 'DM Serif Display', serif;
    font-size: clamp(3rem, 6vw, 5.5rem);
    font-weight: 400;
    line-height: 1.05;
    color: #e8f0e8;
    margin: 0 0 0.6rem;
    letter-spacing: -0.02em;
}

.hero-title em { font-style: italic; color: #6fcf7a; }

.hero-sub {
    font-family: 'DM Sans', sans-serif;
    font-size: 1.05rem;
    font-weight: 300;
    color: #7a9b7a;
    max-width: 520px;
    margin: 0 auto 2.5rem;
    line-height: 1.6;
}

.hero-divider {
    width: 60px;
    height: 1px;
    background: linear-gradient(90deg, transparent, #4caf50, transparent);
    margin: 0 auto 3rem;
}

.stat-row {
    display: flex;
    justify-content: center;
    gap: 1.2rem;
    flex-wrap: wrap;
    margin-bottom: 3rem;
}

.stat-pill {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(76,175,80,0.2);
    border-radius: 100px;
    padding: 0.55rem 1.3rem;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.78rem;
    color: #6fcf7a;
    letter-spacing: 0.05em;
}

.input-card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(76,175,80,0.15);
    border-radius: 20px;
    padding: 2rem 2.2rem 1.8rem;
    margin-bottom: 1.5rem;
    position: relative;
    overflow: hidden;
}

.input-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(111,207,122,0.4), transparent);
}

.card-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.68rem;
    letter-spacing: 0.18em;
    color: #4caf50;
    text-transform: uppercase;
    margin-bottom: 1rem;
    display: block;
}

.stTextArea textarea {
    background: rgba(0,0,0,0.3) !important;
    border: 1px solid rgba(76,175,80,0.25) !important;
    border-radius: 12px !important;
    color: #e8f0e8 !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.95rem !important;
    line-height: 1.6 !important;
    padding: 1rem !important;
}
.stTextArea textarea:focus {
    border-color: rgba(76,175,80,0.6) !important;
    box-shadow: 0 0 0 3px rgba(76,175,80,0.08) !important;
    outline: none !important;
}
.stTextArea textarea::placeholder { color: rgba(122,155,122,0.5) !important; }

.stTextInput input {
    background: rgba(0,0,0,0.3) !important;
    border: 1px solid rgba(76,175,80,0.25) !important;
    border-radius: 10px !important;
    color: #e8f0e8 !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.85rem !important;
    padding: 0.7rem 1rem !important;
}

.stRadio > div { gap: 0.5rem !important; }
.stRadio label {
    background: rgba(255,255,255,0.03) !important;
    border: 1px solid rgba(76,175,80,0.15) !important;
    border-radius: 8px !important;
    padding: 0.5rem 1.1rem !important;
    color: #7a9b7a !important;
    font-size: 0.85rem !important;
    cursor: pointer !important;
    transition: all 0.2s !important;
}
.stRadio label:has(input:checked) {
    background: rgba(76,175,80,0.12) !important;
    border-color: rgba(76,175,80,0.45) !important;
    color: #6fcf7a !important;
}

.stButton button {
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 500 !important;
    font-size: 0.9rem !important;
    border-radius: 10px !important;
    padding: 0.65rem 1.8rem !important;
    transition: all 0.2s !important;
    border: none !important;
}
.stButton button[kind="primary"] {
    background: linear-gradient(135deg, #2d7a32, #4caf50) !important;
    color: #fff !important;
    box-shadow: 0 4px 20px rgba(76,175,80,0.25) !important;
}
.stButton button[kind="primary"]:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 28px rgba(76,175,80,0.4) !important;
}
.stButton button[kind="secondary"] {
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid rgba(76,175,80,0.25) !important;
    color: #6fcf7a !important;
}

.metric-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 1rem;
    margin-bottom: 2rem;
}

.metric-card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(76,175,80,0.12);
    border-radius: 16px;
    padding: 1.4rem 1.6rem;
    position: relative;
    overflow: hidden;
}

.metric-card::after {
    content: '';
    position: absolute;
    bottom: 0; left: 0; right: 0;
    height: 2px;
    border-radius: 0 0 16px 16px;
}
.metric-card.green::after  { background: linear-gradient(90deg, #2d7a32, #6fcf7a); }
.metric-card.amber::after  { background: linear-gradient(90deg, #b45309, #fbbf24); }
.metric-card.red::after    { background: linear-gradient(90deg, #991b1b, #f87171); }
.metric-card.neutral::after{ background: linear-gradient(90deg, #1e3a5f, #60a5fa); }

.metric-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.15em;
    color: #7a9b7a;
    text-transform: uppercase;
    margin-bottom: 0.6rem;
    display: block;
}

.metric-value {
    font-family: 'DM Serif Display', serif;
    font-size: 2.6rem;
    line-height: 1;
    color: #e8f0e8;
    margin-bottom: 0.3rem;
}

.metric-sublabel { font-size: 0.78rem; color: #7a9b7a; font-weight: 300; }

.risk-bar-wrap {
    background: rgba(255,255,255,0.06);
    border-radius: 100px;
    height: 6px;
    overflow: hidden;
    margin: 0.5rem 0 0.8rem;
}
.risk-bar-fill { height: 100%; border-radius: 100px; }

.verdict-banner {
    border-radius: 14px;
    padding: 1.1rem 1.6rem;
    margin-bottom: 1.8rem;
    display: flex;
    align-items: center;
    gap: 1rem;
    font-size: 0.92rem;
    font-weight: 500;
}
.verdict-banner.safe   { background: rgba(34,85,34,0.2); border: 1px solid rgba(76,175,80,0.3); color: #6fcf7a; }
.verdict-banner.warn   { background: rgba(120,80,0,0.2); border: 1px solid rgba(251,191,36,0.3); color: #fbbf24; }
.verdict-banner.danger { background: rgba(120,20,20,0.2); border: 1px solid rgba(248,113,113,0.3); color: #f87171; }
.verdict-icon { font-size: 1.4rem; flex-shrink: 0; }

.section-heading {
    font-family: 'DM Serif Display', serif;
    font-size: 1.25rem;
    color: #c8dfc8;
    margin: 2rem 0 1rem;
    display: flex;
    align-items: center;
    gap: 0.7rem;
}
.section-heading::after {
    content: '';
    flex: 1;
    height: 1px;
    background: rgba(76,175,80,0.15);
}

.sentence-card {
    border-radius: 14px;
    padding: 1.1rem 1.4rem;
    margin-bottom: 0.75rem;
    border-left: 3px solid;
    transition: transform 0.15s;
}
.sentence-card:hover { transform: translateX(3px); }
.sentence-card.fluff    { background: rgba(248,113,113,0.06); border-color: rgba(248,113,113,0.5); }
.sentence-card.evidence { background: rgba(76,175,80,0.06); border-color: rgba(76,175,80,0.5); }

.sentence-verdict {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    margin-bottom: 0.5rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.sentence-verdict.fluff    { color: #f87171; }
.sentence-verdict.evidence { color: #6fcf7a; }

.conf-bar-wrap {
    width: 60px; height: 3px;
    background: rgba(255,255,255,0.08);
    border-radius: 2px; overflow: hidden;
    display: inline-block; vertical-align: middle; margin-left: 0.5rem;
}
.conf-bar { height: 100%; border-radius: 2px; }
.conf-bar.fluff    { background: #f87171; }
.conf-bar.evidence { background: #6fcf7a; }

.sentence-text { font-size: 0.92rem; color: #c8dfc8; line-height: 1.55; margin-bottom: 0.5rem; }
.tag-row { display: flex; flex-wrap: wrap; gap: 0.4rem; margin-top: 0.5rem; }
.tag {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.66rem;
    padding: 0.22rem 0.65rem;
    border-radius: 100px;
    letter-spacing: 0.05em;
}
.tag.buzzword { background: rgba(251,191,36,0.1); border: 1px solid rgba(251,191,36,0.3); color: #fbbf24; }
.tag.cert     { background: rgba(76,175,80,0.1);  border: 1px solid rgba(76,175,80,0.3);  color: #6fcf7a; }

.results-scroll { max-height: 580px; overflow-y: auto; padding-right: 4px; }

label, .stRadio legend {
    color: #7a9b7a !important;
    font-size: 0.82rem !important;
    font-family: 'DM Sans', sans-serif !important;
}
hr { border-color: rgba(76,175,80,0.12) !important; }
[data-testid="column"] { padding: 0 0.5rem !important; }
</style>
""", unsafe_allow_html=True)


# ── Model & data loaders ──────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model():
    return pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

@st.cache_data(show_spinner=False)
def load_cert_db():
    return pd.read_csv("certifications.csv")

BUZZWORDS = [
    "eco-friendly","eco friendly","green","natural","sustainable","conscious",
    "earth-friendly","planet-friendly","clean","pure","organic","bio","ethical",
    "responsible","environmentally friendly","carbon neutral","net zero",
    "biodegradable","non-toxic","chemical-free","cruelty-free","vegan",
    "recycled","upcycled","zero waste",
]

def detect_buzzwords(sentence):
    low = sentence.lower()
    return [bw for bw in BUZZWORDS if bw in low]

def check_certification(sentence, cert_db):
    low = sentence.lower()
    for _, row in cert_db.iterrows():
        if str(row["brand"]).lower() in low or str(row["certification"]).lower() in low:
            return {"brand": row["brand"], "certification": row["certification"], "body": row["certifying_body"]}
    return None

def classify_sentence(sentence, classifier):
    result = classifier(sentence, candidate_labels=["verifiable fact-based claim","vague unverifiable marketing claim"])
    return {"score": result["scores"][0], "is_fluff": "vague" in result["labels"][0]}

def scrape_url(url):
    try:
        resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
        soup = BeautifulSoup(resp.text, "html.parser")
        for tag in soup(["script","style","nav","footer","header"]): tag.decompose()
        return " ".join(soup.get_text(separator=" ").split())[:3000]
    except Exception as e:
        return f"ERROR: {e}"


# ═══════════════════════════════════════════════════════════════════════════════
#  HERO
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="hero-wrap">
    <span class="hero-eyebrow">AI-Powered Greenwashing Detector</span>
    <h1 class="hero-title">Green<em>Truth</em><br>Auditor</h1>
    <p class="hero-sub">Cut through the eco-buzzwords. Every claim analysed, every certification verified.</p>
    <div class="hero-divider"></div>
    <div class="stat-row">
        <span class="stat-pill">65+ certified brands</span>
        <span class="stat-pill">NLI zero-shot model</span>
        <span class="stat-pill">Real-time RAG lookup</span>
        <span class="stat-pill">8 certification bodies</span>
    </div>
</div>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  LAYOUT
# ═══════════════════════════════════════════════════════════════════════════════
col_input, col_gap, col_results = st.columns([5, 0.4, 6])

with col_input:
    # Input card
    st.markdown('<div class="input-card">', unsafe_allow_html=True)
    st.markdown('<span class="card-label">01 — Input</span>', unsafe_allow_html=True)
    mode = st.radio("", ["📋  Paste text", "🔗  Enter URL"], horizontal=True, label_visibility="collapsed")
    raw_text = ""

    if mode == "📋  Paste text":
        raw_text = st.text_area(
            "",
            height=220,
            placeholder="Paste a product description here…\n\ne.g. Our eco-friendly shampoo is made with\nGOTS-certified organic ingredients.",
            label_visibility="collapsed",
        )
    else:
        url_input = st.text_input("", placeholder="https://brand.com/product-page", label_visibility="collapsed")
        if st.button("Fetch page →", use_container_width=True):
            with st.spinner("Fetching…"):
                fetched = scrape_url(url_input)
            if fetched.startswith("ERROR"):
                st.error(fetched)
            else:
                st.session_state["fetched_text"] = fetched
        if "fetched_text" in st.session_state:
            raw_text = st.text_area("", value=st.session_state["fetched_text"], height=160, label_visibility="collapsed")

    st.markdown("</div>", unsafe_allow_html=True)

    run_audit = st.button("🔍  Run Audit", type="primary", use_container_width=True)

    # How it works card
    st.markdown("""
    <div class="input-card" style="margin-top:1.2rem">
        <span class="card-label">02 — How it works</span>
        <div style="display:flex;flex-direction:column;gap:0.9rem;margin-top:0.3rem">
            <div style="display:flex;gap:1rem;align-items:flex-start">
                <span style="font-family:'JetBrains Mono',monospace;font-size:0.7rem;color:#4caf50;min-width:20px;padding-top:2px">01</span>
                <div>
                    <div style="font-size:0.85rem;color:#c8dfc8;font-weight:500;margin-bottom:0.1rem">Sentence tokenisation</div>
                    <div style="font-size:0.78rem;color:#7a9b7a;line-height:1.5">Text split into individual claims for analysis.</div>
                </div>
            </div>
            <div style="display:flex;gap:1rem;align-items:flex-start">
                <span style="font-family:'JetBrains Mono',monospace;font-size:0.7rem;color:#4caf50;min-width:20px;padding-top:2px">02</span>
                <div>
                    <div style="font-size:0.85rem;color:#c8dfc8;font-weight:500;margin-bottom:0.1rem">NLI classification</div>
                    <div style="font-size:0.78rem;color:#7a9b7a;line-height:1.5">BART-MNLI scores each sentence for verifiability.</div>
                </div>
            </div>
            <div style="display:flex;gap:1rem;align-items:flex-start">
                <span style="font-family:'JetBrains Mono',monospace;font-size:0.7rem;color:#4caf50;min-width:20px;padding-top:2px">03</span>
                <div>
                    <div style="font-size:0.85rem;color:#c8dfc8;font-weight:500;margin-bottom:0.1rem">Buzzword detection</div>
                    <div style="font-size:0.78rem;color:#7a9b7a;line-height:1.5">25 tracked vague terms flagged automatically.</div>
                </div>
            </div>
            <div style="display:flex;gap:1rem;align-items:flex-start">
                <span style="font-family:'JetBrains Mono',monospace;font-size:0.7rem;color:#4caf50;min-width:20px;padding-top:2px">04</span>
                <div>
                    <div style="font-size:0.85rem;color:#c8dfc8;font-weight:500;margin-bottom:0.1rem">RAG certification lookup</div>
                    <div style="font-size:0.78rem;color:#7a9b7a;line-height:1.5">Cross-checked against 65+ verified brands & certifications.</div>
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  RESULTS
# ═══════════════════════════════════════════════════════════════════════════════
with col_results:

    if run_audit and raw_text and not raw_text.startswith("ERROR"):

        with st.spinner("Loading model & analysing claims…"):
            classifier = load_model()
            cert_db    = load_cert_db()
            sentences  = [s.strip() for s in sent_tokenize(raw_text) if len(s.strip()) > 15]
            results    = []
            for sent in sentences:
                clf  = classify_sentence(sent, classifier)
                results.append({
                    "sentence": sent,
                    **clf,
                    "buzzwords": detect_buzzwords(sent),
                    "cert":      check_certification(sent, cert_db),
                })

        total       = len(results)
        fluff_count = sum(1 for r in results if r["is_fluff"])
        evid_count  = total - fluff_count
        bw_total    = sum(len(r["buzzwords"]) for r in results)
        cert_total  = sum(1 for r in results if r["cert"])
        fluff_pct   = round(fluff_count / total * 100) if total else 0

        risk_class = "green" if fluff_pct < 30 else ("amber" if fluff_pct < 60 else "red")
        fill_color = {
            "green": "linear-gradient(90deg,#2d7a32,#6fcf7a)",
            "amber": "linear-gradient(90deg,#92400e,#fbbf24)",
            "red":   "linear-gradient(90deg,#991b1b,#f87171)",
        }[risk_class]

        # Metric cards
        st.markdown(f"""
        <div class="metric-grid">
            <div class="metric-card {risk_class}">
                <span class="metric-label">Greenwash risk</span>
                <div class="metric-value">{fluff_pct}%</div>
                <div class="risk-bar-wrap">
                    <div class="risk-bar-fill" style="width:{fluff_pct}%;background:{fill_color}"></div>
                </div>
                <span class="metric-sublabel">of claims unverifiable</span>
            </div>
            <div class="metric-card neutral">
                <span class="metric-label">Claims analysed</span>
                <div class="metric-value">{total}</div>
                <span class="metric-sublabel">{evid_count} evidence · {fluff_count} fluff</span>
            </div>
            <div class="metric-card {'green' if cert_total > 0 else 'neutral'}">
                <span class="metric-label">Cert matches</span>
                <div class="metric-value">{cert_total}</div>
                <span class="metric-sublabel">{bw_total} buzzwords flagged</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Verdict banner
        if fluff_pct < 30:
            banner_cls, icon, msg = "safe",   "✅", "Low greenwashing risk — most claims are fact-based or certified."
        elif fluff_pct < 60:
            banner_cls, icon, msg = "warn",   "⚠️", "Moderate risk — several claims are vague or unverifiable."
        else:
            banner_cls, icon, msg = "danger", "🚨", "High greenwashing risk — the majority of claims lack evidence."

        st.markdown(f"""
        <div class="verdict-banner {banner_cls}">
            <span class="verdict-icon">{icon}</span>
            <span>{msg}</span>
        </div>
        """, unsafe_allow_html=True)

        # Sentence cards
        st.markdown('<div class="section-heading">Sentence breakdown</div>', unsafe_allow_html=True)
        st.markdown('<div class="results-scroll">', unsafe_allow_html=True)

        for r in results:
            cls     = "fluff" if r["is_fluff"] else "evidence"
            verdict = "Marketing Fluff" if r["is_fluff"] else "Evidence-Based"
            conf_w  = int(r["score"] * 60)
            bw_tags = "".join(f'<span class="tag buzzword">⚠ {bw}</span>' for bw in r["buzzwords"])
            c_tag   = f'<span class="tag cert">✔ {r["cert"]["certification"]}</span>' if r["cert"] else ""
            tags    = bw_tags + c_tag

            st.markdown(f"""
            <div class="sentence-card {cls}">
                <div class="sentence-verdict {cls}">
                    ● {verdict}
                    <span class="conf-bar-wrap">
                        <span class="conf-bar {cls}" style="width:{conf_w}px"></span>
                    </span>
                    <span style="font-size:0.62rem;opacity:0.6">{int(r['score']*100)}% conf.</span>
                </div>
                <div class="sentence-text">{r['sentence']}</div>
                {f'<div class="tag-row">{tags}</div>' if tags else ''}
            </div>
            """, unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

    elif run_audit and not raw_text:
        st.markdown("""
        <div class="verdict-banner warn">
            <span class="verdict-icon">💬</span>
            <span>Please paste a product description or fetch a URL first.</span>
        </div>
        """, unsafe_allow_html=True)

    else:
        st.markdown("""
        <div style="
            border: 1px dashed rgba(76,175,80,0.2);
            border-radius: 20px;
            padding: 5rem 2rem;
            text-align: center;
            margin-top: 0.5rem;
        ">
            <div style="font-size:3.5rem;margin-bottom:1.2rem;opacity:0.35">🌿</div>
            <div style="font-family:'DM Serif Display',serif;font-size:1.4rem;color:#7a9b7a;margin-bottom:0.7rem">
                Your audit report will appear here
            </div>
            <div style="font-size:0.82rem;color:rgba(122,155,122,0.55);max-width:280px;margin:0 auto;line-height:1.7">
                Paste a product description on the left and click <em>Run Audit</em> to begin.
            </div>
        </div>
        """, unsafe_allow_html=True)


# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="
    text-align:center;
    padding:3rem 0 1rem;
    font-family:'JetBrains Mono',monospace;
    font-size:0.65rem;
    letter-spacing:0.12em;
    color:rgba(122,155,122,0.3);
    text-transform:uppercase;
">
    GreenTruth Auditor &nbsp;·&nbsp; BART-MNLI &nbsp;·&nbsp; RAG &nbsp;·&nbsp; Streamlit
</div>
""", unsafe_allow_html=True)
