import streamlit as st
import requests
import os
import base64
from PIL import Image
import io

def load_image_base64(path, size=200):
    try:
        img = Image.open(path).convert("RGB")
        w, h = img.size
        # crop on face area
        left   = int(w * 0.20)
        right  = int(w * 0.70)

        top    = int(h * 0.30)
        bottom = int(h * 0.65) 
        img = img.crop((left, top, right, bottom))
        # make square from center
        cw, ch = img.size
        side = min(cw, ch)
        cx = (cw - side) // 2
        cy = (ch - side) // 2
        img = img.crop((cx, cy, cx + side, cy + side))
        img = img.resize((size, size), Image.LANCZOS)
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=95)
        return base64.b64encode(buffer.getvalue()).decode()
    except:
        return None

# ==============================
# Configuration
# ==============================

FASTAPI_URL  = os.getenv("FASTAPI_URL", "http://localhost:8000/ticket")
HEALTH_URL   = FASTAPI_URL.replace("/ticket", "/health")
LINKEDIN_URL = "www.linkedin.com/in/mahmoud-mohamed-b035a3268"
GITHUB_URL   = "https://github.com/Mahmoud6Elhaddad"

# Load avatar photo
_img = load_image_base64("docs/me.jpeg")
AVATAR_HTML = (
    f'<img src="data:image/jpeg;base64,{_img}" '
    f'width="96" height="96" '
    f'style="width:96px;height:96px;border-radius:50%;object-fit:cover;flex-shrink:0;'
    f'box-shadow:0 0 0 3px rgba(59,130,246,0.5),0 4px 12px rgba(59,130,246,0.4);" />'
    if _img else
    '<div class="author-avatar" style="display:inline-flex;align-items:center;justify-content:center;">ME</div>'
)

st.set_page_config(
    page_title="AI Support Ticket System",
    page_icon="🎫",
    layout="wide"
)

# ==============================
# Session State
# ==============================

if "result"    not in st.session_state: st.session_state.result    = None
if "dark_mode" not in st.session_state: st.session_state.dark_mode = True


# ==============================
# Theme Variables
# ==============================

dark = st.session_state.dark_mode

T = {
    # backgrounds
    "app_bg":       "#0a0e1a"  if dark else "#f1f5f9",
    "sidebar_bg":   "#111827"  if dark else "#ffffff",
    "surface":      "#111827"  if dark else "#ffffff",
    "surface2":     "#1a2235"  if dark else "#f8fafc",
    "hero_bg":      "linear-gradient(135deg,#0f172a 0%,#111827 100%)" if dark else "linear-gradient(135deg,#e0e7ff 0%,#f0f4ff 100%)",
    "card_bg":      "#111827"  if dark else "#ffffff",
    "empty_bg":     "#111827"  if dark else "#f8fafc",
    # borders
    "border":       "rgba(99,179,237,0.15)"  if dark else "rgba(59,130,246,0.2)",
    "border2":      "rgba(99,179,237,0.25)"  if dark else "rgba(59,130,246,0.35)",
    "border_dash":  "rgba(99,179,237,0.2)"   if dark else "rgba(59,130,246,0.25)",
    "sidebar_bdr":  "rgba(99,179,237,0.12)"  if dark else "rgba(59,130,246,0.15)",
    # text
    "text":         "#e2e8f0"  if dark else "#0f172a",
    "text_muted":   "#94a3b8"  if dark else "#475569",
    "text_hint":    "#64748b"  if dark else "#94a3b8",
    "label":        "#3b82f6"  if dark else "#2563eb",
    # inputs
    "input_bg":     "rgba(255,255,255,0.04)" if dark else "#ffffff",
    "input_border": "rgba(99,179,237,0.2)"   if dark else "rgba(59,130,246,0.3)",
    "input_color":  "#e2e8f0"  if dark else "#0f172a",
    # misc
    "footer_bdr":   "rgba(99,179,237,0.1)"   if dark else "rgba(59,130,246,0.15)",
    "acc_track":    "rgba(255,255,255,0.07)"  if dark else "rgba(0,0,0,0.08)",
    "feat_bg":      "rgba(255,255,255,0.03)"  if dark else "rgba(59,130,246,0.04)",
    "feat_bdr":     "rgba(99,179,237,0.12)"   if dark else "rgba(59,130,246,0.15)",
    "toggle_icon":  "🌙" if dark else "☀️",
    "toggle_label": "Dark Mode" if dark else "Light Mode",
    # gradient text — only works on dark; on light use solid color
    "hero_title_style": (
        "background:linear-gradient(135deg,#e2e8f0,#93c5fd,#c4b5fd);"
        "-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;"
    ) if dark else "color:#1e3a8a;",
    "hero_desc_color": "#94a3b8" if dark else "#334155",
    "author_name_color": "#e2e8f0" if dark else "#0f172a",
    "author_title_color": "#64748b" if dark else "#64748b",
    "empty_color": "#475569" if dark else "#94a3b8",
    "footer_color": "#475569" if dark else "#64748b",
    "sidebar_text": "#94a3b8" if dark else "#475569",
}


# ==============================
# Custom CSS (theme-aware)
# ==============================

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {{ font-family: 'Space Grotesk', sans-serif !important; }}
.stApp {{ background: {T['app_bg']}; color: {T['text']}; }}
header[data-testid="stHeader"] {{ background: transparent; }}

[data-testid="stSidebar"] {{
    background: {T['sidebar_bg']} !important;
    border-right: 1px solid {T['sidebar_bdr']};
}}
input, textarea,
.stTextInput input,
.stTextArea textarea,
div[data-baseweb="input"] input,
div[data-baseweb="textarea"] textarea {{
    background: {T['input_bg']} !important;
    border: 1px solid {T['input_border']} !important;
    border-radius: 10px !important;
    color: {T['input_color']} !important;
    font-family: 'Space Grotesk', sans-serif !important;
}}
input::placeholder, textarea::placeholder,
.stTextInput input::placeholder,
.stTextArea textarea::placeholder {{
    color: {T['text_hint']} !important;
    opacity: 1 !important;
}}
div[data-baseweb="input"],
div[data-baseweb="textarea"] {{
    background: {T['input_bg']} !important;
    border-radius: 10px !important;
}}
input:focus, textarea:focus,
.stTextInput input:focus,
.stTextArea textarea:focus {{
    border-color: #3b82f6 !important;
    box-shadow: 0 0 0 2px rgba(59,130,246,0.15) !important;
}}
.stTextInput label, .stTextArea label {{
    color: {T['text']} !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 0.9rem !important;
}}
.stButton > button[kind="primary"] {{
    background: linear-gradient(135deg, #3b82f6, #8b5cf6) !important;
    border: none !important; border-radius: 10px !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 600 !important; font-size: 0.95rem !important;
    color: white !important;
}}
.stButton > button[kind="primary"]:hover {{
    box-shadow: 0 6px 20px rgba(59,130,246,0.35) !important;
}}
.stButton > button:not([kind="primary"]) {{
    color: {T['text']} !important;
    background: {T['surface']} !important;
    border: 1px solid {T['border2']} !important;
    border-radius: 10px !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 500 !important;
}}
.stButton > button:not([kind="primary"]):hover {{
    color: {T['text']} !important;
    background: {T['surface2']} !important;
    border-color: #3b82f6 !important;
    box-shadow: none !important;
}}
[data-testid="stMetric"] {{
    background: {T['card_bg']};
    border: 1px solid {T['border']};
    border-radius: 12px; padding: 1rem 1.25rem;
}}
[data-testid="stMetricLabel"] {{
    font-size: 0.72rem !important;
    font-family: 'JetBrains Mono', monospace !important;
    color: {T['text_hint']} !important;
    text-transform: uppercase; letter-spacing: 1px;
}}
[data-testid="stMetricValue"] {{
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 1.4rem !important; color: {T['text']} !important;
}}
[data-testid="stAlert"] {{
    background: rgba(59,130,246,0.08) !important;
    border: 1px solid rgba(59,130,246,0.25) !important;
    border-radius: 10px !important; color: #3b82f6 !important;
}}
[data-testid="stAlert"][data-baseweb="notification"] p,
.stSuccess p, .stSuccess div {{
    color: {"#065f46" if not dark else "#6ee7b7"} !important;
}}
.element-container .stSuccess {{
    background: {"rgba(16,185,129,0.15)" if not dark else "rgba(16,185,129,0.1)"} !important;
    border: {"2px solid rgba(16,185,129,0.6)" if not dark else "1px solid rgba(16,185,129,0.3)"} !important;
    border-radius: 10px !important;
}}
hr {{ border-color: {T['footer_bdr']} !important; }}
[data-testid="stExpander"] {{
    background: {T['surface2']} !important;
    border: 1px solid {T['border']} !important;
    border-radius: 10px !important;
}}
[data-testid="stJson"],
[data-testid="stJson"] > div,
.stJson {{
    background: {T['surface2']} !important;
    border-radius: 8px !important;
}}
[data-testid="stJson"] span {{
    color: {T['text']} !important;
}}
.streamlit-json-value {{ color: {"#f97316" if dark else "#b45309"} !important; }}
.streamlit-json-key   {{ color: {"#93c5fd" if dark else "#1d4ed8"} !important; }}
.streamlit-json-string {{ color: {"#86efac" if dark else "#15803d"} !important; }}

.hero-box {{
    background: {T['hero_bg']};
    border: 1px solid {T['border']};
    border-radius: 16px; padding: 2rem 2.5rem; margin-bottom: 1rem;
}}
.badge {{
    padding: 4px 12px; border-radius: 100px;
    font-size: 11px; font-family: 'JetBrains Mono', monospace; border: 1px solid;
}}
.badge-blue   {{ background:rgba(59,130,246,0.12);  color:#3b82f6; border-color:rgba(59,130,246,0.35); }}
.badge-purple {{ background:rgba(139,92,246,0.12);  color:#8b5cf6; border-color:rgba(139,92,246,0.35); }}
.badge-cyan   {{ background:rgba(6,182,212,0.12);   color:#06b6d4; border-color:rgba(6,182,212,0.35);  }}
.badge-green  {{ background:rgba(16,185,129,0.12);  color:#10b981; border-color:rgba(16,185,129,0.35); }}
.author-avatar {{
    width:72px; height:72px; border-radius:50%;
    background: linear-gradient(135deg, #3b82f6, #8b5cf6);
    display:inline-flex; align-items:center; justify-content:center;
    font-size:22px; font-weight:700; color:white !important; flex-shrink:0;
    box-shadow: 0 0 0 3px {"rgba(255,255,255,0.15)" if dark else "rgba(59,130,246,0.4)"},
                0 4px 12px rgba(59,130,246,0.4);
}}
.btn-linkedin {{
    display:inline-flex; align-items:center; gap:8px; padding:9px 18px;
    background: linear-gradient(135deg, #0077b5, #0a66c2);
    color:white !important; border-radius:8px; text-decoration:none !important;
    font-size:0.82rem; font-weight:600; font-family:'Space Grotesk',sans-serif;
}}
.btn-github {{
    display:inline-flex; align-items:center; gap:8px; padding:9px 18px;
    background:{T['feat_bg']}; color:{T['text']} !important;
    border:1px solid {T['border2']};
    border-radius:8px; text-decoration:none !important;
    font-size:0.82rem; font-weight:600; font-family:'Space Grotesk',sans-serif;
}}
.section-label {{
    font-size:0.68rem; font-family:'JetBrains Mono',monospace;
    color:{T['label']}; letter-spacing:2px; text-transform:uppercase; margin-bottom:0.75rem;
}}
.empty-state {{
    background:{T['empty_bg']}; border:1px dashed {T['border_dash']};
    border-radius:12px; padding:2rem; text-align:center;
    color:{T['empty_color']}; font-size:0.85rem; line-height:1.7;
}}
.footer-bar {{
    text-align:center; padding:1.5rem; margin-top:2rem;
    border-top:1px solid {T['footer_bdr']};
    font-size:0.78rem; color:{T['footer_color']}; font-family:'Space Grotesk',sans-serif;
}}
</style>
""", unsafe_allow_html=True)


# ==============================
# Hero Section
# ==============================

LI_ICON = '<svg width="14" height="14" viewBox="0 0 24 24" fill="white"><path d="M20.447 20.452h-3.554v-5.569c0-1.328-.027-3.037-1.852-3.037-1.853 0-2.136 1.445-2.136 2.939v5.667H9.351V9h3.414v1.561h.046c.477-.9 1.637-1.85 3.37-1.85 3.601 0 4.267 2.37 4.267 5.455v6.286zM5.337 7.433a2.062 2.062 0 0 1-2.063-2.065 2.064 2.064 0 1 1 2.063 2.065zm1.782 13.019H3.555V9h3.564v11.452zM22.225 0H1.771C.792 0 0 .774 0 1.729v20.542C0 23.227.792 24 1.771 24h20.451C23.2 24 24 23.227 24 22.271V1.729C24 .774 23.2 0 22.222 0h.003z"/></svg>'
GH_ICON = '<svg width="15" height="15" viewBox="0 0 24 24" fill="currentColor"><path d="M12 0C5.37 0 0 5.37 0 12c0 5.31 3.435 9.795 8.205 11.385.6.105.825-.255.825-.57 0-.285-.015-1.23-.015-2.235-3.015.555-3.795-.735-4.035-1.41-.135-.345-.72-1.41-1.23-1.695-.42-.225-1.02-.78-.015-.795.945-.015 1.62.87 1.845 1.23 1.08 1.815 2.805 1.305 3.495.99.105-.78.42-1.305.765-1.605-2.67-.3-5.46-1.335-5.46-5.925 0-1.305.465-2.385 1.23-3.225-.12-.3-.54-1.53.12-3.18 0 0 1.005-.315 3.3 1.23.96-.27 1.98-.405 3-.405s2.04.135 3 .405c2.295-1.56 3.3-1.23 3.3-1.23.66 1.65.24 2.88.12 3.18.765.84 1.23 1.905 1.23 3.225 0 4.605-2.805 5.625-5.475 5.925.435.375.81 1.095.81 2.22 0 1.605-.015 2.895-.015 3.3 0 .315.225.69.825.57A12.02 12.02 0 0 0 24 12c0-6.63-5.37-12-12-12z"/></svg>'

st.markdown(f"""
<div class="hero-box">
  <div style="display:flex;gap:8px;flex-wrap:wrap;margin-bottom:1.25rem;">
    <span class="badge badge-blue">AI/ML Project</span>
    <span class="badge badge-purple">NLP</span>
    <span class="badge badge-cyan">RAG System</span>
    <span class="badge badge-green">FastAPI + Streamlit</span>
  </div>
  <div style="font-size:clamp(1.4rem,3vw,2rem);font-weight:700;margin-bottom:0.6rem;
              font-family:'Space Grotesk',sans-serif;{T['hero_title_style']}">
    AI-Powered Customer Support System
  </div>
  <div style="color:{T['hero_desc_color']};font-size:0.95rem;line-height:1.6;
              max-width:640px;margin-bottom:1.5rem;">
    Intelligent ticket classification, sentiment analysis, priority prediction,
    and RAG-based response generation &mdash; built with classical ML pipelines and modern LLM retrieval.
  </div>
  <div style="display:flex;align-items:center;gap:1rem;flex-wrap:wrap;">
    {AVATAR_HTML}
    <div>
      <div style="font-weight:600;font-size:0.95rem;color:{T['author_name_color']};">Mahmoud Mohamed El-Saeed</div>
      <div style="font-size:0.78rem;color:{T['author_title_color']};margin-top:2px;">
        Computer Engineering &middot; AI/ML Engineer &middot; Mansoura, Egypt
      </div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

st.markdown(
    f'<div style="display:flex;gap:10px;flex-wrap:wrap;margin-bottom:2rem;">'
    f'<a href="{LINKEDIN_URL}" target="_blank" class="btn-linkedin">{LI_ICON} Connect on LinkedIn</a>'
    f'<a href="{GITHUB_URL}"   target="_blank" class="btn-github">{GH_ICON} GitHub</a>'
    f'</div>',
    unsafe_allow_html=True
)


# ==============================
# Main Layout
# ==============================

col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.markdown('<div class="section-label">Submit Ticket</div>', unsafe_allow_html=True)
    ticket_title = st.text_input(
        "Ticket Title",
        placeholder="e.g. Can't log into my account...",
        help="Brief summary of the issue"
    )
    ticket_description = st.text_area(
        "Ticket Description",
        placeholder="Describe the issue in detail...",
        height=180,
        help="Detailed description of the problem"
    )
    submit_button = st.button("🚀  Analyze Ticket", type="primary", use_container_width=True)

with col2:
    st.markdown('<div class="section-label">Analysis Results</div>', unsafe_allow_html=True)
    result_placeholder = st.empty()
    if not st.session_state.result:
        st.markdown(
            '<div class="empty-state">Submit a ticket on the left<br>to see AI analysis here</div>',
            unsafe_allow_html=True
        )


# ==============================
# Process Ticket
# ==============================

if submit_button:
    if not ticket_title or not ticket_description:
        st.error("⚠️ Please fill in both title and description!")
        st.stop()

    with st.spinner("Analyzing your ticket..."):
        try:
            response = requests.post(
                FASTAPI_URL,
                json={"title": ticket_title, "description": ticket_description},
                timeout=30
            )
            if response.status_code == 200:
                try:
                    result = response.json()
                except Exception:
                    st.error("Invalid JSON response from backend.")
                    st.stop()
                st.session_state.result = result
            else:
                st.error(f"Backend error {response.status_code}: {response.text}")
                st.stop()

        except requests.exceptions.ConnectionError:
            st.error("❌ Cannot connect to FastAPI backend. Make sure it is running.")
            st.stop()
        except requests.exceptions.Timeout:
            st.error("⏳ Backend request timed out.")
            st.stop()
        except Exception as e:
            st.error(f"Unexpected error: {str(e)}")
            st.stop()


# ==============================
# Display Results
# ==============================

if st.session_state.result:
    result = st.session_state.result

    with result_placeholder.container():
        st.success("✅ Ticket analyzed successfully!")

        category  = result.get("category",  "N/A").upper()
        priority  = result.get("priority",  "N/A").upper()
        sentiment = result.get("sentiment", "N/A").upper()

        priority_icon  = {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🟢"}.get(priority, "⚪")
        sentiment_icon = {"POSITIVE": "😊", "NEGATIVE": "😞", "NEUTRAL": "😐"}.get(sentiment, "😐")

        m1, m2, m3 = st.columns(3)
        with m1: st.metric("Category",  category)
        with m2: st.metric("Priority",  f"{priority_icon} {priority}")
        with m3: st.metric("Sentiment", f"{sentiment_icon} {sentiment}")

        st.divider()
        st.markdown("**💡 Suggested Solution**")
        st.info(result.get("suggested_solution", "No solution available"))

        with st.expander("🔍 Raw JSON Response"):
            st.json(result)


# ==============================
# Sidebar
# ==============================

with st.sidebar:

    # ── Dark / Light Toggle ──
    st.markdown('<div class="section-label">Theme</div>', unsafe_allow_html=True)

    toggle_label = f"{T['toggle_icon']}  {T['toggle_label']}"
    if st.button(toggle_label, use_container_width=True):
        st.session_state.dark_mode = not st.session_state.dark_mode
        st.rerun()

    st.markdown("---")

    # ── About ──
    st.markdown(
        f"<div style='font-size:1.05rem;font-weight:700;color:{T['text']};margin-bottom:0.5rem;'>🎫 About This Project</div>",
        unsafe_allow_html=True
    )
    st.markdown(
        f"<span style='font-size:0.82rem;color:{T['sidebar_text']};'>"
        "An AI-powered system that automatically processes customer support tickets using:"
        "</span>",
        unsafe_allow_html=True
    )

    for icon, label, color in [
        ("🤖", "ML Classification",   "#60a5fa"),
        ("📊", "Priority Detection",  "#a78bfa"),
        ("💬", "Sentiment Analysis",  "#34d399"),
        ("🔍", "RAG Response System", "#22d3ee"),
    ]:
        st.markdown(
            f"<div style='display:flex;align-items:center;gap:8px;padding:8px 12px;"
            f"background:{T['feat_bg']};border:1px solid {T['feat_bdr']};"
            f"border-radius:8px;font-size:0.78rem;color:{T['sidebar_text']};margin-bottom:6px;'>"
            f"<span style='color:{color};'>{icon}</span> {label}</div>",
            unsafe_allow_html=True
        )

    st.markdown("---")
    st.markdown('<div class="section-label">Backend Status</div>', unsafe_allow_html=True)
    try:
        health_response = requests.get(HEALTH_URL, timeout=2)
        if health_response.status_code == 200:
            st.success("✅ Backend is running")
        else:
            st.warning("⚠️ Backend responding with errors")
    except:
        st.error("❌ Backend is not running")

    st.markdown("---")
    st.markdown('<div class="section-label">How to Run</div>', unsafe_allow_html=True)
    st.code("uvicorn app:app --reload", language="bash")
    st.code("streamlit run streamlit_app.py", language="bash")

    st.markdown("---")
    st.markdown('<div class="section-label">Model Accuracy</div>', unsafe_allow_html=True)

    for task, model, acc, color in [
        ("Category",  "Logistic Regression", 97, "#3b82f6"),
        ("Sentiment", "Linear SVM",          64, "#8b5cf6"),
        ("Priority",  "Linear SVM",          60, "#f59e0b"),
    ]:
        st.markdown(
            f"<div style='margin-bottom:10px;'>"
            f"<div style='display:flex;justify-content:space-between;font-size:0.72rem;margin-bottom:4px;'>"
            f"<span style='color:{T['sidebar_text']};'>{task} &middot; {model}</span>"
            f"<span style='color:{color};font-family:monospace;font-weight:600;'>{acc}%</span>"
            f"</div>"
            f"<div style='height:4px;background:{T['acc_track']};border-radius:100px;'>"
            f"<div style='height:4px;width:{acc}%;background:{color};border-radius:100px;'></div>"
            f"</div></div>",
            unsafe_allow_html=True
        )


# ==============================
# Footer
# ==============================

st.markdown(
    f"<div class='footer-bar'>"
    f"Built with <span style='color:#ef4444;'>&#10084;</span> using "
    f"<span style='color:#60a5fa;'>Streamlit</span> + "
    f"<span style='color:#a78bfa;'>FastAPI</span> + "
    f"<span style='color:#22d3ee;'>RAG</span>"
    f" &nbsp;&middot;&nbsp; <b style='color:{T['text_muted']};'>Eng. Mahmoud Mohamed El-Saeed</b>"
    f" &nbsp;&middot;&nbsp; <a href='{LINKEDIN_URL}' target='_blank'"
    f" style='color:#0a66c2;text-decoration:none;'>LinkedIn</a>"
    f"</div>",
    unsafe_allow_html=True
)