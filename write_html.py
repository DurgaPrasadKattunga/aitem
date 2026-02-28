"""Helper: writes full htmlTemplates.py with ChatGPT-style dark UI"""
import os

HTML_CONTENT = r"""css = '''
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
}
.stApp { background-color: #212121 !important; }
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
header { visibility: hidden; }

/* ---- Sidebar ---- */
[data-testid="stSidebar"] {
    background-color: #171717 !important;
    border-right: 1px solid #2f2f2f;
}
[data-testid="stSidebar"] * { color: #ececec !important; }
[data-testid="stSidebar"] hr { border-color: #2f2f2f !important; margin: 0.5rem 0 !important; }

.sidebar-title {
    font-size: 1.1rem; font-weight: 700; color: #fff !important;
    text-align: center; padding: 0.4rem 0 0.1rem 0;
}
.sidebar-subtitle {
    font-size: 0.72rem; color: #8e8ea0 !important;
    text-align: center; padding-bottom: 0.4rem; line-height: 1.4;
}
.section-header {
    font-size: 0.67rem; font-weight: 700; text-transform: uppercase;
    letter-spacing: 1.8px; color: #10a37f !important;
    padding: 0.65rem 0 0.2rem 0; margin: 0;
}
.about-box {
    background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.07);
    border-radius: 10px; padding: 0.7rem 0.85rem; font-size: 0.78rem;
    line-height: 1.55; color: #b0b0c0 !important;
}
.integrity-notice {
    background: rgba(231,76,60,0.07); border: 1px solid rgba(231,76,60,0.18);
    border-left: 3px solid #e74c3c; border-radius: 8px;
    padding: 0.7rem 0.85rem; font-size: 0.75rem; line-height: 1.6; color: #ccccdd !important;
}
.integrity-notice em { color: #e74c3c !important; font-weight: 600; }

.status-badge {
    display: inline-block; padding: 0.2rem 0.65rem;
    border-radius: 20px; font-size: 0.7rem; font-weight: 600; margin: 0.1rem 0;
}
.status-ready {
    background: rgba(16,163,127,0.12); color: #10a37f !important;
    border: 1px solid rgba(16,163,127,0.28);
}
.status-waiting {
    background: rgba(241,196,15,0.1); color: #f1c40f !important;
    border: 1px solid rgba(241,196,15,0.25);
}
.status-row {
    display: flex; justify-content: space-between;
    align-items: center; padding: 0.22rem 0; font-size: 0.77rem;
}
.status-label { color: #8e8ea0 !important; font-weight: 500; }
.status-value { color: #ececec !important; font-weight: 600; }

[data-testid="stSidebar"] .stButton > button {
    width: 100%; border-radius: 8px; font-weight: 600; font-size: 0.82rem;
    padding: 0.42rem 0.8rem; transition: all 0.18s ease;
    border: 1px solid #3f3f3f !important; background: #2f2f2f !important; color: #ececec !important;
}
[data-testid="stSidebar"] .stButton > button:hover {
    background: #3a3a3a !important; border-color: #10a37f !important;
}
[data-testid="stSidebar"] [data-testid="stFileUploader"] {
    background: rgba(255,255,255,0.03); border: 1px dashed #3f3f3f; border-radius: 10px;
}
[data-testid="stSidebar"] .stSelectbox > div > div {
    background: #2f2f2f !important; border-color: #3f3f3f !important; color: #ececec !important;
}

/* ---- Main layout ---- */
.block-container {
    padding-top: 1.2rem !important; padding-bottom: 7rem !important;
    max-width: 860px !important; margin: 0 auto !important;
}
.main-header {
    text-align: center; padding: 0.3rem 0 0.8rem 0;
    border-bottom: 1px solid #2f2f2f; margin-bottom: 1rem;
}
.main-header h1 { font-size: 1.55rem; font-weight: 700; color: #ececec; margin-bottom: 0.15rem; }
.main-header p { font-size: 0.85rem; color: #8e8ea0; margin: 0; }

.welcome-box {
    background: #2a2a2a; border: 1px solid #3f3f3f;
    border-radius: 16px; padding: 1.6rem 1.8rem; margin: 0.5rem 0 1.2rem 0;
}
.welcome-box h3 { color: #10a37f; font-size: 1.15rem; margin-bottom: 0.7rem; }
.welcome-box p, .welcome-box li {
    font-size: 0.88rem; line-height: 1.75; color: #c8c8d8;
}
.welcome-box ul, .welcome-box ol { padding-left: 1.2rem; }
.welcome-box em { color: #8e8ea0; }

/* ---- Chat messages (st.chat_message) ---- */
[data-testid="stChatMessage"] p,
[data-testid="stChatMessage"] li,
[data-testid="stChatMessage"] span {
    color: #ececec !important; font-size: 0.95rem !important; line-height: 1.75 !important;
}
[data-testid="stChatMessage"] h1,
[data-testid="stChatMessage"] h2,
[data-testid="stChatMessage"] h3,
[data-testid="stChatMessage"] h4 { color: #ececec !important; }
[data-testid="stChatMessage"] strong { color: #ffffff !important; }
[data-testid="stChatMessage"] code {
    background: #1e1e1e !important; color: #10a37f !important;
    padding: 0.15rem 0.4rem; border-radius: 4px; font-size: 0.88rem !important;
}
[data-testid="stChatMessage"] pre {
    background: #1a1a1a !important; border: 1px solid #3f3f3f;
    border-radius: 8px; padding: 1rem !important;
}
[data-testid="stChatMessage"] blockquote {
    border-left: 3px solid #10a37f; padding-left: 1rem; color: #aaaacc !important;
}

/* User message background */
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
    background: #2f2f2f !important; border-radius: 14px !important;
    border: 1px solid #3a3a3a !important; padding: 0.8rem 1.1rem !important;
    margin-bottom: 0.4rem !important;
}
/* Assistant message */
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) {
    background: transparent !important; padding: 0.8rem 1.1rem !important;
    margin-bottom: 0.4rem !important;
}

/* Avatars */
[data-testid="stChatMessageAvatarUser"] { background: #10a37f !important; }
[data-testid="stChatMessageAvatarAssistant"] { background: #5436da !important; }

/* ---- Chat input bar ---- */
[data-testid="stChatInput"] {
    background: #2f2f2f !important; border: 1px solid #4f4f4f !important;
    border-radius: 14px !important;
}
[data-testid="stChatInput"] textarea {
    background: transparent !important; color: #ececec !important; font-size: 0.95rem !important;
}
[data-testid="stChatInput"] textarea::placeholder { color: #6e6e80 !important; }
[data-testid="stChatInput"]:focus-within {
    border-color: #10a37f !important;
    box-shadow: 0 0 0 2px rgba(16, 163, 127, 0.15) !important;
}

/* ---- General widget styling ---- */
.stMarkdown p, .stMarkdown span { color: #ececec; }
.stCaption { color: #8e8ea0 !important; }
.stSuccess > div { background: rgba(16,163,127,0.1) !important; color: #ececec !important; }
.stWarning > div { background: rgba(241,196,15,0.08) !important; color: #ececec !important; }
.stError > div { background: rgba(231,76,60,0.08) !important; color: #ececec !important; }
.stInfo > div { background: rgba(16,163,127,0.07) !important; color: #ececec !important; }

[data-testid="stExpander"] {
    background: #2a2a2a !important; border: 1px solid #3f3f3f !important; border-radius: 10px !important;
}
[data-testid="stExpander"] summary { color: #ececec !important; font-size: 0.85rem !important; }
[data-testid="stExpander"] summary:hover { color: #10a37f !important; }

.stSlider label { color: #ececec !important; font-size: 0.82rem !important; }
.stToggle label { color: #ececec !important; font-size: 0.85rem !important; }

/* ---- Footer ---- */
.footer {
    text-align: center; padding: 0.5rem 1rem; font-size: 0.74rem;
    color: #6e6e80; border-top: 1px solid #2f2f2f; margin-top: 0.8rem;
}

.voice-label { font-size: 0.82rem; color: #8e8ea0; margin-bottom: 0.3rem; padding-top: 0.4rem; }

/* ---- Audio player ---- */
audio {
    border-radius: 12px; margin: 0.4rem 0;
    max-width: 360px; height: 34px; display: block;
}

/* ---- Scrollbar ---- */
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: #1a1a1a; }
::-webkit-scrollbar-thumb { background: #3f3f3f; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #5f5f5f; }
</style>
'''

# Fallback templates (not used when st.chat_message is active)
bot_template = '''
<div style="display:flex;align-items:flex-start;gap:10px;padding:0.8rem;margin-bottom:0.4rem;">
    <span style="font-size:1.3rem;">🎓</span>
    <div style="color:#ececec;font-size:0.95rem;line-height:1.75;">{{MSG}}</div>
</div>
'''

user_template = '''
<div style="display:flex;align-items:flex-start;gap:10px;padding:0.8rem;
border-radius:14px;background:#2f2f2f;border:1px solid #3a3a3a;margin-bottom:0.4rem;">
    <span style="font-size:1.3rem;">👤</span>
    <div style="color:#ececec;font-size:0.95rem;line-height:1.75;">{{MSG}}</div>
</div>
'''
"""

with open(os.path.join(os.path.dirname(__file__), "htmlTemplates.py"), "w", encoding="utf-8") as f:
    f.write(HTML_CONTENT)

print(f"htmlTemplates.py written: {len(HTML_CONTENT.splitlines())} lines")
