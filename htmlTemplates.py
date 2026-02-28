css = '''
<style>
/* ============================================================
   ChatGPT-Style Dark Theme for Education Exam Explainer Bot
   ============================================================ */

/* ---- Global Dark Mode ---- */
.stApp, [data-testid="stAppViewContainer"] {
    background-color: #212121 !important;
    color: #ececec !important;
}
[data-testid="stMainBlockContainer"] {
    background-color: #212121 !important;
}
[data-testid="stHeader"] {
    background-color: #212121 !important;
}

/* ---- Sidebar (dark gradient) ---- */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #171717 0%, #1e1e2e 100%);
    border-right: 1px solid rgba(255,255,255,0.06);
}
[data-testid="stSidebar"] * {
    color: #d1d1d1 !important;
}
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stSlider label,
[data-testid="stSidebar"] .stFileUploader label {
    font-size: 0.85rem;
    font-weight: 500;
    color: #a0a0b8 !important;
}
[data-testid="stSidebar"] hr {
    border-color: rgba(255,255,255,0.06);
}
[data-testid="stSidebar"] .sidebar-title {
    font-size: 1.3rem;
    font-weight: 700;
    color: #ffffff !important;
    text-align: center;
    padding: 0.5rem 0 0.2rem 0;
    letter-spacing: 0.5px;
}
[data-testid="stSidebar"] .sidebar-subtitle {
    font-size: 0.78rem;
    color: #8888aa !important;
    text-align: center;
    padding-bottom: 0.6rem;
}
[data-testid="stSidebar"] .section-header {
    font-size: 0.7rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    color: #7b8cde !important;
    padding: 0.8rem 0 0.3rem 0;
    margin: 0;
}
[data-testid="stSidebar"] .status-badge {
    display: inline-block;
    padding: 0.25rem 0.75rem;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 600;
    margin: 0.15rem 0;
}
[data-testid="stSidebar"] .status-ready {
    background: rgba(46, 204, 113, 0.15);
    color: #2ecc71 !important;
    border: 1px solid rgba(46, 204, 113, 0.3);
}
[data-testid="stSidebar"] .status-waiting {
    background: rgba(241, 196, 15, 0.15);
    color: #f1c40f !important;
    border: 1px solid rgba(241, 196, 15, 0.3);
}
[data-testid="stSidebar"] .status-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0.3rem 0;
    font-size: 0.8rem;
}
[data-testid="stSidebar"] .status-label {
    color: #9999bb !important;
    font-weight: 500;
}
[data-testid="stSidebar"] .status-value {
    color: #e0e0e0 !important;
    font-weight: 600;
}
/* Sidebar buttons */
[data-testid="stSidebar"] .stButton > button {
    width: 100%;
    border-radius: 10px;
    font-weight: 600;
    font-size: 0.85rem;
    padding: 0.55rem 1rem;
    transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
    border: none;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white !important;
}
[data-testid="stSidebar"] .stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(102, 126, 234, 0.45);
    filter: brightness(1.08);
}

/* ---- Main Area ---- */
.main-header {
    text-align: center;
    padding: 1.2rem 0 0.6rem 0;
}
.main-header h1 {
    font-size: 1.8rem;
    font-weight: 700;
    background: linear-gradient(135deg, #667eea, #a78bfa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.main-header p {
    font-size: 0.9rem;
    color: #888;
    margin-top: -0.5rem;
}

/* Welcome Box */
.welcome-box {
    background: #2f2f2f;
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 18px;
    padding: 2rem 2.2rem;
    margin: 1.5rem auto;
    max-width: 720px;
    color: #e0e0e0;
    box-shadow: 0 4px 24px rgba(0, 0, 0, 0.3);
    animation: fadeInUp 0.5s ease;
}
.welcome-box h3 {
    color: #a78bfa;
    font-size: 1.3rem;
    margin-bottom: 0.8rem;
}
.welcome-box p, .welcome-box li {
    font-size: 0.92rem;
    line-height: 1.7;
    color: #c8c8d8;
}
.welcome-box ol, .welcome-box ul {
    padding-left: 1.2rem;
}
.welcome-box em {
    color: #8888aa;
}

/* About Box (sidebar) */
[data-testid="stSidebar"] .about-box {
    background: rgba(102, 126, 234, 0.08);
    border: 1px solid rgba(102, 126, 234, 0.15);
    border-radius: 10px;
    padding: 0.75rem 0.9rem;
    font-size: 0.8rem;
    line-height: 1.5;
    color: #b0b0c0 !important;
}

/* Academic Integrity Notice (sidebar) */
[data-testid="stSidebar"] .integrity-notice {
    background: rgba(231, 76, 60, 0.08);
    border: 1px solid rgba(231, 76, 60, 0.2);
    border-left: 3px solid #e74c3c;
    border-radius: 8px;
    padding: 0.75rem 0.9rem;
    font-size: 0.78rem;
    line-height: 1.6;
    color: #ccccdd !important;
}
[data-testid="stSidebar"] .integrity-notice em {
    color: #e74c3c !important;
    font-weight: 600;
}

/* Footer */
.footer {
    text-align: center;
    padding: 0.8rem 1rem;
    font-size: 0.8rem;
    color: #777;
    background: rgba(255, 255, 255, 0.03);
    border-radius: 12px;
    margin-top: 0.5rem;
    border: 1px solid rgba(255,255,255,0.05);
}

/* ============================================================
   Chat Messages — ChatGPT-style rounded bubbles
   ============================================================ */
[data-testid="stChatMessage"] {
    border-radius: 18px;
    margin-bottom: 0.6rem;
    padding: 1rem 1.2rem;
    animation: fadeInUp 0.35s cubic-bezier(0.4, 0, 0.2, 1);
    border: 1px solid rgba(255, 255, 255, 0.06);
    transition: box-shadow 0.2s ease;
}
[data-testid="stChatMessage"]:hover {
    box-shadow: 0 2px 12px rgba(0, 0, 0, 0.25);
}

@keyframes fadeInUp {
    from { opacity: 0; transform: translateY(12px); }
    to { opacity: 1; transform: translateY(0); }
}

/* User messages — subtle purple accent */
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
    background: #2f2f2f;
    border-left: 3px solid #667eea;
}
/* Assistant messages — subtle green accent */
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) {
    background: #343541;
    border-left: 3px solid #10a37f;
}

[data-testid="stChatMessage"] p {
    color: #ececec;
    font-size: 0.95rem;
    line-height: 1.75;
}
[data-testid="stChatMessage"] .stMarkdown {
    color: #ececec;
}
[data-testid="stChatMessage"] li {
    color: #ececec;
    line-height: 1.7;
}
[data-testid="stChatMessage"] strong {
    color: #ffffff;
}
[data-testid="stChatMessage"] code {
    background: rgba(255, 255, 255, 0.08);
    border-radius: 4px;
    padding: 0.1rem 0.35rem;
    font-size: 0.88rem;
    color: #a78bfa;
}

/* ---- Chat Input Styling ---- */
[data-testid="stChatInput"] {
    border-color: rgba(255, 255, 255, 0.1) !important;
}
[data-testid="stChatInput"] textarea {
    background: #2f2f2f !important;
    color: #ececec !important;
    border-radius: 12px !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
}
[data-testid="stChatInput"] textarea:focus {
    border-color: #667eea !important;
    box-shadow: 0 0 0 1px #667eea !important;
}

/* ---- Thinking / Spinner ---- */
.thinking-indicator {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.5rem 1rem;
    color: #888;
    font-size: 0.9rem;
}
.thinking-dots {
    display: inline-flex;
    gap: 4px;
}
.thinking-dots span {
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: #667eea;
    animation: bounce 1.4s infinite ease-in-out both;
}
.thinking-dots span:nth-child(1) { animation-delay: -0.32s; }
.thinking-dots span:nth-child(2) { animation-delay: -0.16s; }
.thinking-dots span:nth-child(3) { animation-delay: 0s; }
@keyframes bounce {
    0%, 80%, 100% { transform: scale(0.5); opacity: 0.4; }
    40% { transform: scale(1); opacity: 1; }
}

/* ---- Voice Controls ---- */
.voice-section {
    background: #2f2f2f;
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 14px;
    padding: 1rem 1.2rem;
    margin: 0.5rem 0;
}
.voice-status {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.3rem 0.8rem;
    border-radius: 20px;
    font-size: 0.8rem;
    font-weight: 600;
    background: rgba(102, 126, 234, 0.1);
    color: #667eea;
    margin: 0.3rem 0;
}
/* Mic recording animation */
.mic-recording {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.4rem 1rem;
    border-radius: 25px;
    font-size: 0.85rem;
    font-weight: 600;
    background: rgba(239, 68, 68, 0.12);
    color: #ef4444;
    border: 1px solid rgba(239, 68, 68, 0.25);
    animation: micPulse 1.5s ease-in-out infinite;
}
.mic-recording .mic-dot {
    width: 10px;
    height: 10px;
    border-radius: 50%;
    background: #ef4444;
    animation: micDotPulse 1s ease-in-out infinite;
}
@keyframes micPulse {
    0%, 100% { box-shadow: 0 0 0 0 rgba(239, 68, 68, 0.2); }
    50% { box-shadow: 0 0 0 8px rgba(239, 68, 68, 0); }
}
@keyframes micDotPulse {
    0%, 100% { opacity: 1; transform: scale(1); }
    50% { opacity: 0.5; transform: scale(0.75); }
}

/* Audio player styling */
audio {
    border-radius: 25px;
    margin: 0.5rem 0;
    filter: invert(0.85) hue-rotate(180deg);
}
audio::-webkit-media-controls-panel {
    background: #2f2f2f;
}

/* TTS toggle in sidebar */
[data-testid="stSidebar"] .stToggle label span {
    font-size: 0.85rem !important;
}

/* ---- Scrollbar (dark) ---- */
::-webkit-scrollbar {
    width: 6px;
}
::-webkit-scrollbar-track {
    background: #212121;
}
::-webkit-scrollbar-thumb {
    background: #444;
    border-radius: 3px;
}
::-webkit-scrollbar-thumb:hover {
    background: #555;
}
</style>
'''

bot_template = '''
<div class="chat-message bot">
    <div class="avatar">
        <img src="https://cdn-icons-png.flaticon.com/512/6134/6134346.png">
    </div>
    <div class="message">{{MSG}}</div>
</div>
'''

user_template = '''
<div class="chat-message user">
    <div class="message">{{MSG}}</div>
    <div class="avatar">
        <img src="https://cdn-icons-png.flaticon.com/512/1077/1077114.png">
    </div>
</div>
'''