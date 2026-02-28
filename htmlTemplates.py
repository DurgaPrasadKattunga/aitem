css = '''
<style>
/* ---- Global & Sidebar ---- */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
}
[data-testid="stSidebar"] * {
    color: #e0e0e0 !important;
}
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stSlider label,
[data-testid="stSidebar"] .stFileUploader label {
    font-size: 0.85rem;
    font-weight: 500;
    color: #b0b0c0 !important;
}
[data-testid="stSidebar"] hr {
    border-color: rgba(255,255,255,0.08);
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
    border-radius: 8px;
    font-weight: 600;
    font-size: 0.85rem;
    padding: 0.5rem 1rem;
    transition: all 0.2s ease;
    border: none;
}
[data-testid="stSidebar"] .stButton > button:first-of-type {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white !important;
}
[data-testid="stSidebar"] .stButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
}

/* ---- Main Area ---- */
.main-header {
    text-align: center;
    padding: 1rem 0 0.5rem 0;
}
.main-header h1 {
    font-size: 1.8rem;
    font-weight: 700;
    color: #667eea;
}
.main-header p {
    font-size: 0.9rem;
    color: #888;
    margin-top: -0.5rem;
}

/* Welcome Box */
.welcome-box {
    background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
    border: 1px solid rgba(102, 126, 234, 0.2);
    border-radius: 16px;
    padding: 1.8rem 2rem;
    margin: 1.5rem 0;
    color: #e0e0e0;
}
.welcome-box h3 {
    color: #667eea;
    font-size: 1.3rem;
    margin-bottom: 0.8rem;
}
.welcome-box p, .welcome-box li {
    font-size: 0.92rem;
    line-height: 1.7;
    color: #c8c8e8;
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
    color: #888;
    background: rgba(30, 41, 59, 0.3);
    border-radius: 10px;
    margin-top: 0.5rem;
}

/* ---- Chat Messages ---- */
.chat-message {
    padding: 1rem 1.2rem;
    border-radius: 12px;
    margin-bottom: 0.8rem;
    display: flex;
    align-items: flex-start;
    animation: fadeIn 0.3s ease;
}
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(8px); }
    to { opacity: 1; transform: translateY(0); }
}
.chat-message.user {
    background: linear-gradient(135deg, #2b313e 0%, #343b4a 100%);
    border-left: 3px solid #667eea;
}
.chat-message.bot {
    background: linear-gradient(135deg, #3a3f52 0%, #475063 100%);
    border-left: 3px solid #2ecc71;
}
.chat-message .avatar {
    width: 40px;
    min-width: 40px;
    margin-right: 1rem;
}
.chat-message.user .avatar {
    order: 2;
    margin-right: 0;
    margin-left: 1rem;
}
.chat-message .avatar img {
    width: 40px;
    height: 40px;
    border-radius: 50%;
    object-fit: cover;
}
.chat-message .message {
    flex: 1;
    padding: 0;
    color: #e8e8e8;
    font-size: 0.95rem;
    line-height: 1.6;
}
.chat-message.user .message {
    text-align: right;
    color: #c8c8e8;
}

/* ---- Voice Controls ---- */
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
.voice-status.recording {
    background: rgba(231, 76, 60, 0.15);
    color: #e74c3c;
    animation: pulse 1.2s infinite;
}
@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.6; }
}
/* Audio player styling */
audio {
    border-radius: 20px;
    margin: 0.5rem 0;
}
/* TTS toggle in sidebar */
[data-testid="stSidebar"] .stToggle label span {
    font-size: 0.85rem !important;
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