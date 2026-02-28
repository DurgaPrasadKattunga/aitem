# ============================================================
# Education Examination & Evaluation Process Explainer Bot
# ============================================================
# A student-friendly AI assistant that explains academic
# examination & evaluation processes using uploaded documents.
#
# Components:
#   1. Knowledge Base   - PDF upload, text extraction, chunking
#   2. LLM Selection    - Groq API (Llama 3.3 70B, Mixtral, Gemma2)
#   3. Prompt Config    - Academic system prompt, generation params
#   4. RAG + Vector DB  - FAISS vector store with HuggingFace embeddings
#   5. Voice I/O        - Speech-to-text input, text-to-speech output
#   6. Academic Safety  - Strict guardrails against misuse
# ============================================================

# ----- Imports -----
import os
import io
import time
import base64
import tempfile
from dotenv import load_dotenv
import streamlit as st
from PyPDF2 import PdfReader
from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from groq import Groq
from htmlTemplates import css, bot_template, user_template
import speech_recognition as sr
from gtts import gTTS

# OCR support (optional — install poppler + pytesseract to enable)
try:
    from pdf2image import convert_from_bytes
    import pytesseract
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False

# ============================================================
# 1. KNOWLEDGE BASE CONFIGURATION
# ============================================================
# Chunking parameters for building the knowledge base
CHUNK_SIZE = 500           # Number of characters per chunk (smaller = more precise retrieval)
CHUNK_OVERLAP = 50         # Overlap between consecutive chunks
CHUNK_SEPARATOR = "\n"     # Separator used to split text

# Embedding model for vectorizing text chunks
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DEVICE = "cpu"

# Number of relevant chunks to retrieve per query
RETRIEVAL_TOP_K = 2

# Local path to persist the FAISS vector store
VECTORSTORE_PERSIST_DIR = "faiss_knowledge_base"

# ============================================================
# 2. LLM SELECTION (Groq)
# ============================================================
# Available Groq models (all free)
AVAILABLE_MODELS = {
    "Llama 3.3 70B": "llama-3.3-70b-versatile",
    "Llama 3.1 8B (Fast)": "llama-3.1-8b-instant",
    "Mixtral 8x7B": "mixtral-8x7b-32768",
    "Gemma2 9B": "gemma2-9b-it",
}
DEFAULT_MODEL = "Llama 3.3 70B"

# ============================================================
# 3. PROMPT CONFIGURATION & GENERATION PARAMETERS
# ============================================================
# System prompt — strict academic examination explainer
SYSTEM_PROMPT = (
    "You are an Education Examination & Evaluation Process Assistant.\n\n"
    "=== STRICT DOCUMENT-GROUNDING RULES (HIGHEST PRIORITY) ===\n"
    "1. You MUST answer ONLY using information that is EXPLICITLY and LITERALLY present "
    "in the RETRIEVED DOCUMENT CONTEXT provided to you.\n"
    "2. INACCURATE / OUT-OF-RANGE VALUES: If the document defines a valid range or set "
    "of permitted values for something (e.g., marks 0\u2013100, specific grade letters, "
    "defined attendance percentages) AND the user asks about a value that FALLS OUTSIDE "
    "or is INCONSISTENT with that defined range, you MUST respond EXACTLY with:\n"
    "   \"\u274c Inaccurate value: According to the uploaded document, [topic] must be "
    "in the range [valid range / permitted values from the document]. "
    "The value you mentioned ([user's value]) is outside this defined range and is therefore invalid.\"\n"
    "   Fill in [topic], [valid range], and [user's value] with the actual values from "
    "the document and the user's query.\n"
    "   EXAMPLE: Document defines grades for marks 0\u2013100. User asks about 101 marks.\n"
    "   Correct response: \"\u274c Inaccurate value: According to the uploaded document, "
    "marks must be in the range 0\u2013100. The value you mentioned (101) is outside this "
    "defined range and is therefore invalid.\"\n"
    "3. MISSING INFORMATION: If the user's question asks about a value, scenario, or case "
    "that is simply NOT mentioned or defined anywhere in the document context, "
    "you MUST respond EXACTLY with:\n"
    "   \"\u26a0\ufe0f This information is not available in the uploaded document(s). "
    "Please refer to your institution's official regulations for this specific query.\"\n"
    "4. You MUST NOT infer, extrapolate, calculate, or assume any information "
    "beyond what is explicitly written in the context.\n"
    "5. You MUST NOT use your general training knowledge to fill gaps. "
    "If it is not in the document, it does not exist for you.\n"
    "6. Partial answers are not allowed. If only part of the question is covered "
    "in the document, answer only that part and explicitly state what is NOT covered.\n\n"
    "=== SCOPE ===\n"
    "Your purpose is to explain (from the document only):\n"
    "- Examination patterns and schedules\n"
    "- Internal and external evaluation methods\n"
    "- Grading systems (CGPA, GPA, letter grades, percentage)\n"
    "- Revaluation and recounting processes\n"
    "- Supplementary and improvement examinations\n"
    "- Attendance rules and eligibility criteria\n"
    "- Hall ticket and registration procedures\n"
    "- Result publication and transcript processes\n\n"
    "=== RESPONSE STYLE ===\n"
    "- Use simple, student-friendly language\n"
    "- Use clear headings, bullet points, or numbered steps\n"
    "- Be encouraging and supportive in tone\n\n"
    "=== ABSOLUTE PROHIBITIONS ===\n"
    "- NEVER predict, estimate, or calculate grades or marks\n"
    "- NEVER answer questions about values/scenarios not present in the document\n"
    "- NEVER solve exam questions or provide model answers\n"
    "- NEVER assist with academic dishonesty\n"
    "- NEVER discuss topics outside examination & evaluation processes\n"
    "- NEVER use knowledge from outside the provided document context\n\n"
    "If a user asks anything outside examination process explanation, reply:\n"
    "'I can only help explain examination and evaluation processes as described "
    "in the uploaded documents. Please ask about exam patterns, grading, "
    "revaluation, or similar topics.'\n"
)

# Default generation parameters (configurable via sidebar)
DEFAULT_TEMPERATURE = 0.0   # 0.0 = fully deterministic, prevents hallucination
DEFAULT_TOP_P = 0.95        # Nucleus sampling threshold
DEFAULT_MAX_TOKENS = 2048   # Maximum output tokens

# ============================================================
# KNOWLEDGE BASE FUNCTIONS
# ============================================================

def get_pdf_text(docs):
    """Extract text from all uploaded PDF files. Falls back to OCR for scanned/image PDFs."""
    text = ""
    for pdf in docs:
        pdf_bytes = pdf.read()
        pdf.seek(0)  # Reset for potential reuse

        # --- Try normal text extraction first ---
        pdf_reader = PdfReader(io.BytesIO(pdf_bytes))
        pdf_text = ""
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                pdf_text += page_text

        # --- Fallback to OCR if text extraction yielded nothing ---
        if not pdf_text.strip() and OCR_AVAILABLE:
            try:
                images = convert_from_bytes(pdf_bytes)
                ocr_text = ""
                for img in images:
                    page_ocr = pytesseract.image_to_string(img)
                    if page_ocr:
                        ocr_text += page_ocr + "\n"
                if ocr_text.strip():
                    pdf_text = ocr_text
                    st.info(f"📷 OCR applied to **{pdf.name}** — scanned PDF detected.")
                else:
                    st.warning(f"⚠️ Could not extract text from **{pdf.name}** (even with OCR).")
            except Exception as e:
                st.warning(f"⚠️ OCR failed for **{pdf.name}**: {e}")
        elif not pdf_text.strip():
            st.warning(f"⚠️ Could not extract text from **{pdf.name}**. Install pytesseract for OCR support.")

        text += pdf_text
    return text


def get_chunks(raw_text):
    """Split extracted text into overlapping chunks for the knowledge base."""
    text_splitter = CharacterTextSplitter(
        separator=CHUNK_SEPARATOR,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len
    )
    chunks = text_splitter.split_text(raw_text)
    return chunks


@st.cache_resource(show_spinner="Loading embedding model (one-time)...")
def get_embeddings():
    """Load and cache the HuggingFace embedding model. Only runs once."""
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={'device': EMBEDDING_DEVICE}
    )
    return embeddings


def get_vectorstore(chunks):
    """Create a FAISS vector store (knowledge base) from text chunks using cached embeddings."""
    embeddings = get_embeddings()
    vectorstore = FAISS.from_texts(texts=chunks, embedding=embeddings)
    return vectorstore


def save_vectorstore(vectorstore):
    """Persist the FAISS knowledge base to disk for reuse."""
    vectorstore.save_local(VECTORSTORE_PERSIST_DIR)
    return True


def load_vectorstore():
    """Load a previously saved FAISS knowledge base from disk."""
    if os.path.exists(VECTORSTORE_PERSIST_DIR):
        embeddings = get_embeddings()
        vectorstore = FAISS.load_local(
            VECTORSTORE_PERSIST_DIR,
            embeddings,
            allow_dangerous_deserialization=True
        )
        return vectorstore
    return None


# ============================================================
# LLM INITIALIZATION (Groq)
# ============================================================

def get_groq_client():
    """Initialize the Groq client."""
    return Groq(api_key=os.getenv("GROQ_API_KEY"))


# ============================================================
# VOICE I/O FUNCTIONS
# ============================================================

def transcribe_audio(audio_file):
    """Convert recorded audio to text using SpeechRecognition (Google Web API)."""
    recognizer = sr.Recognizer()
    try:
        # audio_file is a Streamlit UploadedFile (BytesIO-like)
        audio_file.seek(0)
        with sr.AudioFile(audio_file) as source:
            audio_data = recognizer.record(source)
        text = recognizer.recognize_google(audio_data)
        return text
    except sr.UnknownValueError:
        return None
    except sr.RequestError as e:
        st.warning(f"⚠️ Speech recognition service unavailable: {e}")
        return None
    except Exception as e:
        st.warning(f"⚠️ Audio processing error: {e}")
        return None


def text_to_speech(text):
    """Convert text to speech using gTTS and return MP3 audio bytes."""
    try:
        tts = gTTS(text=text[:500], lang="en", slow=False)
        buf = io.BytesIO()
        tts.write_to_fp(buf)
        buf.seek(0)
        return buf.read()
    except Exception as e:
        st.warning(f"⚠️ Text-to-speech error: {e}")
        return None


def get_audio_player_html(audio_bytes):
    """Generate an auto-play HTML audio element from audio bytes."""
    b64_audio = base64.b64encode(audio_bytes).decode()
    return f'<audio autoplay controls style="max-width:360px;height:34px;border-radius:12px;"><source src="data:audio/mp3;base64,{b64_audio}" type="audio/mp3"></audio>'


# ============================================================
# RAG - RETRIEVAL AUGMENTED GENERATION
# ============================================================

def get_relevant_context(vectorstore, question, k=RETRIEVAL_TOP_K):
    """Retrieve top-k relevant chunks from the FAISS vector DB for RAG."""
    docs = vectorstore.similarity_search(question, k=k)
    context_parts = []
    for i, doc in enumerate(docs, 1):
        context_parts.append(f"[Chunk {i}]:\n{doc.page_content}")
    return "\n\n".join(context_parts)


def build_rag_prompt(question, context, chat_history):
    """Build the full RAG prompt combining system prompt, context, history, and question."""
    # Build chat history string (trim to last 4 turns = 8 messages for speed)
    MAX_HISTORY_TURNS = 4
    history_text = ""
    if chat_history:
        recent_history = chat_history[-(MAX_HISTORY_TURNS * 2):]
        for role, text in recent_history:
            history_text += f"{role}: {text}\n"

    # Compose the full RAG prompt
    prompt = (
        f"### SYSTEM INSTRUCTIONS ###\n{SYSTEM_PROMPT}\n\n"
        f"### RETRIEVED DOCUMENT CONTEXT (from Knowledge Base) ###\n"
        f"--- START OF DOCUMENT CONTEXT ---\n{context}\n--- END OF DOCUMENT CONTEXT ---\n\n"
        f"### CONVERSATION HISTORY ###\n{history_text}\n"
        f"### USER QUESTION ###\n{question}\n\n"
        "### FINAL ANSWERING RULES ###\n"
        "Step 1: Search ONLY within the DOCUMENT CONTEXT above for information directly "
        "relevant to the question.\n"
        "Step 2: CHECK FOR INACCURATE / OUT-OF-RANGE VALUES FIRST: "
        "If the document defines a valid range or set of permitted values for the topic "
        "(e.g., marks 0\u2013100, specific attendance thresholds, defined grade letters), "
        "AND the user's question involves a value that falls OUTSIDE or violates that range, "
        "respond with EXACTLY:\n"
        "  '\u274c Inaccurate value: According to the uploaded document, [topic] must be "
        "in the range [valid range from document]. The value you mentioned ([user value]) "
        "is outside this defined range and is therefore invalid.'\n"
        "  (Replace bracketed placeholders with actual values.)\n"
        "Step 3: CHECK FOR MISSING INFORMATION: If the topic or scenario is simply not "
        "mentioned anywhere in the document context at all, respond with EXACTLY:\n"
        "  '\u26a0\ufe0f This specific information is not available in the uploaded document(s). "
        "Please refer to your institution\'s official regulations.'\n"
        "Step 4: Do NOT extrapolate, calculate, or infer. NEVER use external knowledge. "
        "Your ONLY source of truth is the document context above."
    )
    return prompt


def stream_response(client, model_id, system_prompt, user_prompt, temperature, top_p, max_tokens):
    """Stream response from Groq for typing animation effect."""
    stream = client.chat.completions.create(
        model=model_id,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        stream=True,
    )
    for chunk in stream:
        if chunk.choices[0].delta.content is not None:
            yield chunk.choices[0].delta.content


def handle_question(question):
    """Process user question through the full RAG pipeline: retrieve → prompt → generate."""
    vectorstore = st.session_state.vectorstore
    client = st.session_state.groq_client

    # RAG Step 1: Retrieve relevant context from vector DB knowledge base
    with st.spinner("🔍 Searching knowledge base..."):
        context = get_relevant_context(vectorstore, question, k=st.session_state.retrieval_k)

    # RAG Step 2: Build the augmented prompt
    prompt = build_rag_prompt(question, context, st.session_state.chat_history)

    # Display user message immediately
    with st.chat_message("user", avatar="👤"):
        st.markdown(question)

    # RAG Step 3: Generate response with typing animation (streaming)
    answer = None
    max_retries = 3
    for attempt in range(max_retries):
        try:
            with st.chat_message("assistant", avatar="🎓"):
                with st.spinner("Thinking..."):
                    # Small delay to show spinner before streaming starts
                    pass
                answer = st.write_stream(
                    stream_response(
                        client,
                        st.session_state.selected_model_id,
                        SYSTEM_PROMPT,
                        prompt,
                        st.session_state.temperature,
                        st.session_state.top_p,
                        st.session_state.max_tokens,
                    )
                )
            break
        except Exception as e:
            error_msg = str(e)
            if "429" in error_msg or "rate_limit" in error_msg.lower() or "quota" in error_msg.lower():
                if attempt < max_retries - 1:
                    wait_time = 10 * (attempt + 1)
                    st.warning(f"⏳ Rate limited. Waiting {wait_time}s before retry ({attempt + 1}/{max_retries})...")
                    time.sleep(wait_time)
                else:
                    st.error(
                        "❌ **Rate limit reached.** Please wait a moment and try again.\n\n"
                        "Free tier has rate limits. Try again shortly."
                    )
                    return
            elif "connection" in error_msg.lower() or "timeout" in error_msg.lower() or "connect" in error_msg.lower():
                if attempt < max_retries - 1:
                    wait_time = 5 * (attempt + 1)
                    st.warning(f"⏳ Connection issue. Retrying in {wait_time}s ({attempt + 1}/{max_retries})...")
                    time.sleep(wait_time)
                else:
                    st.error("❌ Connection error. Please check your internet and try again.")
                    return
            else:
                st.error(f"❌ Error: {error_msg}")
                return

    if answer is None:
        return

    # Update chat history
    st.session_state.chat_history.append(("user", question))
    st.session_state.chat_history.append(("assistant", answer))

    # Generate TTS for the bot answer if voice is enabled
    if st.session_state.get("voice_output_enabled", False):
        tts_audio = text_to_speech(answer)
        if tts_audio:
            st.session_state.last_tts_audio = tts_audio


# ============================================================
# MAIN APPLICATION
# ============================================================

def main():
    load_dotenv()
    st.set_page_config(
        page_title="Education Examination & Evaluation Process Explainer",
        page_icon="🎓",
        layout="wide"
    )

    st.write(css, unsafe_allow_html=True)

    # Initialize session state
    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None
    if "groq_client" not in st.session_state:
        st.session_state.groq_client = get_groq_client()
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "kb_doc_count" not in st.session_state:
        st.session_state.kb_doc_count = 0
    if "kb_chunk_count" not in st.session_state:
        st.session_state.kb_chunk_count = 0
    if "retrieval_k" not in st.session_state:
        st.session_state.retrieval_k = RETRIEVAL_TOP_K
    if "temperature" not in st.session_state:
        st.session_state.temperature = DEFAULT_TEMPERATURE
    if "top_p" not in st.session_state:
        st.session_state.top_p = DEFAULT_TOP_P
    if "max_tokens" not in st.session_state:
        st.session_state.max_tokens = DEFAULT_MAX_TOKENS
    if "selected_model_id" not in st.session_state:
        st.session_state.selected_model_id = AVAILABLE_MODELS[DEFAULT_MODEL]
    if "voice_output_enabled" not in st.session_state:
        st.session_state.voice_output_enabled = True
    if "last_tts_audio" not in st.session_state:
        st.session_state.last_tts_audio = None

    # ===== SIDEBAR =====
    with st.sidebar:
        # -- Branding --
        st.markdown('<div class="sidebar-title">🎓 Exam Explainer</div>', unsafe_allow_html=True)
        st.markdown('<div class="sidebar-subtitle">Education Examination & Evaluation Process Explainer Bot</div>', unsafe_allow_html=True)
        st.markdown("---")

        # ---- 📘 About This Bot ----
        st.markdown('<div class="section-header">📘 About This Bot</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="about-box">'
            'This bot helps students understand <b>examination and evaluation processes</b> '
            'from uploaded academic documents. Ask about grading, revaluation, '
            'supplementary exams, attendance rules, and more.'
            '</div>',
            unsafe_allow_html=True
        )
        st.markdown("---")

        # ---- STEP 1: Upload Academic Documents ----
        st.markdown('<div class="section-header">📄 Step 1 — Upload Academic Documents</div>', unsafe_allow_html=True)
        docs = st.file_uploader(
            "Upload exam regulation PDFs, syllabi, or academic handbooks",
            accept_multiple_files=True,
            type=["pdf"],
            label_visibility="collapsed"
        )
        if docs:
            st.caption(f"📎 {len(docs)} document(s) selected")

        col1, col2 = st.columns(2)
        with col1:
            build_btn = st.button("🔨 Build KB", use_container_width=True)
        with col2:
            load_btn = st.button("📂 Load KB", use_container_width=True)

        if build_btn:
            if not docs:
                st.warning("📄 Please upload at least one academic PDF first.")
            else:
                with st.spinner("⏳ Building Academic Knowledge Base..."):
                    raw_text = get_pdf_text(docs)
                    if not raw_text.strip():
                        st.error("Could not extract text from the PDFs.")
                    else:
                        text_chunks = get_chunks(raw_text)
                        vectorstore = get_vectorstore(text_chunks)
                        st.session_state.vectorstore = vectorstore
                        save_vectorstore(vectorstore)
                        st.session_state.kb_doc_count = len(docs)
                        st.session_state.kb_chunk_count = len(text_chunks)
                        st.success(
                            f"📚 **Academic Knowledge Base Successfully Built!**\n\n"
                            f"{len(docs)} document(s) processed into {len(text_chunks)} knowledge chunks."
                        )

        if load_btn:
            with st.spinner("Loading saved academic knowledge base..."):
                vectorstore = load_vectorstore()
                if vectorstore:
                    st.session_state.vectorstore = vectorstore
                    st.success("📚 Academic Knowledge Base Loaded Successfully!")
                else:
                    st.warning("No saved knowledge base found. Please upload and build first.")

        st.markdown("---")

        # ---- STEP 2: Choose AI Model ----
        st.markdown('<div class="section-header">🤖 Step 2 — Choose AI Model</div>', unsafe_allow_html=True)
        selected_model_name = st.selectbox(
            "Model",
            list(AVAILABLE_MODELS.keys()),
            index=list(AVAILABLE_MODELS.keys()).index(DEFAULT_MODEL),
            label_visibility="collapsed"
        )
        selected_model_id = AVAILABLE_MODELS[selected_model_name]

        st.markdown("---")

        # ---- STEP 3: Settings (collapsible) ----
        st.markdown('<div class="section-header">⚙️ Step 3 — Settings</div>', unsafe_allow_html=True)
        with st.expander("Generation Parameters", expanded=False):
            temperature = st.slider("Temperature", 0.0, 1.0, DEFAULT_TEMPERATURE, 0.05,
                                    help="Lower = precise answers, Higher = creative answers")
            top_p = st.slider("Top-P", 0.0, 1.0, DEFAULT_TOP_P, 0.05,
                               help="Nucleus sampling threshold")
            max_tokens = st.slider("Max Tokens", 256, 8192, DEFAULT_MAX_TOKENS, 256,
                                   help="Max response length")

        with st.expander("Retrieval Settings", expanded=False):
            retrieval_k = st.slider("Chunks to retrieve", 1, 10, RETRIEVAL_TOP_K, 1,
                                    help="More chunks = broader context, but slower")
            st.session_state.retrieval_k = retrieval_k

        # Store current settings in session state
        st.session_state.selected_model_id = selected_model_id
        st.session_state.temperature = temperature
        st.session_state.top_p = top_p
        st.session_state.max_tokens = max_tokens

        st.markdown("---")

        # ---- STEP 4: Voice Settings ----
        st.markdown('<div class="section-header">🎙️ Step 4 — Voice</div>', unsafe_allow_html=True)
        st.session_state.voice_output_enabled = st.toggle(
            "🔊 Read answers aloud",
            value=st.session_state.voice_output_enabled,
            help="Enable text-to-speech for bot responses"
        )

        st.markdown("---")

        # ---- 📊 Status Dashboard ----
        st.markdown('<div class="section-header">📊 Status</div>', unsafe_allow_html=True)

        if st.session_state.vectorstore is not None:
            st.markdown(
                '<span class="status-badge status-ready">● Academic Knowledge Base Ready</span>',
                unsafe_allow_html=True
            )
            if st.session_state.kb_doc_count > 0:
                st.markdown(
                    f'<div class="status-row">'
                    f'<span class="status-label">Documents</span>'
                    f'<span class="status-value">{st.session_state.kb_doc_count}</span>'
                    f'</div>'
                    f'<div class="status-row">'
                    f'<span class="status-label">Chunks</span>'
                    f'<span class="status-value">{st.session_state.kb_chunk_count}</span>'
                    f'</div>',
                    unsafe_allow_html=True
                )
        else:
            st.markdown(
                '<span class="status-badge status-waiting">○ No Knowledge Base — Upload PDFs above</span>',
                unsafe_allow_html=True
            )

        st.markdown(
            f'<div class="status-row">'
            f'<span class="status-label">Model</span>'
            f'<span class="status-value">{selected_model_name}</span>'
            f'</div>'
            f'<div class="status-row">'
            f'<span class="status-label">Temperature</span>'
            f'<span class="status-value">{temperature}</span>'
            f'</div>',
            unsafe_allow_html=True
        )

        st.markdown("---")

        # ---- ⚖️ Academic Integrity Notice ----
        st.markdown('<div class="section-header">⚖️ Academic Integrity Notice</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="integrity-notice">'
            '⚠️ This bot is designed <b>only to explain</b> examination and evaluation processes. '
            'It will <b>not</b>:<br>'
            '• Predict or estimate grades<br>'
            '• Solve exam questions<br>'
            '• Provide model answers<br>'
            '• Assist with academic dishonesty<br><br>'
            '<em>Use responsibly and ethically.</em>'
            '</div>',
            unsafe_allow_html=True
        )

        st.markdown("")
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.chat_history = []
            st.session_state.last_tts_audio = None
            st.rerun()

    # ===== MAIN CHAT AREA =====
    st.markdown(
        '<div class="main-header">'
        '<h1>🎓 Education Examination & Evaluation Process Explainer</h1>'
        '<p>Upload academic documents and ask about exam patterns, grading, revaluation, and more</p>'
        '</div>',
        unsafe_allow_html=True
    )

    # Show welcome message when no KB and no chat
    if st.session_state.vectorstore is None and not st.session_state.chat_history:
        st.markdown(
            '<div class="welcome-box">'
            '<h3>👋 Welcome, Student!</h3>'
            '<p>To get started:</p>'
            '<ol>'
            '<li>📄 <b>Upload</b> your exam regulation PDFs or academic handbooks in the sidebar</li>'
            '<li>🔨 <b>Build</b> the Academic Knowledge Base</li>'
            '<li>💬 <b>Ask</b> about examination patterns, grading systems, revaluation, and more</li>'
            '</ol>'
            '<p><em>Example questions you can ask:</em></p>'
            '<ul>'
            '<li>"How does the grading system work?"</li>'
            '<li>"What is the revaluation process?"</li>'
            '<li>"What are the attendance requirements for eligibility?"</li>'
            '<li>"How do supplementary exams work?"</li>'
            '</ul>'
            '</div>',
            unsafe_allow_html=True
        )

    # Show existing chat history using st.chat_message
    if st.session_state.chat_history:
        for role, text in st.session_state.chat_history:
            if role == "user":
                with st.chat_message("user", avatar="👤"):
                    st.markdown(text)
            else:
                with st.chat_message("assistant", avatar="🎓"):
                    st.markdown(text)

    # ---- Voice Input ----
    st.markdown("---")
    st.markdown(
        '<div class="voice-section">'
        '<strong>🎙️ Voice Input</strong> — Record a question using your microphone'
        '</div>',
        unsafe_allow_html=True
    )
    audio_file = st.audio_input("Record your question", key="voice_input", label_visibility="collapsed")

    voice_question = None
    if audio_file is not None:
        audio_id = audio_file.file_id
        if st.session_state.get("last_audio_id") != audio_id:
            st.session_state.last_audio_id = audio_id
            # Show recording animation while transcribing
            st.markdown(
                '<div class="mic-recording">'
                '<div class="mic-dot"></div> Transcribing your voice...'
                '</div>',
                unsafe_allow_html=True
            )
            voice_question = transcribe_audio(audio_file)
            if voice_question:
                st.success(f'🎙️ You said: "{voice_question}"')
            else:
                st.warning("Could not understand the audio. Please try again.")

    # ---- Text Input ----
    question = st.chat_input("Ask a question about your documents...")

    # Use voice question if no typed input
    active_question = question or voice_question

    if active_question:
        if st.session_state.vectorstore is None:
            st.warning("⚠️ Please upload PDFs and build a knowledge base first (see sidebar).")
        elif st.session_state.groq_client is None:
            st.warning("⚠️ Model not ready. Check your GROQ_API_KEY in the .env file.")
        else:
            handle_question(active_question)

            # Auto-play TTS audio if available
            if st.session_state.get("last_tts_audio"):
                st.markdown(
                    get_audio_player_html(st.session_state.last_tts_audio),
                    unsafe_allow_html=True
                )
                st.session_state.last_tts_audio = None

    # ===== FOOTER =====
    st.markdown("---")
    st.markdown(
        '<div class="footer">'
        '📜 <em>This system provides informational guidance only and does not replace '
        'official university regulations. Always refer to your institution\'s official '
        'documents for authoritative information.</em>'
        '</div>',
        unsafe_allow_html=True
    )


if __name__ == '__main__':
    main()