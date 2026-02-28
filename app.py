# ============================================================
# ChatPDF - RAG-based PDF Q&A with Groq (Llama 3.3)
# ============================================================
# Components:
#   1. Knowledge Base   - PDF upload, text extraction, chunking
#   2. LLM Selection    - Groq API (Llama 3.3 70B, Mixtral, Gemma2)
#   3. Prompt Config    - System prompt, temperature, top_p, max tokens
#   4. RAG + Vector DB  - FAISS vector store with HuggingFace embeddings
# ============================================================

# ----- Imports -----
import os
import time
from dotenv import load_dotenv
import streamlit as st
from PyPDF2 import PdfReader
from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from groq import Groq
from htmlTemplates import css, bot_template, user_template

# ============================================================
# 1. KNOWLEDGE BASE CONFIGURATION
# ============================================================
# Chunking parameters for building the knowledge base
CHUNK_SIZE = 1000          # Number of characters per chunk
CHUNK_OVERLAP = 200        # Overlap between consecutive chunks
CHUNK_SEPARATOR = "\n"     # Separator used to split text

# Embedding model for vectorizing text chunks
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DEVICE = "cpu"

# Number of relevant chunks to retrieve per query
RETRIEVAL_TOP_K = 4

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
# System prompt restricting model to academic exam processes only
SYSTEM_PROMPT = (
    "You are an academic exam assistant. You must ONLY explain academic exam processes "
    "such as exam registration, hall tickets, exam schedules, evaluation procedures, "
    "revaluation, grading systems, result publication, supplementary exams, and related "
    "academic procedures. If a question is not related to academic exam processes, "
    "politely decline and remind the user of your scope.\n\n"
    "Rules:\n"
    "- Always base your answer on the provided document context.\n"
    "- If the context does not contain the answer, clearly state that.\n"
    "- Be concise, accurate, and helpful.\n"
    "- Use bullet points or numbered lists when appropriate.\n"
)

# Default generation parameters (configurable via sidebar)
DEFAULT_TEMPERATURE = 0.3   # Controls randomness (0.0 = deterministic, 1.0 = creative)
DEFAULT_TOP_P = 0.95        # Nucleus sampling threshold
DEFAULT_MAX_TOKENS = 2048   # Maximum output tokens

# ============================================================
# KNOWLEDGE BASE FUNCTIONS
# ============================================================

def get_pdf_text(docs):
    """Extract text from all uploaded PDF files to build the knowledge base."""
    text = ""
    for pdf in docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
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
    # Build chat history string
    history_text = ""
    if chat_history:
        for role, text in chat_history:
            history_text += f"{role}: {text}\n"

    # Compose the full RAG prompt
    prompt = (
        f"### SYSTEM INSTRUCTIONS ###\n{SYSTEM_PROMPT}\n\n"
        f"### RETRIEVED DOCUMENT CONTEXT (from Knowledge Base) ###\n{context}\n\n"
        f"### CONVERSATION HISTORY ###\n{history_text}\n"
        f"### USER QUESTION ###\n{question}\n\n"
        "### INSTRUCTIONS ###\n"
        "Answer the question based ONLY on the provided document context above. "
        "If the answer is not in the context, clearly state that you don't have "
        "enough information from the uploaded documents. "
        "Do not make up information."
    )
    return prompt


def handle_question(question):
    """Process user question through the full RAG pipeline: retrieve → prompt → generate."""
    vectorstore = st.session_state.vectorstore
    client = st.session_state.groq_client

    # RAG Step 1: Retrieve relevant context from vector DB knowledge base
    context = get_relevant_context(vectorstore, question, k=st.session_state.retrieval_k)

    # RAG Step 2: Build the augmented prompt
    prompt = build_rag_prompt(question, context, st.session_state.chat_history)

    # RAG Step 3: Generate response using Groq (with retry for rate limits)
    answer = None
    max_retries = 3
    for attempt in range(max_retries):
        try:
            with st.spinner("Generating answer..." if attempt == 0 else f"Rate limited — retrying ({attempt}/{max_retries})..."):
                response = client.chat.completions.create(
                    model=st.session_state.selected_model_id,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=st.session_state.temperature,
                    top_p=st.session_state.top_p,
                    max_tokens=st.session_state.max_tokens,
                )
                answer = response.choices[0].message.content
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
                        "xAI free tier has rate limits. Try again shortly."
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
    st.session_state.chat_history.append(("User", question))
    st.session_state.chat_history.append(("Bot", answer))

    # Display full chat history
    for role, text in st.session_state.chat_history:
        if role == "User":
            st.write(user_template.replace("{{MSG}}", text), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace("{{MSG}}", text), unsafe_allow_html=True)


# ============================================================
# MAIN APPLICATION
# ============================================================

def main():
    load_dotenv()
    st.set_page_config(page_title="ChatPDF - Academic Exam Assistant", page_icon="🎓", layout="wide")
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

    # ===== SIDEBAR =====
    with st.sidebar:
        # -- Branding --
        st.markdown('<div class="sidebar-title">🎓 ChatPDF</div>', unsafe_allow_html=True)
        st.markdown('<div class="sidebar-subtitle">Academic Exam Assistant</div>', unsafe_allow_html=True)
        st.markdown("---")

        # ---- STEP 1: Upload Documents ----
        st.markdown('<div class="section-header">📄 Step 1 — Upload Documents</div>', unsafe_allow_html=True)
        docs = st.file_uploader(
            "Drop your PDFs here",
            accept_multiple_files=True,
            type=["pdf"],
            label_visibility="collapsed"
        )
        if docs:
            st.caption(f"📎 {len(docs)} file(s) selected")

        col1, col2 = st.columns(2)
        with col1:
            build_btn = st.button("🔨 Build KB", use_container_width=True)
        with col2:
            load_btn = st.button("📂 Load KB", use_container_width=True)

        if build_btn:
            if not docs:
                st.warning("Upload at least one PDF first.")
            else:
                with st.spinner("⏳ Building knowledge base..."):
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
                        st.success(f"✅ {len(docs)} PDF(s) → {len(text_chunks)} chunks")

        if load_btn:
            with st.spinner("Loading saved knowledge base..."):
                vectorstore = load_vectorstore()
                if vectorstore:
                    st.session_state.vectorstore = vectorstore
                    st.success("✅ Knowledge base loaded!")
                else:
                    st.warning("No saved knowledge base found.")

        st.markdown("---")

        # ---- STEP 2: Choose Model ----
        st.markdown('<div class="section-header">🤖 Step 2 — Choose Model</div>', unsafe_allow_html=True)
        selected_model_name = st.selectbox(
            "Model",
            list(AVAILABLE_MODELS.keys()),
            index=list(AVAILABLE_MODELS.keys()).index(DEFAULT_MODEL),
            label_visibility="collapsed"
        )
        selected_model_id = AVAILABLE_MODELS[selected_model_name]

        st.markdown("---")

        # ---- STEP 3: Tune Settings (collapsible) ----
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

        # ---- Status Dashboard ----
        st.markdown('<div class="section-header">📊 Status</div>', unsafe_allow_html=True)

        # Knowledge base status
        if st.session_state.vectorstore is not None:
            st.markdown(
                '<span class="status-badge status-ready">● Knowledge Base Ready</span>',
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
                '<span class="status-badge status-waiting">○ No Knowledge Base</span>',
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

        st.markdown("")
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()

    # ===== MAIN CHAT AREA =====
    st.markdown(
        '<div class="main-header">'
        '<h1>🎓 Academic Exam Assistant</h1>'
        '<p>Upload your exam PDFs and ask anything about academic exam processes</p>'
        '</div>',
        unsafe_allow_html=True
    )

    # Show existing chat history
    if st.session_state.chat_history:
        for role, text in st.session_state.chat_history:
            if role == "User":
                st.write(user_template.replace("{{MSG}}", text), unsafe_allow_html=True)
            else:
                st.write(bot_template.replace("{{MSG}}", text), unsafe_allow_html=True)

    # Chat input
    question = st.chat_input("Ask a question about your documents...")
    if question:
        if st.session_state.vectorstore is None:
            st.warning("⚠️ Please upload PDFs and build a knowledge base first (see sidebar).")
        elif st.session_state.groq_client is None:
            st.warning("⚠️ Model not ready. Check your GROQ_API_KEY in the .env file.")
        else:
            handle_question(question)


if __name__ == '__main__':
    main()