# Education Examination & Evaluation Process Explainer Bot

## Overview

A student-friendly AI assistant that explains academic examination and evaluation processes using uploaded institutional documents. Built with a RAG (Retrieval-Augmented Generation) pipeline, it retrieves relevant information from uploaded PDFs and provides clear, structured answers about exam patterns, grading systems, revaluation, supplementary exams, and more.

**This bot does NOT predict grades, solve exam questions, or provide model answers.**

---

## Integrated Components

### 1. Knowledge Base Creation

- **PDF Upload**: Upload exam regulation PDFs, syllabi, or academic handbooks via the sidebar.
- **Text Extraction**: `PyPDF2` extracts text from every page of each uploaded document.
- **OCR Fallback**: If a PDF is scanned/image-based (no extractable text), `pdf2image` converts pages to images and `pytesseract` performs OCR to extract text. Works for both text PDFs and image PDFs.
- **Text Chunking**: Extracted text is split into overlapping chunks using `CharacterTextSplitter`.
  - **Chunk Size**: 500 characters
  - **Chunk Overlap**: 50 characters
- **Embedding Generation**: Each chunk is vectorized using `sentence-transformers/all-MiniLM-L6-v2` via HuggingFace.
- **FAISS Vector Store**: Embeddings are indexed in FAISS for fast similarity search.
- **Persistence**: Knowledge base saved to disk and reloaded without re-uploading.

### 2. LLM Selection (Groq — Free)

| Model | ID | Description |
|-------|-----|-------------|
| **Llama 3.3 70B** (default) | `llama-3.3-70b-versatile` | Best quality |
| **Llama 3.1 8B** | `llama-3.1-8b-instant` | Fastest |
| **Mixtral 8x7B** | `mixtral-8x7b-32768` | Best for long context |
| **Gemma2 9B** | `gemma2-9b-it` | Lightweight |

- **SDK**: Groq Python SDK
- **API Key**: Loaded from `GROQ_API_KEY` in `.env` file

### 3. Prompt Configuration

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| **Temperature** | 0.2 | 0.0–1.0 | Lower = focused, Higher = creative |
| **Top-P** | 0.95 | 0.0–1.0 | Nucleus sampling threshold |
| **Max Tokens** | 2048 | 256–8192 | Max response length |

**System Prompt** strictly restricts the bot to:
- Explain examination patterns, grading, evaluation, revaluation, supplementary exams, attendance
- Answer in simple, student-friendly language with structured formatting
- **Never** predict grades, solve exam questions, provide model answers, or assist dishonesty

### 4. RAG with Vector DB

Full Retrieval-Augmented Generation pipeline:

1. **Retrieve**: FAISS similarity search finds top-K relevant chunks (configurable 1–10, default 2)
2. **Augment**: Chunks + system prompt + conversation history + question → structured prompt
3. **Generate**: Groq LLM generates context-grounded answer

### 5. Voice I/O

- **Voice Input**: Native Streamlit `st.audio_input` mic recorder → Google Speech Recognition transcription
- **Voice Output**: `pyttsx3` offline text-to-speech (no internet needed) — toggle "Read answers aloud" in sidebar

### 6. Academic Safety & Integrity

- Strict system prompt guardrails against misuse
- Academic Integrity Notice displayed in sidebar
- Bot refuses off-topic, grade prediction, or exam-solving requests
- Footer disclaimer: informational guidance only

---

## File Structure

| File | Description |
|------|-------------|
| `app.py` | Main Streamlit application |
| `htmlTemplates.py` | HTML/CSS templates for UI |
| `.env` | `GROQ_API_KEY` (not committed) |
| `requirements.txt` | Python dependencies |
| `DESCRIPTION.md` | This documentation file |
| `faiss_knowledge_base/` | Persisted FAISS vector store |

---

## How to Run

```bash
pip install -r requirements.txt
# Create .env with: GROQ_API_KEY=your_key_here
python -m streamlit run app.py
```

---

## Tech Stack

- **LLM**: Groq (Llama 3.3 70B Versatile)
- **Vector DB**: FAISS (`faiss-cpu`)
- **Embeddings**: HuggingFace `all-MiniLM-L6-v2`
- **Text Splitting**: LangChain `CharacterTextSplitter`
- **PDF Parsing**: PyPDF2 + OCR fallback (`pdf2image` + `pytesseract`)
- **Voice**: SpeechRecognition + pyttsx3
- **Web UI**: Streamlit
- **Environment**: python-dotenv
