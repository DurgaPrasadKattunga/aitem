# 🎓 Exam Evaluation Explainer Bot

> A student-friendly AI assistant that explains academic examination and evaluation processes using your institution's official documents.

Built by **Malipeddi Sekhar** | AITAM | Error Squad | 6th Semester Project

---

## What It Does

Upload your exam regulation PDFs and ask questions in plain language. The bot retrieves the exact relevant sections from your documents and explains them clearly — covering grading systems, revaluation, supplementary exams, attendance rules, and more.

**What it will NOT do:** predict grades, solve exam questions, or provide model answers.

---

## Features

| Feature | Details |
|---|---|
| **RAG Pipeline** | FAISS vector search + Groq LLM for grounded, accurate answers |
| **Multi-PDF Support** | Upload multiple exam regulation documents at once |
| **OCR Support** | Handles scanned/image PDFs via Tesseract |
| **Voice Input** | Speak your question — transcribed via Groq Whisper (< 0.5s) |
| **Voice Output** | Answers read aloud via gTTS |
| **Model Selection** | Switch between Llama 3.3 70B, Mixtral, Gemma2 in the sidebar |
| **ChatGPT-style UI** | Dark theme, pill search bar, streaming responses |
| **Persistent KB** | Knowledge base saved to disk — no re-upload needed |

---

## Tech Stack

```
Frontend      → Streamlit
LLM           → Groq API  (llama-3.3-70b-versatile)
Voice STT     → Groq Whisper API  (whisper-large-v3-turbo)
Voice TTS     → gTTS
Embeddings    → HuggingFace  (sentence-transformers/all-MiniLM-L6-v2)
Vector DB     → FAISS
PDF Parsing   → PyPDF2 + pytesseract (OCR)
```

---

## Project Structure

```
ERROR SQUAD/
│
├── app.py                  ← Main Streamlit application (entry point)
├── htmlTemplates.py        ← All CSS / HTML UI templates
│
├── .env                    ← Your API key (not committed to git)
├── .env.example            ← Template — copy to .env and fill key
├── .gitignore              ← Git ignore rules
├── requirements.txt        ← Python dependencies
│
├── faiss_knowledge_base/   ← Generated at runtime (auto-created)
│   └── index.faiss         ← FAISS vector index
│
└── docs/
    ├── ARCHITECTURE.md     ← Full system architecture & RAG flow
    └── SETUP.md            ← Detailed installation & setup guide
```

---

## Quick Start

### 1. Clone & Set Up

```bash
git clone <repo-url>
cd "ERROR SQUAD"

python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS/Linux

pip install -r requirements.txt
```

### 2. Configure API Key

```bash
cp .env.example .env
# Open .env and set: GROQ_API_KEY=your_key_here
```

Get your free key → https://console.groq.com

### 3. Run

```bash
streamlit run app.py
```

Open → http://localhost:8501

### 4. Use

1. Open the **sidebar** → upload your exam regulation PDFs
2. Click **"Build Knowledge Base"** and wait for ✅
3. Type or **speak** your question in the search bar
4. Get clear, document-grounded answers instantly

---

## Available Models

| Model | Best For |
|---|---|
| Llama 3.3 70B *(default)* | Best quality answers |
| Llama 3.1 8B Instant | Fastest responses |
| Mixtral 8x7B | Long context documents |
| Gemma2 9B | Lightweight / low latency |

---

## Configuration (Sidebar)

| Setting | Default | Description |
|---|---|---|
| Temperature | 0.2 | Lower = focused; Higher = creative |
| Top-P | 0.95 | Nucleus sampling |
| Max Tokens | 2048 | Max response length |
| Top-K Retrieval | 2 | Chunks retrieved per query |

---

## Academic Safety

The bot enforces strict guardrails:
- Answers **only** from uploaded documents
- Refuses grade prediction, exam solving, or dishonesty assistance
- Displays academic integrity notice in sidebar
- Footer disclaimer on every page

---

## Developer

**Malipeddi Sekhar**
AITAM — Aditya Institute of Technology and Management
B.Tech — 6th Semester
Team: Error Squad

---

## Detailed Docs

- [Architecture & RAG Flow](docs/ARCHITECTURE.md)
- [Full Setup Guide](docs/SETUP.md)

