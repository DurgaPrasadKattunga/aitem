# ChatPDF - RAG-based PDF Q&A with Gemini Flash

## Overview

ChatPDF is a Streamlit-based application that allows users to upload PDF documents and ask questions about their content using Google's Gemini Flash LLM. It uses a Retrieval-Augmented Generation (RAG) pipeline with a FAISS vector database to provide accurate, context-grounded answers restricted to academic exam processes.

---

## Integrated Components

### 1. Knowledge Base Creation

- **PDF Upload**: Users upload one or more PDFs via the sidebar.
- **Text Extraction**: `PyPDF2` extracts text from every page of each uploaded document.
- **Text Chunking**: The extracted text is split into overlapping chunks using `CharacterTextSplitter` from LangChain.
  - **Chunk Size**: 1000 characters (configurable via `CHUNK_SIZE`)
  - **Chunk Overlap**: 200 characters (configurable via `CHUNK_OVERLAP`)
- **Embedding Generation**: Each chunk is converted into a vector embedding using the `sentence-transformers/all-MiniLM-L6-v2` model via HuggingFace.
- **FAISS Vector Store**: Embeddings are indexed in a FAISS vector database for fast similarity search.
- **Persistence**: The knowledge base can be saved to disk (`faiss_knowledge_base/`) and reloaded later without re-uploading PDFs.

### 2. LLM Selection

- **Model Options**: Users can select from multiple Gemini models via a dropdown in the sidebar:
  - **Gemini 1.5 Flash** (default) — fast and efficient
  - **Gemini 1.5 Pro** — higher quality, slower
  - **Gemini 2.0 Flash** — latest flash model
- **SDK**: Uses `google-generativeai` Python SDK.
- **API Key**: Loaded from `GOOGLE_API_KEY` in `.env` file using `python-dotenv`.

### 3. Prompt Configuration Parameters

All generation parameters are **configurable via the sidebar** in real-time:

| Parameter         | Default | Range        | Description                                              |
|-------------------|---------|--------------|----------------------------------------------------------|
| **Temperature**   | 0.3     | 0.0 – 1.0   | Controls randomness. Lower = focused, Higher = creative   |
| **Top-K**         | 40      | 1 – 100      | Number of top tokens considered at each generation step    |
| **Top-P**         | 0.95    | 0.0 – 1.0   | Nucleus sampling cumulative probability threshold          |
| **Max Tokens**    | 2048    | 256 – 8192   | Maximum number of tokens in the generated response         |

**System Prompt**: A hardcoded system instruction restricts the model to ONLY answer questions about academic exam processes (registration, hall tickets, schedules, evaluation, grading, results, revaluation, supplementary exams, etc.). Any off-topic questions are politely declined.

### 4. RAG with Vector DB

The application implements a full **Retrieval-Augmented Generation (RAG)** pipeline:

1. **Retrieve**: When a user asks a question, the FAISS vector store performs a similarity search to find the top-K most relevant text chunks from the knowledge base.
   - **Retrieval Top-K** is configurable (1–10, default 4) via the sidebar.
2. **Augment**: The retrieved chunks are combined with the system prompt, conversation history, and user question into a structured RAG prompt.
3. **Generate**: The augmented prompt is sent to the selected Gemini model via `model.generate_content()`, which returns a context-grounded answer.

**RAG Prompt Structure**:
```
### SYSTEM INSTRUCTIONS ###
[Academic exam assistant rules]

### RETRIEVED DOCUMENT CONTEXT (from Knowledge Base) ###
[Chunk 1]: ...
[Chunk 2]: ...

### CONVERSATION HISTORY ###
User: ...
Bot: ...

### USER QUESTION ###
[Current question]

### INSTRUCTIONS ###
Answer based ONLY on the provided context...
```

---

## File Structure

| File                  | Description                                            |
|-----------------------|--------------------------------------------------------|
| `app.py`              | Main Streamlit application with all integrated components |
| `htmlTemplates.py`    | HTML/CSS templates for the chat UI                      |
| `.env`                | Environment file containing `GOOGLE_API_KEY`            |
| `requirements.txt`    | Python dependencies                                     |
| `DESCRIPTION.md`      | This file — documentation of all components             |
| `faiss_knowledge_base/` | Persisted FAISS vector store (created after first build) |

---

## How to Run

```bash
pip install -r requirements.txt
python -m streamlit run app.py
```

---

## Environment Variables

| Variable          | Description                     |
|-------------------|---------------------------------|
| `GOOGLE_API_KEY`  | Google AI API key for Gemini    |

Set in `.env` file at the project root:
```
GOOGLE_API_KEY=your_api_key_here
```

---

## Tech Stack

- **LLM**: Google Gemini 1.5 Flash (via `google-generativeai` SDK)
- **Vector DB**: FAISS (`faiss-cpu`)
- **Embeddings**: HuggingFace `all-MiniLM-L6-v2`
- **Text Splitting**: LangChain `CharacterTextSplitter`
- **PDF Parsing**: PyPDF2
- **Web UI**: Streamlit
- **Environment**: python-dotenv
