# Multi-PDF RAG Chatbot

A simple academic project to ask questions from multiple PDF files. The online version uses the Gemini API. Ollama is also available for local use.

## Technologies
Python, Streamlit, LangChain, FAISS, Google Gemini API, and optional Ollama.

## How it works
1. Upload PDFs and click **Process PDFs**.
2. Read the PDF text and split it into smaller chunks.
3. Create embeddings with Gemini and store them in FAISS.
4. Find the chunks related to your question.
5. Send those chunks to Gemini to generate an answer, with source names and page numbers.

## Run locally
Use Python 3.11:

```bash
pip install -r requirements.txt
streamlit run app.py
```

Choose **Gemini API** and enter a key from [Google AI Studio](https://aistudio.google.com/apikey) in the sidebar. The key is held in your current session. Alternatively, set the `GEMINI_API_KEY` environment variable or create `.streamlit/secrets.toml` using the example file. The chat model defaults to `gemini-2.5-flash`; override `GEMINI_CHAT_MODEL` if needed for your account.

## Deploy a live demo
1. Sign in at [Streamlit Community Cloud](https://share.streamlit.io/) with GitHub.
2. Click **Create app** and choose:
   - Repository: `DivyanshWatma29/Multi-PDF-RAG-Chatbot`
   - Branch: `main`
   - Main file: `app.py`
3. Under **Advanced settings**, select Python 3.11.
4. Deploy. Visitors can enter their own Gemini key in the sidebar.
5. For a demo that works without visitors entering a key, add this in the app's **Secrets** settings:

```toml
GEMINI_API_KEY = "your-key-here"
GEMINI_CHAT_MODEL = "gemini-2.5-flash"
```

Never commit your actual API key. A shared owner key uses the owner's quota and may incur charges if billing is enabled. This academic app has no global usage limiter; visitor-supplied keys are the default approach.

The deployment creates a `streamlit.app` URL that can be added to this README and the repository's About section. No live deployment has been verified yet.

## Ollama option
Install [Ollama](https://ollama.com/download), then run:

```bash
ollama pull llama3.2:3b
ollama pull nomic-embed-text
```

Select **Ollama (local)** in the app. The app owner can set `OLLAMA_BASE_URL`, `OLLAMA_CHAT_MODEL`, and `OLLAMA_EMBED_MODEL` as environment variables. The default server is `http://localhost:11434` on the machine hosting Streamlit.

## Limits and privacy
- Up to 5 PDFs, 10 MB total, and 200 text chunks per upload.
- Scanned PDFs without selectable text need OCR and are not supported.
- In Gemini mode, extracted text, questions, and recent chat context are sent to Google for processing. Use non-sensitive sample PDFs for the demo.
- Uploaded files and the index are held in the current session; the app does not write them to disk.
- Retrieved passages may be unrelated, and generated answers can be wrong. Check the source PDFs.
- API availability and rate limits depend on the provider and account.

## Tests
```bash
pip install pytest
pytest -q
```

## Project reference
Learning project based on [Alejandro AO's Ask Multiple PDFs tutorial](https://github.com/alejandro-ao/ask-multiple-pdfs), adapted to support Ollama and Gemini. Credit to the original author.
