# Multi-PDF RAG Chatbot

A simple academic project to ask questions from multiple PDF files using a local AI model.

## Technologies
- Python
- Streamlit (user interface)
- LangChain (text splitting and embeddings integration)
- FAISS (finding relevant PDF text)
- Ollama (running the AI model locally)

## How it works
1. Upload one or more PDF files.
2. Click **Process PDFs**.
3. The app reads the PDF text and splits it into smaller parts.
4. Ollama creates embeddings, and FAISS finds text related to your question.
5. Ollama uses that text to answer your question.

## Run on your computer

Install Python 3.11 and [Ollama](https://ollama.com/download). Start Ollama, then download the models:

```bash
ollama pull llama3.2:3b
ollama pull nomic-embed-text
```

Install the Python packages:

```bash
pip install -r requirements.txt
```

Start the app:

```bash
streamlit run app.py
```

Open the local address shown in the terminal. Upload PDFs, click **Process PDFs**, and start asking questions.

**Note:** Ollama must be running on your computer. This project does not have a public live demo yet. For a hosted demo, you would need a separate Ollama server or another model API. Scanned PDFs without selectable text are not supported.

## Project reference

This is a learning project based on [Alejandro AO's Ask Multiple PDFs tutorial](https://github.com/alejandro-ao/ask-multiple-pdfs), with changes to use Ollama locally instead of OpenAI. Credit to the original author.
