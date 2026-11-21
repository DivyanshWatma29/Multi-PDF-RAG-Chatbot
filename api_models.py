"""Small Gemini adapters for the existing LangChain/FAISS pipeline."""
from types import SimpleNamespace
from google import genai
from google.genai import types
from langchain_core.embeddings import Embeddings


# The tutorial compares OpenAI and Hugging Face. This adapter supplies Gemini
# embeddings for the same FAISS workflow. Ollama is wired separately in app.py.
class GeminiEmbeddings(Embeddings):
    def __init__(self, api_key):
        self.client = genai.Client(api_key=api_key, http_options=types.HttpOptions(timeout=60000))

    def _embed(self, texts, task):
        result = self.client.models.embed_content(
            model="gemini-embedding-001", contents=texts,
            config=types.EmbedContentConfig(task_type=task),
        )
        if not result.embeddings or len(result.embeddings) != len(texts):
            raise ValueError("The embedding service returned an incomplete result.")
        return [item.values for item in result.embeddings]

    def embed_documents(self, texts):
        # Build one vector per chunk in batches. Keep the vector order aligned
        # with the original chunks so each result points to the right page.
        vectors = []
        for start in range(0, len(texts), 20):
            vectors.extend(self._embed(texts[start:start + 20], "RETRIEVAL_DOCUMENT"))
        return vectors

    def embed_query(self, text):
        # A search query uses the same model with a query-specific task hint.
        return self._embed([text], "RETRIEVAL_QUERY")[0]


class GeminiChat:
    def __init__(self, api_key, model):
        self.client = genai.Client(api_key=api_key, http_options=types.HttpOptions(timeout=60000))
        self.model = model

    def invoke(self, prompt):
        # Send the finished prompt to Gemini. This adapter handles generation.
        # The calling code has already retrieved the relevant PDF excerpts.
        response = self.client.models.generate_content(
            model=self.model, contents=prompt,
            config=types.GenerateContentConfig(temperature=0.1, max_output_tokens=2048),
        )
        if not response.text:
            raise ValueError("The model did not return a text answer.")
        # Match the .content interface of Ollama's chat reply so the UI can
        # display either provider's answer in the same way.
        return SimpleNamespace(content=response.text)
