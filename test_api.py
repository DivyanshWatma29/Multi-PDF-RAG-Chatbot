from types import SimpleNamespace
from unittest.mock import MagicMock, patch
import pytest
from streamlit.testing.v1 import AppTest
from api_models import GeminiEmbeddings, GeminiChat


def test_embedding_batches_and_query_task():
    with patch('api_models.genai.Client') as client:
        models = client.return_value.models
        models.embed_content.side_effect = lambda **kw: SimpleNamespace(
            embeddings=[SimpleNamespace(values=[1., 0., 0.]) for _ in kw['contents']])
        embeddings = GeminiEmbeddings('test-key')
        assert len(embeddings.embed_documents(['text'] * 21)) == 21
        assert embeddings.embed_query('question') == [1., 0., 0.]
        assert [c.kwargs['config'].task_type for c in models.embed_content.call_args_list] == [
            'RETRIEVAL_DOCUMENT', 'RETRIEVAL_DOCUMENT', 'RETRIEVAL_QUERY']


def test_chat_rejects_empty_response():
    with patch('api_models.genai.Client') as client:
        client.return_value.models.generate_content.return_value = SimpleNamespace(text=None)
        with pytest.raises(ValueError):
            GeminiChat('test-key', 'test-model').invoke('question')


def test_app_starts_without_secrets_and_requires_key():
    app = AppTest.from_file('app.py').run(timeout=30)
    assert not app.exception
    assert app.selectbox[0].value == 'Gemini API'
    assert app.chat_input[0].disabled
    app.button[0].click().run()
    assert 'API key' in app.warning[0].value
    assert not app.exception
