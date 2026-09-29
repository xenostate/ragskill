from unittest.mock import MagicMock

import pytest

import scripts.config as cfg
from scripts.knowledge import index_answer_document


class _Embedding:
    def tolist(self):
        return [0.1, 0.2, 0.3]


def test_index_answer_document_stores_and_embeds_customer_answer(monkeypatch):
    document_insert = MagicMock()
    document_insert.execute.return_value = MagicMock(data=[{"id": 42}])
    chunk_insert = MagicMock()
    chunk_insert.execute.return_value = MagicMock(data=[{"id": 99}])
    documents = MagicMock()
    documents.insert.return_value = document_insert
    chunks = MagicMock()
    chunks.insert.return_value = chunk_insert
    sb = MagicMock()
    sb.table.side_effect = lambda name: documents if name == "documents" else chunks
    embed_model = MagicMock()
    embed_model.encode.return_value = [_Embedding()]
    monkeypatch.setattr(cfg, "sb", sb)
    monkeypatch.setattr(cfg, "embed_model", embed_model)

    result = index_answer_document(7, "Do you deliver?", "Yes, within two business days.")

    assert result["document_id"] == 42
    assert result["chunks"] == 1
    document = documents.insert.call_args.args[0]
    assert document["site_id"] == 7
    assert document["url"].startswith("answer://site-7/")
    chunk = chunks.insert.call_args.args[0][0]
    assert "Question: Do you deliver?" in chunk["text"]
    assert "Answer: Yes, within two business days." in chunk["text"]


@pytest.mark.parametrize("question,answer", [("", "Answer"), ("Question", "")])
def test_index_answer_document_requires_both_fields(question, answer):
    with pytest.raises(ValueError, match="required"):
        index_answer_document(7, question, answer)
