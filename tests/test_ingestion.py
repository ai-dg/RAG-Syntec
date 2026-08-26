import logging

from langchain_core.documents import Document

from app.services.ingestion import check_document_content


def test_check_document_content_rejects_empty_document(caplog):
    empty_doc = Document(page_content="   ", metadata={"source": "empty.md"})

    with caplog.at_level(logging.WARNING):
        result = check_document_content(empty_doc)

    assert result is None
    assert "empty content" in caplog.text


def test_check_document_content_keeps_valid_document():
    doc = Document(page_content="  real content  ", metadata={"source": "ok.md"})

    result = check_document_content(doc)

    assert result is not None
    assert result.page_content == "real content"


def test_check_document_content_normalizes_line_endings():
    doc = Document(page_content="line one\r\nline two\r", metadata={"source": "ok.md"})

    result = check_document_content(doc)

    assert result.page_content == "line one\nline two"
