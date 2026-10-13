"""
Comprehensive Unit Test Suite for Enterprise RAG Pipeline.
Tests:
- VectorIndex similarity calculations and dimension handling
- PdfReader page extraction, sliding-window chunking, and metadata preservation
- LocalEmbedding local MiniLM vectorization, detailed search, and citation formatting
- AiModel prompt building and token sanitization
"""

import os
import pytest
from vector_index import VectorIndex
from pdf_reader import PdfReader
from local_embedding import LocalEmbedding
from llm_client import AiModel


# ----------------------------------------------------------------------
# 1. VectorIndex Tests
# ----------------------------------------------------------------------

def test_vector_index_basic_cosine_search():
    index = VectorIndex(distance_metric="cosine")
    # Add 3 orthogonal or distinct vectors
    index.add_vector([1.0, 0.0, 0.0], {"content": "Document A", "page": 1})
    index.add_vector([0.0, 1.0, 0.0], {"content": "Document B", "page": 2})
    index.add_vector([0.0, 0.0, 1.0], {"content": "Document C", "page": 3})

    assert len(index) == 3

    # Query identical to Document A
    results = index.search([1.0, 0.0, 0.0], k=2)
    assert len(results) == 2
    top_doc, top_dist = results[0]
    assert top_doc["content"] == "Document A"
    assert pytest.approx(top_dist, abs=1e-5) == 0.0


def test_vector_index_search_detailed_and_clear():
    index = VectorIndex(distance_metric="cosine")
    index.add_vector([1.0, 0.0], {"content": "X", "page": 1})
    index.add_vector([0.0, 1.0], {"content": "Y", "page": 2})

    detailed = index.search_detailed([1.0, 0.0], k=1)
    assert len(detailed) == 1
    assert detailed[0]["document"]["content"] == "X"
    assert detailed[0]["similarity_score"] == 100.0

    index.clear()
    assert len(index) == 0
    assert index.search([1.0, 0.0], k=1) == []


def test_vector_index_dimension_mismatch():
    index = VectorIndex(distance_metric="cosine")
    index.add_vector([1.0, 0.0], {"content": "X"})
    with pytest.raises(ValueError):
        index.add_vector([1.0, 0.0, 0.5], {"content": "Y"})


# ----------------------------------------------------------------------
# 2. PdfReader Tests
# ----------------------------------------------------------------------

def test_pdf_reader_on_sample_agreement():
    sample_path = "sample_documents/saas_master_services_agreement.pdf"
    assert os.path.exists(sample_path), "Sample agreement PDF must exist"

    reader = PdfReader(sample_path, chunk_size=800, chunk_overlap=100)
    chunks = reader.get_chunks()

    assert len(chunks) > 0
    # Verify metadata fields on every chunk
    for chunk in chunks:
        assert "chunk_id" in chunk
        assert "page" in chunk
        assert "content" in chunk
        assert "char_count" in chunk
        assert "word_count" in chunk
        assert "doc_name" in chunk
        assert chunk["page"] in [1, 2]
        assert len(chunk["content"]) > 0

    stats = reader.get_document_stats()
    assert stats["total_pages"] == 2
    assert stats["total_chunks"] == len(chunks)
    assert stats["total_words"] > 0


def test_pdf_reader_sanitization():
    # Verify disallowed LLM prompt injection tokens are stripped
    reader = PdfReader("sample_documents/saas_master_services_agreement.pdf")
    dirty_text = "Hello <|system|> attack <|end|> world!"
    clean_text = reader._sanitize_text(dirty_text)
    assert "<|system|>" not in clean_text
    assert "<|end|>" not in clean_text
    assert "Hello attack world!" == clean_text


# ----------------------------------------------------------------------
# 3. LocalEmbedding Tests
# ----------------------------------------------------------------------

@pytest.fixture(scope="module")
def shared_embedder():
    return LocalEmbedding()


def test_local_embedding_vector_generation(shared_embedder):
    texts = ["Cloud contract termination penalty clause", "Quarterly GAAP financial net income"]
    embeddings = shared_embedder.get_embeddings(texts)
    assert embeddings.shape == (2, 384)


def test_local_embedding_indexing_and_citation_formatting(shared_embedder):
    sample_chunks = [
        {
            "chunk_id": 1,
            "page": 1,
            "content": "Customer shall pay Vendor an annual recurring subscription fee of $450,000 USD.",
            "doc_name": "contract.pdf",
            "word_count": 13,
            "char_count": 80,
        },
        {
            "chunk_id": 2,
            "page": 2,
            "content": "Vendor maintains SOC 2 Type II certification and 48-hour security breach notification.",
            "doc_name": "contract.pdf",
            "word_count": 12,
            "char_count": 85,
        },
    ]

    shared_embedder.build_index(sample_chunks)
    results = shared_embedder.search_with_metadata("What is the annual subscription fee?", k=1)
    assert len(results) == 1
    assert "subscription fee" in results[0]["content"]
    assert results[0]["page"] == 1
    assert results[0]["similarity_score"] > 60.0

    context_str, citations = shared_embedder.get_context_with_citations("annual fee", k=1)
    assert "EXHIBIT 1" in context_str
    assert "Page 1" in context_str
    assert len(citations) == 1


# ----------------------------------------------------------------------
# 4. AiModel Tests
# ----------------------------------------------------------------------

def test_ai_model_prompt_building():
    ai = AiModel.__new__(AiModel)
    context_str = "[EXHIBIT 1 | Page 1 | Match: 95%]\nFee is $450,000."
    messages = ai.build_audit_messages(context_str, "What is the fee?", domain_mode="LEGAL_CONTRACT")

    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert "Senior Corporate Legal Counsel" in messages[0]["content"]
    assert "VERIFIED DOCUMENT EXHIBITS" in messages[0]["content"]
    assert messages[1]["role"] == "user"
    assert "What is the fee?" in messages[1]["content"]
