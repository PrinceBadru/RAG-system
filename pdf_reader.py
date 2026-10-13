"""
Enterprise Document Ingestion & Chunking Engine.

Extracts text page-by-page from PDFs, tracking exact page numbers and metadata.
Applies semantic sliding-window chunking to prevent token truncation bugs
in embedding models while preserving clause/paragraph boundaries.
"""

import os
import re
from typing import List, Dict, Any, Optional
import pypdf


# Special model control tokens to sanitize against prompt injection
DISALLOWED_TOKENS = [
    "<|end|>",
    "<|system|>",
    "<|user|>",
    "<|assistant|>",
    "<|endoftext|>",
    "<|im_start|>",
    "<|im_end|>",
]


class PdfReader:
    def __init__(
        self,
        path: str,
        chunk_size: int = 1200,     # ~250-300 tokens (safe under 512 token limit)
        chunk_overlap: int = 200,    # ~40-50 tokens overlap for semantic continuity
    ):
        """
        Initialize the enterprise PDF reader with page tracking and smart chunking.

        Args:
            path: Absolute or relative path to the PDF file.
            chunk_size: Target character length per chunk.
            chunk_overlap: Number of characters to overlap between consecutive chunks.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"PDF file not found at: {path}")

        self.path = path
        self.doc_name = os.path.basename(path)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        self.reader = pypdf.PdfReader(path)
        self.total_pages = len(self.reader.pages)
        self.pages_data: List[Dict[str, Any]] = []  # List of {"page": int, "text": str}
        self.chunks: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Text sanitization & extraction
    # ------------------------------------------------------------------

    def _sanitize_text(self, text: str) -> str:
        """
        Clean raw text: normalizes whitespace and removes potential LLM prompt injection tokens.
        """
        if not text:
            return ""

        # Strip disallowed model control tokens
        for token in DISALLOWED_TOKENS:
            text = text.replace(token, "")

        # Normalize line breaks and spaces
        text = re.sub(r"\r\n|\r", "\n", text)
        # Collapse multiple empty lines into double newline (paragraph boundary)
        text = re.sub(r"\n{3,}", "\n\n", text)
        # Collapse horizontal whitespace
        text = re.sub(r"[ \t]+", " ", text)
        return text.strip()

    def extract_text(self) -> str:
        """
        Extract text from all pages while recording page-level boundaries.
        Returns the concatenated sanitized text.
        """
        self.pages_data = []
        combined_text_list = []

        for page_idx, page in enumerate(self.reader.pages):
            page_num = page_idx + 1
            raw_text = page.extract_text() or ""
            clean_text = self._sanitize_text(raw_text)

            if clean_text:
                self.pages_data.append({
                    "page": page_num,
                    "text": clean_text,
                })
                combined_text_list.append(clean_text)

        full_text = "\n\n".join(combined_text_list)
        return full_text

    # ------------------------------------------------------------------
    # Semantic Sliding-Window Chunker
    # ------------------------------------------------------------------

    def _split_into_sentences(self, text: str) -> List[str]:
        """
        Splits text on sentence boundaries or paragraph breaks.
        """
        # Split on paragraph breaks first or sentence terminators (. ? !)
        pattern = r"(?<=[.?!])\s+(?=[A-Z0-9])|\n\n"
        sentences = re.split(pattern, text)
        return [s.strip() for s in sentences if s.strip()]

    def get_chunks(self) -> List[Dict[str, Any]]:
        """
        Chunks the document page-by-page using a sliding window with overlap.
        Preserves page attribution and prevents truncation in embedding models.

        Returns:
            List of dictionaries:
            {
                "chunk_id": int,
                "page": int,
                "content": str,
                "char_count": int,
                "word_count": int,
                "doc_name": str,
            }
        """
        if not self.pages_data:
            self.extract_text()

        self.chunks = []
        chunk_counter = 1

        for page_entry in self.pages_data:
            page_num = page_entry["page"]
            page_text = page_entry["text"]

            if len(page_text) <= self.chunk_size:
                # Page is small enough to be a single chunk
                self.chunks.append({
                    "chunk_id": chunk_counter,
                    "page": page_num,
                    "content": page_text,
                    "char_count": len(page_text),
                    "word_count": len(page_text.split()),
                    "doc_name": self.doc_name,
                })
                chunk_counter += 1
                continue

            # Page is larger than chunk_size: use sliding window with overlap
            sentences = self._split_into_sentences(page_text)
            current_chunk_sentences = []
            current_length = 0

            for sentence in sentences:
                sentence_len = len(sentence)
                if current_length + sentence_len > self.chunk_size and current_chunk_sentences:
                    chunk_text = " ".join(current_chunk_sentences).strip()
                    if chunk_text:
                        self.chunks.append({
                            "chunk_id": chunk_counter,
                            "page": page_num,
                            "content": chunk_text,
                            "char_count": len(chunk_text),
                            "word_count": len(chunk_text.split()),
                            "doc_name": self.doc_name,
                        })
                        chunk_counter += 1

                    # Slide window with overlap: keep the tail sentences
                    overlap_sentences = []
                    overlap_len = 0
                    for s in reversed(current_chunk_sentences):
                        if overlap_len + len(s) <= self.chunk_overlap:
                            overlap_sentences.insert(0, s)
                            overlap_len += len(s)
                        else:
                            break

                    current_chunk_sentences = overlap_sentences + [sentence]
                    current_length = sum(len(s) for s in current_chunk_sentences) + len(current_chunk_sentences)
                else:
                    current_chunk_sentences.append(sentence)
                    current_length += sentence_len + 1

            # Append the remainder
            if current_chunk_sentences:
                remaining_text = " ".join(current_chunk_sentences).strip()
                if remaining_text:
                    self.chunks.append({
                        "chunk_id": chunk_counter,
                        "page": page_num,
                        "content": remaining_text,
                        "char_count": len(remaining_text),
                        "word_count": len(remaining_text.split()),
                        "doc_name": self.doc_name,
                    })
                    chunk_counter += 1

        return self.chunks

    def get_paragraphs(self) -> List[str]:
        """
        Backwards-compatible API for returning list of chunk strings.
        """
        chunks = self.get_chunks()
        return [c["content"] for c in chunks]

    def get_document_stats(self) -> Dict[str, Any]:
        """
        Provides high-level document metrics for audit reporting.
        """
        chunks = self.get_chunks()
        total_words = sum(c["word_count"] for c in chunks)
        total_chars = sum(c["char_count"] for c in chunks)

        return {
            "doc_name": self.doc_name,
            "total_pages": self.total_pages,
            "total_chunks": len(chunks),
            "total_words": total_words,
            "total_chars": total_chars,
            "avg_chunk_size": int(total_chars / len(chunks)) if chunks else 0,
        }
