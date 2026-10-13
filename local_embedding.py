"""
Enterprise Local Embedding Engine using Hugging Face transformers + torch.
Replicates sentence-transformers/all-MiniLM-L6-v2 behavior locally without
requiring sentence-transformers or third-party cloud embedding APIs.

Output: L2-normalized float32 vectors of dimension 384, compatible with
cosine similarity search in VectorIndex.
"""

from typing import List, Dict, Any, Union, Tuple
import torch
from transformers import AutoTokenizer, AutoModel
from vector_index import VectorIndex


class LocalEmbedding:
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        distance_metric: str = "cosine",
    ):
        """
        Initialise the local embedding model and in-memory vector index.

        Args:
            model_name:      Hugging Face model ID to load locally.
            distance_metric: Distance metric for the index ('cosine' or 'euclidean').
        """
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"[LocalEmbedding] Loading {self.model_name} on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name)
        self.model.to(self.device)
        self.model.eval()
        print("[LocalEmbedding] Model ready.")

        self.store = VectorIndex(
            distance_metric=distance_metric,
            embedding_fn=self._embed_one,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _mean_pool(self, last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Compute attention-mask-weighted mean over token embeddings.
        Padding tokens are zeroed out via the mask.
        """
        mask_expanded = attention_mask.unsqueeze(-1).float()
        sum_embeddings = (last_hidden_state * mask_expanded).sum(dim=1)
        token_counts = mask_expanded.sum(dim=1).clamp(min=1e-9)
        return sum_embeddings / token_counts

    def _embed_one(self, text: str) -> List[float]:
        """
        Embed a single string and return it as a float list.
        Used internally by VectorIndex at search time.
        """
        return self.get_embeddings([text])[0].tolist()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_embeddings(self, texts: List[str], batch_size: int = 32) -> torch.Tensor:
        """
        Embed a list of strings locally in micro-batches to prevent OOM.

        Args:
            texts: List of strings to embed.
            batch_size: Micro-batch size for model inference.

        Returns:
            Float32 tensor of shape (N, 384), L2-normalised, on CPU.
        """
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            encoded = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )
            encoded = {k: v.to(self.device) for k, v in encoded.items()}

            with torch.no_grad():
                output = self.model(**encoded)

            embeddings = self._mean_pool(output.last_hidden_state, encoded["attention_mask"])
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
            all_embeddings.append(embeddings.cpu())

        if not all_embeddings:
            return torch.empty((0, 384), dtype=torch.float32)

        return torch.cat(all_embeddings, dim=0)

    def build_index(self, chunks: Union[List[str], List[Dict[str, Any]]]) -> None:
        """
        Embed all chunks in batches and store them in the vector index.
        Accepts either raw text strings or rich metadata dictionaries from PdfReader.

        Args:
            chunks: List of text strings or list of chunk metadata dictionaries.
        """
        if not chunks:
            print("[LocalEmbedding] Warning: No chunks provided to index.")
            return

        self.store.clear()

        # Handle both list of dicts and list of strings
        if isinstance(chunks[0], dict):
            text_list = [c["content"] for c in chunks]
            doc_dicts = chunks
        else:
            text_list = chunks
            doc_dicts = [
                {"chunk_id": idx + 1, "page": 1, "content": text, "doc_name": "document.pdf"}
                for idx, text in enumerate(chunks)
            ]

        embeddings = self.get_embeddings(text_list)

        for doc, embedding in zip(doc_dicts, embeddings):
            self.store.add_vector(embedding.tolist(), doc)

        print(f"[LocalEmbedding] Successfully indexed {len(chunks)} chunks with metadata.")

    def search(self, question: str, k: int = 3) -> List[Tuple[Dict[str, Any], float]]:
        """
        Legacy search API: returns list of (document_dict, distance) tuples.
        """
        return self.store.search(question, k=k)

    def search_with_metadata(self, question: str, k: int = 4) -> List[Dict[str, Any]]:
        """
        Enterprise search API: returns structured match dictionaries with page numbers,
        similarity confidence percentages, and document attribution.
        """
        detailed = self.store.search_detailed(question, k=k)
        results = []

        for item in detailed:
            doc = item["document"]
            sim_score = item["similarity_score"]
            distance = item["distance"]

            results.append({
                "chunk_id": doc.get("chunk_id", 0),
                "page": doc.get("page", 1),
                "content": doc.get("content", ""),
                "similarity_score": sim_score,
                "distance": distance,
                "doc_name": doc.get("doc_name", "document.pdf"),
                "word_count": doc.get("word_count", len(doc.get("content", "").split())),
            })

        return results

    def get_context(self, question: str, k: int = 3) -> str:
        """
        Legacy context retriever: joined by separator.
        """
        results = self.search(question, k=k)
        return "\n\n---\n\n".join(doc["content"] for doc, _ in results)

    def get_context_with_citations(self, question: str, k: int = 4) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Enterprise context retriever: constructs an auditable, cited context block
        for the LLM prompt while returning the citation metadata for UI rendering.

        Returns:
            Tuple of:
            - Formatted context string with explicit [Source N | Page X | Match: Y%] headers
            - List of citation metadata dictionaries
        """
        citations = self.search_with_metadata(question, k=k)
        context_blocks = []

        for idx, cite in enumerate(citations, 1):
            header = f"[EXHIBIT {idx} | Source: {cite['doc_name']} | Page {cite['page']} | Relevance: {cite['similarity_score']}%]"
            block = f"{header}\n\"{cite['content']}\""
            context_blocks.append(block)

        context_str = "\n\n" + ("=" * 40) + "\n\n".join(context_blocks) + "\n\n" + ("=" * 40)
        return context_str, citations

    def get_all_documents(self) -> List[Dict[str, Any]]:
        """
        Returns all stored document chunks for the Document Inspector UI.
        """
        return list(self.store.documents)