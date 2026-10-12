import hashlib
import re
from typing import List, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import numpy as np
from src.data_loader import load_all_documents

class EmbeddingPipeline:
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        strategy: str = "recursive",
    ):
        if chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {chunk_size}")
        if chunk_overlap < 0:
            raise ValueError(f"chunk_overlap cannot be negative, got {chunk_overlap}")
        if chunk_overlap >= chunk_size:
            raise ValueError(
                f"chunk_overlap ({chunk_overlap}) must be strictly less than chunk_size ({chunk_size})"
            )
        if strategy not in ("recursive", "structure_aware"):
            raise ValueError(f"Unsupported strategy '{strategy}'; expected 'recursive' or 'structure_aware'")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.strategy = strategy
        self.model = SentenceTransformer(model_name)
        print(f"[INFO] Loaded embedding model: {model_name} (strategy={strategy}, size={chunk_size}, overlap={chunk_overlap})")

    def _create_splitter(self) -> RecursiveCharacterTextSplitter:
        strategy = getattr(self, "strategy", "recursive")
        if strategy == "structure_aware":
            # Semantic separators respecting headers, lists, sentence boundaries
            separators = [
                "\n# ",
                "\n## ",
                "\n### ",
                "\n#### ",
                "\n\n",
                "\n* ",
                "\n- ",
                "\n1. ",
                "\n",
                ". ",
                "? ",
                "! ",
                "; ",
                " ",
                "",
            ]
        else:
            # Baseline recursive separators
            separators = ["\n\n", "\n", " ", ""]

        return RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=separators,
        )

    def chunk_documents(self, documents: List[Any]) -> List[Any]:
        if not documents:
            return []

        splitter = self._create_splitter()
        raw_chunks = splitter.split_documents(documents)
        strat = getattr(self, "strategy", "recursive")
        print(f"[INFO] Split {len(documents)} documents into {len(raw_chunks)} chunks using strategy '{strat}'.")

        enriched_chunks = []
        doc_chunk_counters: dict[str, int] = {}

        for chunk in raw_chunks:
            # Preserve existing metadata
            meta = dict(chunk.metadata or {})

            # Deterministic document ID fallback
            doc_id = meta.get("doc_id")
            if not doc_id:
                src_name = meta.get("source", "unknown_doc")
                doc_id = hashlib.sha256(src_name.encode("utf-8")).hexdigest()[:12]
                meta["doc_id"] = doc_id

            # Stable chunk counter per document
            idx = doc_chunk_counters.get(doc_id, 0)
            doc_chunk_counters[doc_id] = idx + 1

            chunk_text = chunk.page_content.strip()
            content_hash = hashlib.sha256(chunk_text.encode("utf-8")).hexdigest()[:16]

            meta["chunk_id"] = f"{doc_id}:{idx}"
            meta["chunk_index"] = idx
            meta["content_hash"] = content_hash
            chunk.metadata = meta
            enriched_chunks.append(chunk)

        return enriched_chunks

    def embed_chunks(self, chunks: List[Any]) -> np.ndarray:
        texts = [chunk.page_content for chunk in chunks]
        print(f"[INFO] Generating embeddings for {len(texts)} chunks...")
        embeddings = self.model.encode(texts, show_progress_bar=True)
        print(f"[INFO] Embeddings shape: {embeddings.shape}")
        return embeddings

# Example usage
if __name__ == "__main__":
    
    docs = load_all_documents("data")
    emb_pipe = EmbeddingPipeline()
    chunks = emb_pipe.chunk_documents(docs)
    embeddings = emb_pipe.embed_chunks(chunks)
    print("[INFO] Example embedding:", embeddings[0] if len(embeddings) > 0 else None)
