import os

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

import faiss
import numpy as np
import pickle
import torch
from typing import List, Any, Dict, Optional

try:
    torch.set_num_threads(2)
except Exception:
    pass

from sentence_transformers import SentenceTransformer
from src.embedding import EmbeddingPipeline

class FaissVectorStore:
    """
    FAISS-based vector store with persistence, alignment validation,
    and safe retrieval guards against invalid indices and out-of-bounds states.
    """
    def __init__(
        self,
        persist_dir: str = "faiss_store",
        embedding_model: str = "all-MiniLM-L6-v2",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        chunking_strategy: str = "recursive",
    ):
        self.persist_dir = persist_dir
        os.makedirs(self.persist_dir, exist_ok=True)
        self.index: Optional[faiss.Index] = None
        self.metadata: List[Dict[str, Any]] = []
        self.embedding_model = embedding_model
        try:
            self.model = SentenceTransformer(
                embedding_model, model_kwargs={"low_cpu_mem_usage": True}
            )
        except Exception:
            self.model = SentenceTransformer(embedding_model)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.chunking_strategy = chunking_strategy
        print(f"[INFO] Loaded embedding model: {embedding_model}")

    def validate_alignment(self) -> None:
        """
        Validate that the FAISS index vector count matches the metadata count.
        Raises ValueError if out of sync.
        """
        if self.index is not None:
            ntotal = getattr(self.index, "ntotal", None)
            if ntotal is not None and ntotal != len(self.metadata):
                raise ValueError(
                    f"Index/metadata mismatch: FAISS has {ntotal} vectors, "
                    f"but metadata has {len(self.metadata)} entries."
                )

    def build_from_documents(self, documents: List[Any]) -> None:
        if not documents:
            print("[WARNING] No documents provided to build vector store.")
            return

        print(f"[INFO] Building vector store from {len(documents)} raw documents...")
        emb_pipe = EmbeddingPipeline(
            model_name=self.embedding_model,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            strategy=getattr(self, "chunking_strategy", "recursive"),
        )
        chunks = emb_pipe.chunk_documents(documents)
        if not chunks:
            print("[WARNING] Chunking produced 0 chunks; index not rebuilt.")
            return

        # Deduplicate chunks by content_hash to prevent duplicate vectors
        seen_hashes = set()
        deduped_chunks = []
        for chunk in chunks:
            chash = chunk.metadata.get("content_hash")
            if chash:
                if chash in seen_hashes:
                    continue
                seen_hashes.add(chash)
            deduped_chunks.append(chunk)

        if len(deduped_chunks) < len(chunks):
            print(
                f"[INFO] Deduplicated {len(chunks) - len(deduped_chunks)} repeated chunks; "
                f"{len(deduped_chunks)} unique chunks will be indexed."
            )

        embeddings = emb_pipe.embed_chunks(deduped_chunks)

        # Preserve full metadata contract: text + provenance fields
        metadatas = [
            {"text": chunk.page_content, **chunk.metadata}
            for chunk in deduped_chunks
        ]

        # Stage and validate the new index before replacing the active index
        dim = embeddings.shape[1]
        staged_index = faiss.IndexFlatL2(dim)
        staged_index.add(np.array(embeddings).astype("float32"))

        if staged_index.ntotal != len(metadatas):
            raise ValueError(
                f"Staging validation failed: index vectors ({staged_index.ntotal}) "
                f"!= metadata entries ({len(metadatas)}). Existing index preserved."
            )

        # Atomic switch in memory only after staging succeeds
        self.index = staged_index
        self.metadata = metadatas
        self.validate_alignment()

        # Atomic persistence to disk
        self.save()
        print(f"[INFO] Vector store successfully built and saved to {self.persist_dir}")

    def add_embeddings(self, embeddings: np.ndarray, metadatas: Optional[List[Any]] = None) -> None:
        dim = embeddings.shape[1]
        if self.index is None:
            self.index = faiss.IndexFlatL2(dim)
        self.index.add(embeddings)
        if metadatas:
            self.metadata.extend(metadatas)
        self.validate_alignment()
        print(f"[INFO] Added {embeddings.shape[0]} vectors to Faiss index.")

    def save(self) -> None:
        """
        Safely persist FAISS index and metadata.
        Uses temporary files and atomic replacement to prevent index corruption.
        """
        if self.index is None:
            raise ValueError("Cannot save uninitialized index.")

        self.validate_alignment()

        faiss_path = os.path.join(self.persist_dir, "faiss.index")
        meta_path = os.path.join(self.persist_dir, "metadata.pkl")
        model_path = os.path.join(self.persist_dir, "embedding_model.txt")

        faiss_tmp = faiss_path + ".tmp"
        meta_tmp = meta_path + ".tmp"
        model_tmp = model_path + ".tmp"

        # Write to temporary files first
        try:
            faiss.write_index(self.index, faiss_tmp)
            with open(meta_tmp, "wb") as f:
                pickle.dump(self.metadata, f)
            with open(model_tmp, "w", encoding="utf-8") as f:
                f.write(self.embedding_model)

            # Atomic swap into place
            os.replace(faiss_tmp, faiss_path)
            os.replace(meta_tmp, meta_path)
            os.replace(model_tmp, model_path)
        except Exception:
            for tmp in (faiss_tmp, meta_tmp, model_tmp):
                if os.path.exists(tmp):
                    try:
                        os.remove(tmp)
                    except OSError:
                        pass
            raise

        print(f"[INFO] Saved Faiss index and metadata to {self.persist_dir}")

    def load(self) -> None:
        """
        Load persisted FAISS index and metadata with validation.
        """
        faiss_path = os.path.join(self.persist_dir, "faiss.index")
        meta_path = os.path.join(self.persist_dir, "metadata.pkl")
        model_path = os.path.join(self.persist_dir, "embedding_model.txt")

        if not os.path.exists(faiss_path):
            raise FileNotFoundError(f"FAISS index file missing: {faiss_path}")
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"Metadata file missing: {meta_path}")

        loaded_index = faiss.read_index(faiss_path)
        with open(meta_path, "rb") as f:
            loaded_metadata = pickle.load(f)

        if not isinstance(loaded_metadata, list):
            raise TypeError(f"Expected metadata list in {meta_path}, got {type(loaded_metadata)}")

        if loaded_index.ntotal != len(loaded_metadata):
            raise ValueError(
                f"Corrupted index store: index vectors ({loaded_index.ntotal}) "
                f"!= metadata items ({len(loaded_metadata)})."
            )

        self.index = loaded_index
        self.metadata = loaded_metadata

        if os.path.exists(model_path):
            with open(model_path, "r", encoding="utf-8") as f:
                stored_model = f.read().strip()
                if stored_model and stored_model != self.embedding_model:
                    print(
                        f"[WARNING] Stored model '{stored_model}' differs from "
                        f"store instance model '{self.embedding_model}'."
                    )

        print(f"[INFO] Loaded Faiss index and metadata from {self.persist_dir}")

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search the index with safe guards against empty index, invalid indices,
        and FAISS -1 sentinel values.
        """
        if self.index is None or getattr(self.index, "ntotal", 0) == 0:
            return []

        if top_k <= 0:
            return []

        # Ensure query embedding is 2D float32
        if query_embedding.ndim == 1:
            query_embedding = np.expand_dims(query_embedding, axis=0)
        query_embedding = query_embedding.astype("float32")

        D, I = self.index.search(query_embedding, top_k)
        results = []
        for idx, dist in zip(I[0], D[0]):
            idx_int = int(idx)
            # FAISS sentinel is -1 for missing slots; also guard against out-of-bounds
            if 0 <= idx_int < len(self.metadata):
                meta = self.metadata[idx_int]
            else:
                meta = None

            results.append({
                "index": idx_int,
                "distance": float(dist),
                "metadata": meta,
            })
        return results

    def query(self, query_text: str, top_k: int = 5) -> List[Dict[str, Any]]:
        if not query_text or not query_text.strip():
            return []

        print(f"[INFO] Querying vector store for: '{query_text}'")
        query_emb = self.model.encode([query_text]).astype("float32")
        return self.search(query_emb, top_k=top_k)

# Example usage
if __name__ == "__main__":
    from data_loader import load_all_documents
    docs = load_all_documents("data")
    store = FaissVectorStore("faiss_store")
    store.build_from_documents(docs)
    store.load()
    print(store.query("What is attention mechanism?", top_k=3))
