from __future__ import annotations

from typing import Any, Callable

from .chunking import _dot
from .embeddings import _mock_embed
from .models import Document


class EmbeddingStore:
    """
    A vector store for text chunks.

    Tries to use ChromaDB if available; falls back to an in-memory store.
    The embedding_fn parameter allows injection of mock embeddings for tests.
    """

    def __init__(
        self,
        collection_name: str = "documents",
        embedding_fn: Callable[[str], list[float]] | None = None,
    ) -> None:
        if embedding_fn is None:
            import os
            try:
                from dotenv import load_dotenv
                load_dotenv()
            except ImportError:
                pass

            provider = os.getenv("EMBEDDING_PROVIDER")
            openai_key = os.getenv("OPENAI_API_KEY")

            if provider == "openai" or (openai_key and provider != "mock" and provider != "local"):
                from .embeddings import OpenAIEmbedder
                self._embedding_fn = OpenAIEmbedder()
            elif provider == "local":
                from .embeddings import LocalEmbedder
                self._embedding_fn = LocalEmbedder()
            else:
                self._embedding_fn = _mock_embed
        else:
            self._embedding_fn = embedding_fn
        self._collection_name = collection_name
        self._use_chroma = False
        self._store: list[dict[str, Any]] = []
        self._collection = None
        self._next_index = 0

        try:
            import chromadb  # noqa: F401

            client = chromadb.Client()
            self._collection = client.get_or_create_collection(name=collection_name)
            self._use_chroma = True
        except Exception:
            self._use_chroma = False
            self._collection = None

    def _make_record(self, doc: Document) -> dict[str, Any]:
        metadata = doc.metadata.copy()
        if "doc_id" not in metadata:
            metadata["doc_id"] = doc.id
            
        return {
            "id": doc.id,
            "content": doc.content,
            "metadata": metadata,
            "embedding": self._embedding_fn(doc.content)
        }

    def _search_records(self, query: str, records: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
        query_emb = self._embedding_fn(query)
        scored_records = []
        for rec in records:
            score = _dot(query_emb, rec["embedding"])
            scored_records.append({**rec, "score": score})
        scored_records.sort(key=lambda x: x["score"], reverse=True)
        return scored_records[:top_k]

    def add_documents(self, docs: list[Document]) -> None:
        """
        Embed each document's content and store it.

        For ChromaDB: use collection.add(ids=[...], documents=[...], embeddings=[...])
        For in-memory: append dicts to self._store
        """
        if self._use_chroma:
            ids = [doc.id for doc in docs]
            documents = [doc.content for doc in docs]
            embeddings = [self._embedding_fn(doc.content) for doc in docs]
            metadatas = []
            for doc in docs:
                m = doc.metadata.copy()
                if "doc_id" not in m:
                    m["doc_id"] = doc.id
                metadatas.append(m)
            self._collection.add(ids=ids, documents=documents, embeddings=embeddings, metadatas=metadatas)
        else:
            for doc in docs:
                self._store.append(self._make_record(doc))

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """
        Find the top_k most similar documents to query.

        For in-memory: compute dot product of query embedding vs all stored embeddings.
        """
        if self._use_chroma:
            query_emb = self._embedding_fn(query)
            res = self._collection.query(query_embeddings=[query_emb], n_results=top_k)
            results = []
            if res and res.get("ids") and len(res["ids"]) > 0:
                for i in range(len(res["ids"][0])):
                    results.append({
                        "id": res["ids"][0][i],
                        "content": res["documents"][0][i],
                        "metadata": res["metadatas"][0][i] if res["metadatas"] else {},
                        "score": 1 - res["distances"][0][i] if "distances" in res and res["distances"] else 0.0
                    })
            return results
        else:
            return self._search_records(query, self._store, top_k)

    def get_collection_size(self) -> int:
        """Return the total number of stored chunks."""
        if self._use_chroma:
            return self._collection.count()
        return len(self._store)

    def search_with_filter(self, query: str, top_k: int = 3, metadata_filter: dict = None) -> list[dict]:
        """
        Search with optional metadata pre-filtering.

        First filter stored chunks by metadata_filter, then run similarity search.
        """
        if self._use_chroma:
            query_emb = self._embedding_fn(query)
            res = self._collection.query(
                query_embeddings=[query_emb],
                n_results=top_k,
                where=metadata_filter if metadata_filter else None
            )
            results = []
            if res and res.get("ids") and len(res["ids"]) > 0:
                for i in range(len(res["ids"][0])):
                    results.append({
                        "id": res["ids"][0][i],
                        "content": res["documents"][0][i],
                        "metadata": res["metadatas"][0][i] if res["metadatas"] else {},
                        "score": 1 - res["distances"][0][i] if "distances" in res and res["distances"] else 0.0
                    })
            return results
        else:
            filtered = self._store
            if metadata_filter:
                filtered = []
                for r in self._store:
                    match = True
                    for k, v in metadata_filter.items():
                        if r.get("metadata", {}).get(k) != v:
                            match = False
                            break
                    if match:
                        filtered.append(r)
            return self._search_records(query, filtered, top_k)

    def delete_document(self, doc_id: str) -> bool:
        """
        Remove all chunks belonging to a document.

        Returns True if any chunks were removed, False otherwise.
        """
        if self._use_chroma:
            try:
                sz_before = self._collection.count()
                self._collection.delete(where={"doc_id": doc_id})
                sz_after = self._collection.count()
                return sz_after < sz_before
            except Exception:
                return False
        else:
            initial_len = len(self._store)
            self._store = [r for r in self._store if r.get("metadata", {}).get("doc_id") != doc_id]
            return len(self._store) < initial_len
