import uuid
from pathlib import Path
from src.chunking import MarkdownChunker
from src.store import EmbeddingStore
from src.models import Document
from src.embeddings import LocalEmbedder

def main():
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
        
    # 1. Read files
    print("Reading data files...")
    all_text = ""
    data_dir = Path("data")
    
    # We will iterate through data1.md to data5.md if they exist
    for i in range(1, 6):
        path = data_dir / f"data{i}.md"
        if path.exists():
            print(f"  - Loading {path}")
            all_text += path.read_text(encoding="utf-8") + "\n"
        else:
            print(f"  - Warning: {path} not found")

    if not all_text:
        print("Error: No data found in data/ directory.")
        return

    # 2. Chunking with MarkdownChunker
    print("\nChunking text with MarkdownChunker...")
    chunker = MarkdownChunker(max_chunk_size=600, overlap=100)
    chunks = chunker.chunk(all_text)
    print(f"✅ Total chunks created: {len(chunks)}")

    # 3. Create Document objects
    print("Creating Document objects...")
    docs = []
    for i, chunk_text in enumerate(chunks):
        docs.append(
            Document(
                id=str(uuid.uuid4()),
                content=chunk_text,
                metadata={
                    "source": "combined_markdown",
                    "chunk_index": i,
                }
            )
        )

    # 4. Initialize Store and Index
    print("\nInitializing EmbeddingStore (Mock) and indexing...")
    from src.embeddings import _mock_embed
    embedder = _mock_embed
    
    store = EmbeddingStore(
        collection_name="vinuni_rag_test",
        embedding_fn=embedder
    )
    
    store.add_documents(docs)
    print(f"✅ Indexed {store.get_collection_size()} chunks into store.")

    # 5. Test Search
    query = "điều kiện học bổng tài năng dành cho sinh viên mới là gì?"
    print(f"\n🔍 Searching for: '{query}'")
    print("=" * 60)
    
    results = store.search(query, top_k=3)
    
    for i, res in enumerate(results):
        print(f"\n--- Result {i+1} (Score: {res['score']:.4f}) ---")
        # In store.py, search results have 'content' and 'metadata'
        print(f"Metadata: {res.get('metadata', {})}")
        print("-" * 30)
        print(res['content'])
        print("-" * 60)

if __name__ == "__main__":
    main()
