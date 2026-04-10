from src.chunking import ChunkingStrategyComparator

comparator = ChunkingStrategyComparator()

# for fname in ["data/data1.md", "data/data2.md", "data/data3.md", "data/data4.md", "data/data5.md"]:
for fname in ["data/data2.md"]:
    with open(fname, "r") as f:
        text = f.read()
    sep = "=" * 60
    print(f"\n{sep}")
    print(f"FILE: {fname} (length={len(text)} chars)")
    print(sep)
    result = comparator.compare(text, chunk_size=200)
    for strategy, info in result.items():
        print(f"\n--- {strategy} ---")
        print(f"  count: {info['count']}")
        print(f"  avg_length: {info['avg_length']:.1f}")
        print(f"  max_length: {info['max_length']}")
        print(f"  min_length: {info['min_length']}")
        print(f"  chunks:")
        for i, c in enumerate(info["chunks"]):
            preview = c[:80].replace("\n", "\\n")
            print(f"    [{i}] ({len(c)} chars) {preview}...")
