
from pathlib import Path
from collections import Counter
import chromadb

# ==========================================
# STEP 1: Import existing chunks and embeddings
# ==========================================

from embed_documents import all_chunks, embeddings


# ==========================================
# STEP 2: Verify embeddings
# ==========================================

print("\nChunks ready:", len(all_chunks))
print("Embedding shape:", embeddings.shape)

if embeddings.shape != (len(all_chunks), 384):
    raise ValueError(
        "Unexpected embedding shape!"
    )


# ==========================================
# STEP 3: Connect to existing ChromaDB
# ==========================================

BASE_DIR = Path(__file__).resolve().parent

DATABASE_PATH = BASE_DIR / "chroma_db"

client = chromadb.PersistentClient(
    path=str(DATABASE_PATH)
)

collection = client.get_or_create_collection(
    name="company_policies"
)

print(
    "\nRecords before update:",
    collection.count()
)


# ==========================================
# STEP 4: Prepare updated records
# ==========================================

ids = [
    f"chunk-{i}"
    for i in range(len(all_chunks))
]

documents = [
    chunk["text"]
    for chunk in all_chunks
]

metadatas = [
    {"source": chunk["source"]}
    for chunk in all_chunks
]


# ==========================================
# STEP 5: Insert or update all 17 chunks
# ==========================================

collection.upsert(
    ids=ids,
    embeddings=embeddings.tolist(),
    documents=documents,
    metadatas=metadatas
)

print("Updated current chunk records.")


# ==========================================
# STEP 6: Find obsolete chunk records
# ==========================================

stored_ids = collection.get()["ids"]

current_ids = set(ids)

obsolete_ids = [
    record_id
    for record_id in stored_ids
    if record_id.startswith("chunk-")
    and record_id not in current_ids
]

print(
    "Obsolete records found:",
    len(obsolete_ids)
)


# ==========================================
# STEP 7: Delete obsolete chunk records
# ==========================================

if obsolete_ids:

    collection.delete(
        ids=obsolete_ids
    )

    print(
        "Deleted obsolete records:",
        sorted(obsolete_ids)
    )

else:

    print("No obsolete records found.")


# ==========================================
# STEP 8: Verify final database
# ==========================================

print("\nCollection:", collection.name)

print(
    "Records after update:",
    collection.count()
)

stored = collection.get(
    include=["metadatas"]
)

counts = Counter(
    metadata["source"]
    for metadata in stored["metadatas"]
)

print("\nChunks by document:")

for filename, count in sorted(counts.items()):

    print(
        f"{filename}: {count} chunks"
    )


# ==========================================
# STEP 9: Final verification
# ==========================================

if collection.count() != len(all_chunks):

    raise ValueError(
        "Database record count does not match "
        "the current number of chunks."
    )

print("\nChromaDB update complete!")
