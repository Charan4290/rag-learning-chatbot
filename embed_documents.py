
from sentence_transformers import SentenceTransformer
from chunk_documents import get_chunks

# Step 1: Get chunks from another Python file
all_chunks = get_chunks()

print("Chunks received:", len(all_chunks), flush=True)

# Step 2: Extract the text from each chunk
chunk_texts = [chunk["text"] for chunk in all_chunks]

# Step 3: Load the embedding model
print("Loading embedding model...", flush=True)
model = SentenceTransformer("all-MiniLM-L6-v2")

# Step 4: Convert the chunks into embeddings
print("Generating embeddings...", flush=True)
embeddings = model.encode(chunk_texts)

# Step 5: Check the results
print("Embedding shape:", embeddings.shape)
print("First chunk source:", all_chunks[0]["source"])
print("First 5 values:", embeddings[0][:5])
