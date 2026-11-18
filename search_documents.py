
from pathlib import Path
import chromadb
from sentence_transformers import SentenceTransformer

# Step 1: Connect to the existing database
database_path = Path(__file__).resolve().parent / "chroma_db"

client = chromadb.PersistentClient(path=str(database_path))

collection = client.get_collection(name="company_policies")

print("Stored chunks:", collection.count())

# Step 2: Ask a real employee question
question = "I entered the wrong password too many times. What should I do?"

print("\nEmployee question:", question)

# Step 3: Convert the question into an embedding
model = SentenceTransformer("all-MiniLM-L6-v2")

question_embedding = model.encode(question).tolist()

print("Question embedding dimensions:", len(question_embedding))

# Step 4: Search the stored document embeddings
results = collection.query(
    query_embeddings=[question_embedding],
    n_results=3,
    include=["documents", "metadatas", "distances"]
)

# Step 5: Display the matching policy chunks
for i in range(len(results["ids"][0])):
    print(f"\nRESULT {i + 1}")
    print("Source:", results["metadatas"][0][i]["source"])
    print("Distance:", round(results["distances"][0][i], 4))
    print("Policy text:")
    print(results["documents"][0][i])
