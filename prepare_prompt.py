
from pathlib import Path
import chromadb
from sentence_transformers import SentenceTransformer

# Step 1: Open our saved vector database
database_path = Path(__file__).resolve().parent / "chroma_db"

client = chromadb.PersistentClient(path=str(database_path))
collection = client.get_collection(name="company_policies")

# Step 2: Employee question
question = "I entered the wrong password too many times. What should I do?"

# Step 3: Generate only the question embedding
model = SentenceTransformer("all-MiniLM-L6-v2")
question_embedding = model.encode(question).tolist()

# Step 4: Retrieve the 3 closest document chunks
results = collection.query(
    query_embeddings=[question_embedding],
    n_results=3,
    include=["documents", "metadatas"]
)

# Step 5: Build context with source filenames
context_parts = []

for i, document in enumerate(results["documents"][0]):
    source = results["metadatas"][0][i]["source"]

    context_parts.append(
        f"Source: {source}\n{document}"
    )

context = "\n\n---\n\n".join(context_parts)

# Step 6: Create the prompt for our future LLM
prompt = f"""
You are an IT and HR assistant for ABC Technologies.

Answer the employee's question using ONLY the company
policy context provided below.

If the context does not contain the answer, say:
"I could not find that information in the company policies."

Do not invent company rules.
Mention the source document in your answer.

COMPANY POLICY CONTEXT:
{context}

EMPLOYEE QUESTION:
{question}

ANSWER:
"""

print("\n===== PREPARED RAG PROMPT =====")
print(prompt)
