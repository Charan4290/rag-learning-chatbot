
from pathlib import Path
import chromadb
from sentence_transformers import SentenceTransformer
from gpt4all import GPT4All

# Step 1: Connect to ChromaDB
database_path = Path(__file__).resolve().parent / "chroma_db"

client = chromadb.PersistentClient(path=str(database_path))
collection = client.get_collection(name="company_policies")

# Step 2: Load the embedding model
print("Loading embedding model...", flush=True)
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Step 3: Load the local LLM
print("Loading local LLM...", flush=True)
llm = GPT4All(
    "Phi-3-mini-4k-instruct.Q4_0.gguf",
    device="cpu",
    n_ctx=2048,
    allow_download=False
)

print("\nABC Technologies RAG Assistant is ready!")

# Step 4: Ask for employee input
question = input("\nEnter your question: ").strip()

if not question:
    print("Please enter a question.")
else:
    # Step 5: Convert the question into an embedding
    question_embedding = embedding_model.encode(question).tolist()

    # Step 6: Retrieve matching policies
    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=3,
        include=["documents", "metadatas"]
    )

    # Step 7: Build the retrieved context
    context_parts = []

    for i, document in enumerate(results["documents"][0]):
        source = results["metadatas"][0][i]["source"]
        context_parts.append(f"Source: {source}\n{document}")

    context = "\n\n---\n\n".join(context_parts)

    # Step 8: Prepare the prompt
    prompt = f"""
You are an assistant for ABC Technologies.

Answer using ONLY the company policy context.
Do not invent company rules.
If the answer is missing, say you cannot find it.
Mention the source document and section.

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
"""

    # Step 9: Generate the answer
    print("\nGenerating answer...", flush=True)

    with llm.chat_session():
        answer = llm.generate(
            prompt,
            max_tokens=180,
            temp=0.1
        )

    print("\n===== ANSWER =====")
    print(answer)

llm.close()
