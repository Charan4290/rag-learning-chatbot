
from pathlib import Path
import time
import chromadb
from sentence_transformers import SentenceTransformer
from gpt4all import GPT4All

# STEP 1: Connect to the existing ChromaDB database
database_path = Path(__file__).resolve().parent / "chroma_db"

client = chromadb.PersistentClient(path=str(database_path))
collection = client.get_collection(name="company_policies")

# STEP 2: Load the embedding model
print("Loading embedding model...", flush=True)

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# STEP 3: Load the local LLM
print("Loading Phi-3 Mini...", flush=True)

llm = GPT4All(
    "Phi-3-mini-4k-instruct.Q4_0.gguf",
    device="cpu",
    n_ctx=2048,
    allow_download=False
)

print("\nABC Technologies Chatbot is ready!")
print("Type 'exit' to stop.")

try:
    # STEP 4: Keep accepting employee questions
    while True:
        question = input("\nYou: ").strip()

        if question.lower() == "exit":
            print("Chatbot closed.")
            break

        if not question:
            print("Please enter a question.")
            continue

        # STEP 5: Convert the question into an embedding
        embedding_start = time.perf_counter()

        question_embedding = embedding_model.encode(
            question
        ).tolist()

        embedding_time = time.perf_counter() - embedding_start

        # STEP 6: Search ChromaDB
        retrieval_start = time.perf_counter()

        results = collection.query(
            query_embeddings=[question_embedding],
            n_results=1,
            include=["documents", "metadatas", "distances"]
        )

        retrieval_time = time.perf_counter() - retrieval_start

        # STEP 7: Prepare retrieved context
        documents = results["documents"][0]
        metadatas = results["metadatas"][0]

        if not documents:
            print("\nAssistant: No matching company policies found.")
            continue

        context_parts = []

        for i, document in enumerate(documents):
            source = metadatas[i]["source"]

            context_parts.append(
                f"Source: {source}\n{document}"
            )

        context = "\n\n---\n\n".join(context_parts)

        # Show retrieved source for debugging
        print("\nRetrieved source:", metadatas[0]["source"])

        # STEP 8: Create a shorter RAG prompt
        prompt = f"""
Answer the question using only the company policy below.
Be brief. Include the source filename and section.
If the policy does not answer the question, say:
"The policy does not contain this information."

POLICY:
{context}

QUESTION:
{question}

ANSWER:
"""

        # STEP 9: Display pipeline timings
        print(f"\nEmbedding time: {embedding_time:.2f} seconds")
        print(f"Retrieval time: {retrieval_time:.2f} seconds")

        # STEP 10: Generate the answer with streaming
        generation_start = time.perf_counter()
        first_token_time = None

        print("\nAssistant: ", end="", flush=True)

        with llm.chat_session():
            for token in llm.generate(
                prompt,
                max_tokens=50,
                temp=0.1,
                streaming=True
            ):
                if first_token_time is None:
                    first_token_time = (
                        time.perf_counter() - generation_start
                    )

                print(token, end="", flush=True)

        generation_time = time.perf_counter() - generation_start

        print()

        if first_token_time is not None:
            print(
                f"Time to first token: {first_token_time:.2f} seconds"
            )
        else:
            print("No tokens generated.")

        print(
            f"LLM generation time: {generation_time:.2f} seconds"
        )

except KeyboardInterrupt:
    print("\nChatbot stopped by user.")

finally:
    llm.close()
