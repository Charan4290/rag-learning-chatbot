
from pathlib import Path
import time
import chromadb
from sentence_transformers import SentenceTransformer
from gpt4all import GPT4All

# STEP 1: Connect to our existing ChromaDB database
database_path = Path(__file__).resolve().parent / "chroma_db"

client = chromadb.PersistentClient(path=str(database_path))

collection = client.get_collection(
    name="company_policies"
)

# STEP 2: Load the embedding model
print("Loading embedding model...", flush=True)

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

# STEP 3: Load our local Llama model
print("Loading Llama 3.2 1B...", flush=True)

llm = GPT4All(
    "Llama-3.2-1B-Instruct-Q4_0.gguf",
    device="cpu",
    n_ctx=2048,
    allow_download=False
)

print("\nABC Technologies RAG Chatbot is ready!")
print("Type 'exit' to stop.")

try:
    # STEP 4: Keep accepting questions
    while True:

        question = input("\nYou: ").strip()

        if question.lower() == "exit":
            print("Chatbot closed.")
            break

        if not question:
            print("Please enter a question.")
            continue

        # STEP 5: Convert question into embedding
        embedding_start = time.perf_counter()

        question_embedding = embedding_model.encode(
            question
        ).tolist()

        embedding_time = (
            time.perf_counter() - embedding_start
        )

        # STEP 6: Retrieve relevant document
        retrieval_start = time.perf_counter()

        results = collection.query(
            query_embeddings=[question_embedding],
            n_results=1,
            include=["documents", "metadatas"]
        )

        retrieval_time = (
            time.perf_counter() - retrieval_start
        )

        if not results["documents"][0]:
            print("\nAssistant: No matching policy found.")
            continue

        # STEP 7: Extract the actual source data
        document = results["documents"][0][0]
        source = results["metadatas"][0][0]["source"]

        # Extract the section heading from retrieved text
        section = next(
            (
                line.strip()
                for line in document.splitlines()
                if line.strip().startswith("SECTION ")
            ),
            "Section not specified"
        )

        # STEP 8: Build a concise RAG prompt
        prompt = f"""
Company policy:
Source: {source}
{document}

Question: {question}

Give a direct answer using only the facts above.
Maximum two short sentences.
Do not invent policy numbers, page numbers, or rules.
Do not repeat the source text.
If the policy does not contain the answer, say so.
"""

        # STEP 9: Apply Llama instruction formatting
        formatted_prompt = (
            "<|begin_of_text|>"
            "<|start_header_id|>system<|end_header_id|>\n\n"
            "You are an ABC Technologies HR and IT assistant. "
            "Answer using only the retrieved company policy. "
            "Be concise and factual. "
            "Do not invent missing information."
            "<|eot_id|>"
            "<|start_header_id|>user<|end_header_id|>\n\n"
            f"{prompt}"
            "<|eot_id|>"
            "<|start_header_id|>assistant<|end_header_id|>\n\n"
        )

        # STEP 10: Print retrieval timings
        print(
            f"\nEmbedding time: {embedding_time:.2f} seconds"
        )
        print(
            f"Retrieval time: {retrieval_time:.2f} seconds"
        )

        # STEP 11: Generate answer with streaming
        generation_start = time.perf_counter()
        first_token_time = None

        print("\nAssistant: ", end="", flush=True)

        for token in llm.generate(
            formatted_prompt,
            max_tokens=45,
            temp=0.1,
            streaming=True
        ):

            if first_token_time is None:
                first_token_time = (
                    time.perf_counter() - generation_start
                )

            print(token, end="", flush=True)

        generation_time = (
            time.perf_counter() - generation_start
        )

        print()

        # STEP 12: Display source attribution
        print(f"\nSource document: {source}")
        print(f"Source section: {section}")

        # STEP 13: Print generation timings
        if first_token_time is not None:
            print(
                f"\nTime to first token: "
                f"{first_token_time:.2f} seconds"
            )
        else:
            print("\nNo tokens generated.")

        print(
            f"LLM generation time: "
            f"{generation_time:.2f} seconds"
        )

except KeyboardInterrupt:
    print("\nChatbot stopped by user.")

finally:
    # STEP 14: Release local LLM resources
    llm.close()
