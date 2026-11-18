
import time
from gpt4all import GPT4All

# STEP 1: Load the smaller model
print("Loading Llama 3.2 1B...", flush=True)

model = GPT4All(
    "Llama-3.2-1B-Instruct-Q4_0.gguf",
    device="cpu",
    n_ctx=2048
)

print("Model loaded successfully!")

# STEP 2: Ask the same question used in our RAG tests
question = "How many vacation days do full-time employees receive?"

# STEP 3: Provide a short company policy
context = """
ABC Technologies HR Policy - Section 1:
Full-time employees receive 20 paid vacation days annually.
"""

prompt = f"""
Answer using only this company policy.
Keep your answer short and mention the source.

POLICY:
{context}

QUESTION:
{question}

ANSWER:
"""

# STEP 4: Measure generation speed
start = time.perf_counter()
first_token_time = None

print("\nAssistant: ", end="", flush=True)

try:
    with model.chat_session():
        for token in model.generate(
            prompt,
            max_tokens=50,
            temp=0.1,
            streaming=True
        ):
            if first_token_time is None:
                first_token_time = time.perf_counter() - start

            print(token, end="", flush=True)

    total_time = time.perf_counter() - start

    print()

    if first_token_time is not None:
        print(f"Time to first token: {first_token_time:.2f} seconds")

    print(f"Generation time: {total_time:.2f} seconds")

finally:
    model.close()
