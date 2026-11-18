
from gpt4all import GPT4All

# Step 1: Reuse our existing RAG prompt
from prepare_prompt import prompt

# Step 2: Load our local language model
print("\nLoading local LLM...", flush=True)

model = GPT4All(
    "Phi-3-mini-4k-instruct.Q4_0.gguf",
    device="cpu",
    n_ctx=2048,
    allow_download=False
)

print("Local LLM loaded successfully!")

# Step 3: Give the retrieved policy context to the LLM
print("\nGenerating grounded answer...", flush=True)

with model.chat_session():
    answer = model.generate(
        prompt,
        max_tokens=160,
        temp=0.1
    )

# Step 4: Display the generated answer
print("\n===== RAG ANSWER =====")
print(answer)

# Step 5: Release the model
model.close()
