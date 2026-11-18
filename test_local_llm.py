
import time
from gpt4all import GPT4All

print("Loading Phi-3 Mini...", flush=True)

model = GPT4All(
    "Phi-3-mini-4k-instruct.Q4_0.gguf",
    device="cpu",
    n_ctx=2048,
    allow_download=False
)

print("Model ready!")

question = "Say hello in one sentence."

print("\nGenerating...", flush=True)

start = time.perf_counter()

try:
    with model.chat_session():
        for token in model.generate(
            question,
            max_tokens=20,
            temp=0.1,
            streaming=True
        ):
            print(token, end="", flush=True)

    print()
    print(
        "Generation time:",
        round(time.perf_counter() - start, 2),
        "seconds"
    )

finally:
    model.close()
