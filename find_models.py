
from gpt4all import GPT4All

print("Checking available GPT4All models...")

models = GPT4All.list_models()

for model in models:
    name = model.get("filename", "")
    size = model.get("filesize")

    if size is not None:
        size_gb = int(size) / (1024 ** 3)

        if size_gb < 1.5:
            print(
                f"Model: {name} | "
                f"Download size: {size_gb:.2f} GB"
            )
