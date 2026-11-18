
from pathlib import Path

# Find the documents folder next to this Python file
documents_folder = Path(__file__).resolve().parent / "documents"

# List the company documents we want to load
filenames = ["HR_Policy.txt", "IT_Support_Guide.txt"]

# Read each document
for filename in filenames:
    file_path = documents_folder / filename

    print(f"\nLoading: {filename}", flush=True)

    text = file_path.read_text(encoding="utf-8")

    print("Characters loaded:", len(text))
    print("Preview:", text[:120])
    print("Document loaded successfully!")
