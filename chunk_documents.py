
from pathlib import Path
import re


# ==========================================
# STEP 1: Read and split company documents
# ==========================================

def get_chunks():

    # Find the existing documents folder
    documents_folder = (
        Path(__file__).resolve().parent / "documents"
    )

    # Automatically discover all .txt files
    files = sorted(documents_folder.glob("*.txt"))

    all_chunks = []

    # Process each company document
    for file_path in files:

        # Read document content
        text = file_path.read_text(encoding="utf-8")

        # ==================================
        # STEP 2: Split at section headings
        # ==================================

        parts = re.split(
            r"(?=^SECTION \d+:)",
            text,
            flags=re.MULTILINE
        )

        # ==================================
        # STEP 3: Fix header-only chunks
        # ==================================

        # Combine document header with first section
        if len(parts) > 1 and parts[0].strip():

            parts[1] = parts[0] + "\n" + parts[1]

            # Remove the standalone header
            parts = parts[1:]

        # ==================================
        # STEP 4: Save each chunk
        # ==================================

        for part in parts:

            chunk = part.strip()

            if chunk:

                all_chunks.append({
                    "source": file_path.name,
                    "text": chunk
                })

    return all_chunks


# ==========================================
# STEP 5: Test chunking in terminal
# ==========================================

if __name__ == "__main__":

    chunks = get_chunks()

    print("\nTotal chunks:", len(chunks))

    for index, chunk in enumerate(chunks, start=1):

        print(
            f"Chunk {index}: {chunk['source']}"
        )

        # Show the section heading, if available
        section = next(
            (
                line.strip()
                for line in chunk["text"].splitlines()
                if line.strip().startswith("SECTION ")
            ),
            "Section not specified"
        )

        print("Section:", section)
        print("-" * 40)
