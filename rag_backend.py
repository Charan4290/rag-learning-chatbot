
from pathlib import Path
import re

import chromadb
from sentence_transformers import SentenceTransformer


# ==========================================
# STEP 1: Configuration
# ==========================================

BASE_DIR = Path(__file__).resolve().parent
DATABASE_PATH = BASE_DIR / "chroma_db"
COLLECTION_NAME = "company_policies"

NOT_FOUND = (
    "I couldn't find that information "
    "in the available company policies."
)

MIN_SIMILARITY = 0.48


# ==========================================
# STEP 2: Load embedding model
# ==========================================

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)


# ==========================================
# STEP 3: Helper functions
# ==========================================

def not_found_response():
    return (
        NOT_FOUND,
        "No supporting source found",
        "Not available"
    )


def normalize_question(question):

    text = question.lower()

    replacements = {
        "annual leave entitlement": "vacation days",
        "annual leave allowance": "vacation days",
        "paid annual leave": "vacation days",
        "vacation entitlement": "vacation days",
        "unsuccessful login": "failed login",
        "lose their id": "lost id",
        "password reset link valid":
            "password reset links expire",
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    return text


def is_ambiguous(question):

    q = question.lower()

    broad = (
        "time off" in q
        or "days off" in q
    )

    specific = any(
        term in q
        for term in [
            "vacation",
            "sick",
            "parental",
            "maternity",
            "paternity"
        ]
    )

    return broad and not specific


def extract_section(document):

    for line in document.splitlines():

        line = line.strip()

        if line.startswith("SECTION "):
            return line

    return "Section not specified"


# ==========================================
# STEP 4: Extract complete policy sentences
# ==========================================

def extract_policy_sentences(document):

    sentences = []
    pending = ""

    for raw_line in document.splitlines():

        line = raw_line.strip()

        if not line:
            continue

        # Skip document headings and metadata
        if line.startswith((
            "SECTION ",
            "Document ID:",
            "Version:",
            "Effective Date:",
            "ABC TECHNOLOGIES"
        )):
            continue

        if line.isupper():
            continue

        # Join wrapped lines into one sentence
        pending = (
            f"{pending} {line}".strip()
            if pending
            else line
        )

        # A sentence ends at punctuation
        while True:

            match = re.search(
                r"[.!?](?=\s|$)",
                pending
            )

            if not match:
                break

            end = match.end()

            sentence = pending[:end].strip()

            pending = pending[end:].strip()

            if len(sentence.split()) >= 4:
                sentences.append(sentence)

    # Preserve remaining text if no final period
    if pending and len(pending.split()) >= 4:
        sentences.append(pending)

    return sentences


# ==========================================
# STEP 5: Topic evidence check
# ==========================================

def topic_supported(question, document):

    q = question.lower()
    d = document.lower()

    protected_topics = {
        "parental": ["parental"],
        "maternity": ["maternity"],
        "paternity": ["paternity"],
        "bonus": ["bonus"],
        "insurance": ["insurance"],
        "premium": ["premium"],
        "salary": ["salary"],
        "laptop": ["laptop"],
    }

    for keyword, evidence_words in protected_topics.items():

        if keyword in q:

            if not any(
                word in d
                for word in evidence_words
            ):
                return False

    return True


# ==========================================
# STEP 6: Find best supporting sentence
# ==========================================

def find_best_sentence(question, candidates):

    if not candidates:
        return None

    texts = [
        item["sentence"]
        for item in candidates
    ]

    question_vector = embedding_model.encode(
        question,
        normalize_embeddings=True
    )

    sentence_vectors = embedding_model.encode(
        texts,
        normalize_embeddings=True
    )

    similarities = sentence_vectors @ question_vector

    ranked = sorted(
        enumerate(similarities),
        key=lambda item: float(item[1]),
        reverse=True
    )

    # Check whether the user asks for a
    # particular numeric unit
    requested_unit = None

    q = question.lower()

    if "how many hours" in q:
        requested_unit = "hour"
    elif "how many days" in q:
        requested_unit = "day"
    elif "how many minutes" in q:
        requested_unit = "minute"

    for index, similarity in ranked:

        candidate = candidates[index]

        if float(similarity) < MIN_SIMILARITY:
            break

        sentence = candidate["sentence"]

        if not topic_supported(question, sentence):
            continue

        # If the question asks for hours,
        # require an hours quantity in evidence.
        if requested_unit is not None:

            pattern = (
                rf"\b\d+\s+"
                rf"{requested_unit}s?\b"
            )

            if not re.search(
                pattern,
                sentence,
                flags=re.IGNORECASE
            ):
                continue

        print(
            "Best sentence similarity:",
            round(float(similarity), 4)
        )

        print(
            "Supporting sentence:",
            sentence
        )

        return candidate

    return None


# ==========================================
# STEP 7: Main RAG function
# ==========================================

def ask_rag(question):

    if not question.strip():
        return not_found_response()

    if is_ambiguous(question):

        return (
            "Do you mean paid vacation days "
            "or paid sick leave?",
            "Clarification needed",
            "Leave type not specified"
        )

    normalized_question = normalize_question(
        question
    )

    client = chromadb.PersistentClient(
        path=str(DATABASE_PATH)
    )

    collection = client.get_collection(
        name=COLLECTION_NAME
    )

    if collection.count() == 0:
        return not_found_response()

    question_embedding = embedding_model.encode(
        normalized_question
    ).tolist()

    # Retrieve several relevant policy chunks
    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=min(5, collection.count()),
        include=["documents", "metadatas"]
    )

    if (
        not results["documents"]
        or not results["documents"][0]
    ):
        return not_found_response()

    candidates = []

    for document, metadata in zip(
        results["documents"][0],
        results["metadatas"][0]
    ):

        source = metadata["source"]
        section = extract_section(document)

        # This now returns complete sentences
        sentences = extract_policy_sentences(
            document
        )

        for sentence in sentences:

            candidates.append({
                "sentence": sentence,
                "source": source,
                "section": section
            })

    best = find_best_sentence(
        normalized_question,
        candidates
    )

    if best is None:

        print(
            "Evidence check: "
            "No clear supporting sentence."
        )

        return not_found_response()

    return (
        best["sentence"],
        best["source"],
        best["section"]
    )


# ==========================================
# STEP 8: Direct terminal test
# ==========================================

if __name__ == "__main__":

    question = (
        "How many hours of mandatory training "
        "must employees complete annually?"
    )

    answer, source, section = ask_rag(
        question
    )

    print("\nQuestion:", question)
    print("Assistant:", answer)
    print("Source:", source)
    print("Section:", section)
