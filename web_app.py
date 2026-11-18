
import hmac
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import streamlit as st


# ==========================================
# STEP 1: Project configuration
# ==========================================

BASE_DIR = Path(__file__).resolve().parent
DOCUMENTS_DIR = BASE_DIR / "documents"

MAX_FILE_BYTES = 1024 * 1024

FILE_NAME_PATTERN = re.compile(
    r"[A-Za-z0-9][A-Za-z0-9_.-]{0,99}\.txt\Z",
    re.IGNORECASE
)

st.set_page_config(
    page_title="ABC Technologies AI Assistant",
    page_icon="🤖",
    layout="centered"
)


# ==========================================
# STEP 2: Load existing RAG backend
# ==========================================

@st.cache_resource
def load_backend():
    from rag_backend import ask_rag
    return ask_rag


# ==========================================
# STEP 3: List uploaded documents
# ==========================================

def document_names():

    DOCUMENTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    return sorted(
        path.name
        for path in DOCUMENTS_DIR.glob("*.txt")
        if path.is_file()
    )


# ==========================================
# STEP 4: Validate uploaded documents
# ==========================================

def valid_uploaded_text(uploaded_file):

    if uploaded_file is None:
        raise ValueError(
            "Choose a UTF-8 .txt file first."
        )

    name = Path(uploaded_file.name).name

    if (
        name != uploaded_file.name
        or not FILE_NAME_PATTERN.fullmatch(name)
        or name.startswith(".")
        or ".." in name
    ):
        raise ValueError(
            "Filename must be a simple .txt filename "
            "without folders or '..'."
        )

    content = uploaded_file.getvalue()

    if not content or len(content) > MAX_FILE_BYTES:
        raise ValueError(
            "File must be nonempty and no larger than 1 MB."
        )

    try:
        decoded = content.decode("utf-8")

    except UnicodeDecodeError as exc:
        raise ValueError(
            "Upload a UTF-8 encoded text file."
        ) from exc

    if not decoded.strip():
        raise ValueError(
            "The text document is empty."
        )

    return name, content


# ==========================================
# STEP 5: Safely replace document files
# ==========================================

def atomic_replace(path, data):

    temp_name = None

    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=DOCUMENTS_DIR,
            prefix=".upload-",
            suffix=".tmp",
            delete=False
        ) as temp:

            temp_name = temp.name
            temp.write(data)
            temp.flush()
            os.fsync(temp.fileno())

        os.replace(temp_name, path)

    finally:
        if temp_name and os.path.exists(temp_name):
            os.unlink(temp_name)


# ==========================================
# STEP 6: Automatically update ChromaDB
# ==========================================

def reindex():

    result = subprocess.run(
        [
            sys.executable,
            str(BASE_DIR / "store_embeddings.py")
        ],
        cwd=str(BASE_DIR),
        capture_output=True,
        text=True,
        timeout=300
    )

    if result.returncode != 0:

        raise RuntimeError(
            (
                result.stderr
                or result.stdout
                or "Unknown indexing failure"
            )[-3500:]
        )

    return result.stdout


# ==========================================
# STEP 7: Save, replace or delete policy
# ==========================================

def save_and_index(mode, target_name, content=None):

    DOCUMENTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    if (
        target_name not in document_names()
        and mode != "add"
    ):
        raise ValueError(
            "Selected document no longer exists."
        )

    path = DOCUMENTS_DIR / target_name

    existed = path.exists()

    old_data = (
        path.read_bytes()
        if existed
        else None
    )

    if mode == "add":

        if existed:
            raise FileExistsError(
                "A document with that name already exists."
            )

        with path.open("xb") as handle:
            handle.write(content)

    elif mode == "replace":

        atomic_replace(path, content)

    elif mode == "delete":

        path.unlink()

    else:
        raise ValueError(
            "Unknown document operation."
        )

    # Update the vector database
    try:
        return reindex()

    except Exception as original_error:

        # Attempt to restore the previous document
        # and rebuild its original index
        try:

            if existed:
                atomic_replace(path, old_data)

            elif path.exists():
                path.unlink()

            reindex()

        except Exception as rollback_error:

            raise RuntimeError(
                "Indexing failed and automatic recovery "
                "could not be verified. Check the documents "
                "folder and rerun store_embeddings.py after "
                "fixing the error. "
                f"Original: {original_error}; "
                f"recovery: {rollback_error}"
            ) from original_error

        raise RuntimeError(
            "Indexing failed; the previous document "
            "and index were restored. "
            f"Original error: {original_error}"
        ) from original_error


# ==========================================
# STEP 8: Administrator authentication
# ==========================================

password_configured = bool(
    os.environ.get("RAG_ADMIN_PASSWORD", "")
)

if "is_admin" not in st.session_state:
    st.session_state.is_admin = False


with st.sidebar:

    st.subheader("🔐 Administrator access")

    if st.session_state.is_admin:

        st.success("Admin signed in")

        if st.button("Log out"):

            st.session_state.is_admin = False

            st.rerun()

    elif not password_configured:

        st.info(
            "Admin login is disabled until "
            "RAG_ADMIN_PASSWORD is configured."
        )

    else:

        with st.form(
            "admin_login",
            clear_on_submit=True
        ):

            entered_password = st.text_input(
                "Admin password",
                type="password"
            )

            submitted = st.form_submit_button(
                "Sign in"
            )

            if submitted:

                correct = os.environ[
                    "RAG_ADMIN_PASSWORD"
                ]

                if hmac.compare_digest(
                    entered_password,
                    correct
                ):

                    st.session_state.is_admin = True

                    st.rerun()

                else:

                    st.error("Incorrect password.")


# ==========================================
# STEP 9: Website heading
# ==========================================

st.title("🤖 ABC Technologies AI Assistant")

st.caption(
    "Company HR & IT policy assistant"
)


# ==========================================
# STEP 10: Chat history
# ==========================================

if "messages" not in st.session_state:
    st.session_state.messages = []


for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.write(message["content"])


# ==========================================
# STEP 11: Employee asks a question
# ==========================================

question = st.chat_input(
    "Ask about company policies..."
)


if question:

    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    with st.chat_message("user"):
        st.write(question)

    # ======================================
    # STEP 12: Call existing RAG backend
    # ======================================

    with st.chat_message("assistant"):

        with st.spinner(
            "Searching company policies..."
        ):

            try:

                answer, source, section = (
                    load_backend()(question)
                )

                st.write(answer)

                st.caption(
                    f"Source: {source} — {section}"
                )

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": (
                        f"{answer}\n\n"
                        f"Source: {source} — {section}"
                    )
                })

            except Exception:

                st.error(
                    "Unable to answer. "
                    "Check the Streamlit terminal."
                )


st.divider()


# ==========================================
# STEP 13: Admin document management
# ==========================================

if st.session_state.is_admin:

    st.header("📁 Admin document management")

    st.caption(
        "Manage UTF-8 text policies. "
        "Every change automatically rebuilds ChromaDB."
    )

    names = document_names()

    # ======================================
    # FEATURE 1: View existing documents
    # ======================================

    st.subheader("1. View uploaded documents")

    if names:

        for name in names:

            size = (
                DOCUMENTS_DIR / name
            ).stat().st_size

            st.write(
                f"📄 {name} — {size:,} bytes"
            )

        selected_preview = st.selectbox(
            "Preview document",
            names,
            key="preview_document"
        )

        with st.expander(
            "View selected policy text"
        ):

            st.text(
                (
                    DOCUMENTS_DIR / selected_preview
                ).read_text(encoding="utf-8")
            )

    else:

        st.info(
            "No documents have been uploaded yet."
        )

    # ======================================
    # FEATURE 2: Upload new documents
    # ======================================

    st.subheader("2. Upload a new document")

    new_file = st.file_uploader(
        "Choose a new .txt file",
        type=["txt"],
        key="new_doc"
    )

    if st.button(
        "Upload and index",
        disabled=new_file is None
    ):

        try:

            filename, content = valid_uploaded_text(
                new_file
            )

            with st.spinner(
                "Saving and indexing all policies..."
            ):

                log = save_and_index(
                    "add",
                    filename,
                    content
                )

            st.success(
                f"Uploaded and indexed: {filename}"
            )

            with st.expander("Indexing log"):
                st.code(log)

        except Exception as exc:
            st.error(str(exc))

    # ======================================
    # FEATURE 3: Replace existing documents
    # ======================================

    st.subheader(
        "3. Replace an existing document"
    )

    if names:

        replace_target = st.selectbox(
            "Existing policy to replace",
            names,
            key="replace_target"
        )

        replacement = st.file_uploader(
            "Choose the updated .txt content",
            type=["txt"],
            key="replacement_doc"
        )

        st.warning(
            "This replaces the selected policy's "
            "contents, then reindexes. "
            "Keep a copy of important originals."
        )

        confirm_replace = st.checkbox(
            "I confirm I want to replace this policy",
            key="confirm_replace"
        )

        if st.button(
            "Replace and reindex",
            disabled=not (
                replacement and confirm_replace
            )
        ):

            try:

                _, content = valid_uploaded_text(
                    replacement
                )

                with st.spinner(
                    "Replacing and updating the index..."
                ):

                    log = save_and_index(
                        "replace",
                        replace_target,
                        content
                    )

                st.success(
                    f"Replaced and indexed: {replace_target}"
                )

                with st.expander("Indexing log"):
                    st.code(log)

            except Exception as exc:
                st.error(str(exc))

    else:

        st.info(
            "Upload a document before using replacement."
        )

    # ======================================
    # FEATURE 4: Delete outdated documents
    # ======================================

    st.subheader(
        "4. Delete an outdated document"
    )

    if names:

        delete_target = st.selectbox(
            "Document to delete",
            names,
            key="delete_target"
        )

        confirm_delete = st.checkbox(
            (
                f"I understand {delete_target} "
                "will be deleted from disk and the index"
            ),
            key="confirm_delete"
        )

        if st.button(
            "Delete and reindex",
            disabled=not confirm_delete
        ):

            try:

                with st.spinner(
                    "Deleting and updating the index..."
                ):

                    log = save_and_index(
                        "delete",
                        delete_target
                    )

                st.success(
                    "Deleted and removed from search: "
                    f"{delete_target}"
                )

                with st.expander("Indexing log"):
                    st.code(log)

            except Exception as exc:
                st.error(str(exc))

else:

    st.info(
        "Document management is available only "
        "to signed-in administrators."
    )
