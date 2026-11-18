
# RAG Learning Chatbot

A Retrieval-Augmented Generation (RAG) chatbot developed using Python, Sentence Transformers, ChromaDB, and local language models.

## Project Overview

This project explores how an AI chatbot can answer questions using information retrieved from a collection of documents.

The example knowledge base contains company policy and IT support documents.

## Technologies Used

- Python
- Sentence Transformers
- ChromaDB
- GPT4All / Local LLMs
- Vector Embeddings
- Retrieval-Augmented Generation (RAG)

## How It Works

1. Load text documents.
2. Divide documents into smaller chunks.
3. Convert chunks into vector embeddings.
4. Store embeddings in ChromaDB.
5. Receive a user's question.
6. Retrieve relevant document chunks.
7. Construct a prompt using the retrieved information.
8. Generate an answer using a language model.

## Project Structure

- `documents/` - Knowledge base documents
- `load_documents.py` - Document loading
- `chunk_documents.py` - Text chunking
- `embed_documents.py` - Embedding generation
- `store_embeddings.py` - Vector database storage
- `search_documents.py` - Similarity search
- `prepare_prompt.py` - Prompt preparation
- `rag_backend.py` - RAG backend logic
- `web_app.py` - Web application
- `requirements.txt` - Python dependencies

## Key Concepts

- Semantic search
- Vector embeddings
- Document chunking
- Vector databases
- LLM integration
- Context-aware answer generation

## Project Status

Learning and development project.
