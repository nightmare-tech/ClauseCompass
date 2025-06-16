# Chatty
This project implements a sophisticated conversational AI chatbot capable of engaging in dynamic conversations and answering questions based on a custom, updatable knowledge base. It utilizes Retrieval Augmented Generation (RAG) by integrating ChromaDB as a vector store for semantic search over ingested documents.

Key Features:

- Backend: Robust and asynchronous API built with FastAPI (Python).
- Conversational AI: Powered by AI21's Jamba-Mini language model.
- Retrieval Augmented Generation (RAG): ChromaDB stores and searches document embeddings (e.g., from Sentence Transformers) to provide relevant context to the LLM.
- Custom Knowledge Base: Includes a data ingestion pipeline (ingest_data.py) to load, chunk, embed, and store documents (PDF, TXT, MD) into ChromaDB.
- Authentication: Secure JWT-based authentication for user registration and login.
- Data Storage: MongoDB for user profiles and chat history.
- Configurable: Designed to restrict answers to the knowledge base or allow more general responses based on configuration.

# Installation
```
git clone github.com/nightmare-tech/chatty.git
pip install -r requirements.txt
```