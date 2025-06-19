# Chatty
## MongoDB Atlas Vector Search Integration Branch (`mongodb-vector-search`)

This branch contains an alternative implementation of the "Chatty" RAG chatbot, utilizing **MongoDB Atlas Vector Search** as the vector database instead of ChromaDB.

**Key Differences from `main` (ChromaDB version):**
*   **Vector Storage & Search:** Employs MongoDB Atlas for storing text embeddings and performing semantic similarity searches using the `$vectorSearch` aggregation stage.
*   **Data Ingestion:** The `ingest_kb.py` script is modified to generate embeddings directly and load them, along with text chunks and metadata, into a dedicated MongoDB collection.
*   **Application Logic:** The FastAPI application (`app.py`) queries MongoDB Atlas for relevant context to augment LLM prompts.

This branch demonstrates the adaptability of the RAG architecture to different vector store solutions. **Note:** Requires a MongoDB Atlas cluster with a configured Vector Search Index on the knowledge base collection.