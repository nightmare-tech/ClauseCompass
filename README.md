# ClauseCompass 🧭

[![Python Version](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Status](https://img.shields.io/badge/status-active-success.svg)]()

An AI-powered decision engine designed to analyze complex, unstructured documents and provide structured, justified answers using Retrieval Augmented Generation (RAG).

<!-- 
**DEMO**
(This is where you should insert a GIF or short video of your CLI in action. 
It's the most impactful part of a README!)
-->
![Demo of ClauseCompass]([link_to_your_demo.gif])

## 🚀 About ClauseCompass

Developed as a capstone project for the IGT Solutions internship and for the HackRx 6.0 hackathon, **ClauseCompass** tackles a critical challenge for businesses: enabling Large Language Models (LLMs) to perform accurate, auditable reasoning based on specific, private knowledge bases like insurance policies or legal contracts.

Instead of a generic chatbot, **ClauseCompass** acts as a transactional **decision engine**. It takes a user query, retrieves relevant clauses from documents using semantic search, and instructs an LLM to generate a structured JSON response with a clear decision and justification, citing the source clauses.

### Key Features

*   **Dual-Mode RAG Architecture:**
    *   **Persistent Mode:** Queries a pre-ingested, admin-managed knowledge base stored in MongoDB Atlas for maximum speed and scalability.
    *   **Temporary Mode:** Allows users to dynamically upload their own documents for private, on-the-fly analysis in a single session.
*   **Advanced RAG Pipeline:** Leverages text embeddings and a vector database for semantic search, ensuring the LLM receives the most relevant context.
*   **Structured & Justified Outputs:** Employs sophisticated "Chain of Thought" prompt engineering to force the LLM to return a consistent JSON object, including the specific policy clauses used for its decision, providing auditability.
*   **Secure & Asynchronous API:** Backend built with a robust, asynchronous FastAPI application, featuring secure JWT-based user authentication.
*   **Developer-Friendly CLI:** A powerful, stateful command-line interface for testing, interaction, and demonstration of all features.

## 🏗️ Architecture Overview

The system is split into two main workflows: Data Ingestion (for the persistent knowledge base) and Query Processing (for both persistent and temporary modes).

<!-- You can create a simple diagram using Mermaid.js or an image -->
![Architecture Diagram]([link_to_your_architecture_diagram.png])

1.  **Data Ingestion (Offline):** An admin script (`ingest_kb.py`) loads, chunks, and embeds documents into a persistent MongoDB Atlas collection, where a Vector Search Index is utilized for fast retrieval.
2.  **Query Processing (Real-time):**
    *   The FastAPI backend exposes endpoints for different query modes.
    *   `/evaluate` queries the persistent MongoDB collection, using metadata filtering for document selection.
    *   `/evaluate-with-docs` dynamically creates an in-memory ChromaDB instance to process user-uploaded files for a single session.
    *   Both paths construct a detailed prompt to guide the AI21 LLM in its step-by-step reasoning process.

## 🛠️ Tech Stack

*   **Backend:** Python, FastAPI
*   **Database (Persistent KB):** MongoDB with Atlas Vector Search
*   **Database (Temporary KB):** ChromaDB (in-memory)
*   **AI / LLM:** AI21 Labs (Jamba-Mini)
*   **Embeddings:** Sentence Transformers (`all-MiniLM-L6-v2`)
*   **Data Processing:** LangChain (Document Loaders, Text Splitters)
*   **Authentication:** JWT (via `python-jose`), Passlib (bcrypt)
*   **CLI:** Rich, shlex (for a custom interactive shell)

## ⚙️ Setup and Usage

### Prerequisites

*   Python 3.10+
*   A MongoDB Atlas cluster with a configured Vector Search Index.
*   A `.env` file with your API keys and database credentials.

### Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/nightmare-tech/ClauseCompass.git
    cd ClauseCompass
    ```
2.  Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

### Configuration

Create a `.env` file in the root directory and add the following variables:
```
AI21_API_KEY="your_ai21_api_key"
USERN="your_mongodb_username"
PASSW="your_mongodb_password"
JWT_SECRET="a_strong_random_secret_key"
```


### 1. Data Ingestion (for Persistent Mode)

Place your knowledge base documents (e.g., PDFs) into a directory (e.g., `knowledge_base_docs/`). Then, run the ingestion script.

```bash
python ingest_kb.py --docs_path ./knowledge_base_docs/ --clear
```
(Important: Ensure you have created the Vector Search Index in your MongoDB Atlas cluster first! See ingest_kb.py for an example definition.)

### 2. Running the Application
1. Start the Backend Server:
    ```
    python app.py
    ```
2. Run the CLI Client (in a separate terminal):
   ```
   python cli.py
   ```

### CLI Usage Example
```bash
Welcome to the ClauseCompass Decision Engine CLI!
ClauseCompass (persistent) (logged out) > login
Email: user@example.com
Password: ***
✔ Login successful.

ClauseCompass (persistent) (user@example.com) > list_docs
# Shows list of ingested documents

ClauseCompass (persistent) (user@example.com) > set_docs policy_A.pdf
# Sets the context for the query

ClauseCompass (persistent) (user@example.com) [1 doc] > 46M, knee surgery, 3-month policy
# Sends query to the /evaluate endpoint... (receives JSON response)

# --- Switching Modes ---

ClauseCompass (persistent) (user@example.com) > mode temporary
✔ Mode switched to: temporary

ClauseCompass (temporary) (user@example.com) > add_doc /path/to/my_local_file.pdf
# Stages a local file for upload

ClauseCompass (temporary) (user@example.com) [1 doc] > What is the termination clause?
# Sends query and file to the /evaluate-with-docs endpoint... (receives JSON response)
```

### 📜 License
This project is licensed under the MIT License - see the LICENSE file for details.