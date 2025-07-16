import os
import logging
import re
import json
import hashlib
from pathlib import Path
from dotenv import load_dotenv
import argparse

# --- Embedding Model Setup (e.g., Sentence Transformers) ---
# Ensure this is installed: pip install sentence-transformers
from sentence_transformers import SentenceTransformer

# --- LangChain for Document Loading and Splitting ---
# Ensure these are installed: pip install langchain langchain-community pypdf
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# --- MongoDB ---
from pymongo.mongo_client import MongoClient as MongoClientDB
from pymongo.server_api import ServerApi
from urllib.parse import quote_plus

# --- Logging Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()

# --- MongoDB Configuration ---
MONGO_USER_ENV = os.getenv("USERN")
MONGO_PASS_ENV = os.getenv("PASSW")
MONGO_KB_COLLECTION_NAME = "company_kb_vectorized"

if not MONGO_USER_ENV or not MONGO_PASS_ENV:
    raise ValueError("MongoDB username (USERN) or password (PASSW) not found in environment variables.")

escaped_username = quote_plus(MONGO_USER_ENV)
escaped_password = quote_plus(MONGO_PASS_ENV)
MONGO_URI = f"mongodb+srv://{escaped_username}:{escaped_password}@chattydb.dfuykzc.mongodb.net/?retryWrites=true&w=majority&appName=chattydb"

mongo_client_instance = None
mongo_kb_collection = None

try:
    mongo_client_instance = MongoClientDB(MONGO_URI, server_api=ServerApi('1'))
    mongo_client_instance.admin.command('ping')
    logger.info("Successfully connected to MongoDB for ingestion!")
    db = mongo_client_instance["chattydb"]
    mongo_kb_collection = db[MONGO_KB_COLLECTION_NAME]
except Exception as e:
    logger.error(f"Failed to connect to MongoDB: {e}", exc_info=True)
    exit()

# --- Embedding Function Configuration ---
EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'
try:
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    test_emb = embedding_model.encode("test")
    EMBEDDING_DIMENSION = len(test_emb)
    logger.info(f"Initialized Embedding Model: {EMBEDDING_MODEL_NAME} with dimension {EMBEDDING_DIMENSION}")
except Exception as e:
    logger.error(f"Failed to load embedding model '{EMBEDDING_MODEL_NAME}': {e}", exc_info=True)
    exit()


def extract_and_enrich_metadata(pages: list):
    """
    Iterates through LangChain Document pages, finds clause/section headings,
    and enriches the metadata of each Document object.
    """
    # Regex to find patterns like "SECTION D) EXCLUSIONS-...", "1)", "a."
    section_pattern = re.compile(r"^\s*SECTION\s+([A-Z])\)", re.IGNORECASE)
    clause_pattern = re.compile(r"^\s*(\d+)\)\s*(.*)")
    
    current_section = None
    
    for page in pages:
        # We need to process the text line by line to find headings and apply them
        page_content = page.page_content
        lines = page_content.split('\n')
        
        # Reset current_clause for each page to avoid carry-over, but carry over section
        current_clause_num = None
        current_clause_title = None

        for line in lines:
            line_stripped = line.strip()
            
            section_match = section_pattern.match(line_stripped)
            if section_match:
                current_section = f"SECTION {section_match.group(1).upper()}"
                continue # Don't process this line further

            clause_match = clause_pattern.match(line_stripped)
            if clause_match:
                current_clause_num = clause_match.group(1)
                # Take first few words of the clause as its title
                current_clause_title = ' '.join(clause_match.group(2).split()[:5])

        # Enrich the page metadata with the last found section/clause on that page
        if current_section:
            page.metadata["policy_section"] = current_section
        if current_clause_num:
            page.metadata["policy_clause_num"] = current_clause_num
            page.metadata["policy_clause_title"] = current_clause_title
            
    return pages


def load_and_chunk_documents(docs_path: str, chunk_size: int = 1000, chunk_overlap: int = 150):
    logger.info(f"Loading ONLY PDF documents from: {docs_path}")
    source_path = Path(docs_path)
    pdf_files = sorted(list(source_path.rglob("*.pdf")))

    if not pdf_files:
        logger.warning(f"No PDF files found in {docs_path}. Exiting."); return []
    
    logger.info(f"Found {len(pdf_files)} PDF files to process.")
    
    all_pages_from_loader = []
    for file_path in pdf_files:
        try:
            loader = PyPDFLoader(str(file_path))
            pages = loader.load()
            all_pages_from_loader.extend(pages)
            logger.info(f"Successfully loaded {len(pages)} pages from {file_path}.")
        except Exception as e:
            logger.error(f"Error loading PDF file {file_path}: {e}")

    if not all_pages_from_loader:
        logger.warning("No pages were loaded. Exiting."); return []

    # --- New Step: Enrich metadata before chunking ---
    logger.info("Enriching documents with section and clause metadata...")
    enriched_pages = extract_and_enrich_metadata(all_pages_from_loader)
    
    logger.info(f"Splitting {len(enriched_pages)} pages into smaller chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, length_function=len
    )
    
    chunked_documents = text_splitter.split_documents(enriched_pages)
    logger.info(f"Split pages into {len(chunked_documents)} final chunks.")
    return chunked_documents

def generate_stable_id(text_content, source_path):
    hasher = hashlib.md5()
    hasher.update(str(source_path).encode('utf-8'))
    hasher.update(text_content.encode('utf-8'))
    return hasher.hexdigest()

def ingest_into_mongodb(
    chunked_docs: list,
    clear_collection: bool = False,
    batch_size: int = 100
):
    if not chunked_docs: logger.info("No documents to ingest."); return
    if mongo_kb_collection is None: logger.error("MongoDB KB collection is not initialized."); return

    if clear_collection:
        logger.warning(f"Clearing ALL existing documents from MongoDB collection: '{MONGO_KB_COLLECTION_NAME}'")
        delete_result = mongo_kb_collection.delete_many({})
        logger.info(f"Successfully deleted {delete_result.deleted_count} documents.")

    logger.info(f"Preparing to ingest {len(chunked_docs)} chunks into MongoDB...")
    
    # Inform user about the required Atlas Index (this is very helpful)
    logger.warning("="*80)
    logger.warning("IMPORTANT: Ensure a VECTOR SEARCH INDEX is created in MongoDB Atlas on the "
                   f"'{MONGO_KB_COLLECTION_NAME}' collection.")
    logger.warning(f"The index needs to be on the 'embedding_vector' field with {EMBEDDING_DIMENSION} dimensions and 'cosine' similarity.")
    logger.warning("It's also HIGHLY recommended to index 'metadata.source_document' for filtering.")
    logger.warning("Example Atlas Index Definition (JSON):")
    logger.warning(json.dumps({
        "fields": [
            {"type": "vector", "path": "embedding_vector", "numDimensions": EMBEDDING_DIMENSION, "similarity": "cosine"},
            {"type": "filter", "path": "metadata.source_document", "representation": "string"}
        ]
    }, indent=2))
    logger.warning("="*80)

    num_chunks = len(chunked_docs)
    for i in range(0, num_chunks, batch_size):
        batch_of_docs = chunked_docs[i:i + batch_size]
        texts_to_embed = [doc.page_content for doc in batch_of_docs]
        
        try:
            batch_embeddings = embedding_model.encode(texts_to_embed)
        except Exception as e:
            logger.error(f"Error generating embeddings for batch, skipping: {e}"); continue
        
        documents_for_mongo_batch = []
        for idx, lc_doc in enumerate(batch_of_docs):
            source_file_path = lc_doc.metadata.get("source", "unknown_source")
            
            # Prepare a clean metadata dictionary for storage
            metadata_to_store = {
                "source_document": Path(source_file_path).name,
                "page_number": lc_doc.metadata.get("page", 0) + 1, # page is 0-indexed
                "policy_section": lc_doc.metadata.get("policy_section", "N/A"),
                "policy_clause_num": lc_doc.metadata.get("policy_clause_num", "N/A"),
                "policy_clause_title": lc_doc.metadata.get("policy_clause_title", "N/A")
            }

            mongo_doc = {
                "_id": generate_stable_id(lc_doc.page_content, source_file_path),
                "text_chunk": lc_doc.page_content,
                "embedding_vector": batch_embeddings[idx].tolist(),
                "metadata": metadata_to_store
            }
            documents_for_mongo_batch.append(mongo_doc)

        if not documents_for_mongo_batch: continue

        try:
            mongo_kb_collection.insert_many(documents_for_mongo_batch, ordered=False)
            logger.info(f"Added batch {i//batch_size + 1} with {len(documents_for_mongo_batch)} documents to MongoDB.")
        except Exception as e:
            logger.error(f"Error adding batch to MongoDB: {e}")

    logger.info(f"Finished ingesting. Final count in '{MONGO_KB_COLLECTION_NAME}': {mongo_kb_collection.count_documents({})}")

# --- Main execution block (argparse) ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest PDF documents into MongoDB for Vector Search.")
    parser.add_argument("--docs_path", type=str, required=True, help="Path to the directory containing documents.")
    parser.add_argument("--chunk_size", type=int, default=1000, help="Target size for text chunks.")
    parser.add_argument("--chunk_overlap", type=int, default=150, help="Overlap between text chunks.")
    parser.add_argument("--clear", action="store_true", help="Clear the MongoDB collection before ingesting.")
    parser.add_argument("--batch_size", type=int, default=100, help="Number of documents to insert in a single batch.")

    args = parser.parse_args()

    try:
        document_chunks = load_and_chunk_documents(args.docs_path, args.chunk_size, args.chunk_overlap)
        if document_chunks:
            ingest_into_mongodb(
                document_chunks,
                clear_collection=args.clear,
                batch_size=args.batch_size
            )
    except Exception as e:
        logger.error(f"An error occurred during the ingestion process: {e}", exc_info=True)