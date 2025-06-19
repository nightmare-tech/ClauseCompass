import os
import logging
from pathlib import Path
from dotenv import load_dotenv
import argparse
import json
import hashlib

from langchain_community.document_loaders import PyPDFLoader 
from langchain.text_splitter import RecursiveCharacterTextSplitter

from pymongo.mongo_client import MongoClient as MongoClientDB
from pymongo.server_api import ServerApi
from urllib.parse import quote_plus

from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
load_dotenv()

# --- MongoDB Configuration ---
MONGO_USER_ENV = os.getenv("USERN")
MONGO_PASS_ENV = os.getenv("PASSW")
# Define a new collection name for vectorized KB in MongoDB
MONGO_KB_COLLECTION_NAME = "company_kb_vectorized" # NEW

if not MONGO_USER_ENV and not MONGO_PASS_ENV:
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
    db = mongo_client_instance["chattydb"] # Assuming same DB, new collection
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
    logger.info(f"Initialized Sentence Transformer embedding model: {EMBEDDING_MODEL_NAME} with dimension {EMBEDDING_DIMENSION}")
except Exception as e:
    logger.error(f"Failed to load embedding model '{EMBEDDING_MODEL_NAME}': {e}", exc_info=True)
    exit()


      
def load_and_chunk_documents(docs_path: str, chunk_size: int = 1000, chunk_overlap: int = 150):
    logger.info(f"Loading ONLY PDF documents from: {docs_path} using PyPDFLoader.")
    source_path = Path(docs_path)
    
    pdf_files = sorted(list(source_path.rglob("*.pdf")))

    if not pdf_files:
        logger.warning(f"No PDF files found in {docs_path}. Exiting document loading.")
        return []
    
    logger.info(f"Found {len(pdf_files)} PDF files to process.")
    
    all_pages_from_loader = [] 
    for file_path in pdf_files:
        try:
            logger.info(f"Loading {file_path}...")
            loader = PyPDFLoader(str(file_path))
            # .load() returns a list of Document objects, one for each page
            pages = loader.load() 
            all_pages_from_loader.extend(pages)
            logger.info(f"Successfully loaded {len(pages)} pages from {file_path}.")
        except Exception as e:
            logger.error(f"Error loading PDF file {file_path}: {e}", exc_info=False)

    if not all_pages_from_loader:
        logger.warning("No pages were successfully loaded from any PDF files. Exiting.")
        return []

    logger.info(f"Loaded a total of {len(all_pages_from_loader)} pages. Now splitting into smaller chunks.")
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        add_start_index=True
    )
    
    # PyPDFLoader already puts the source file path in metadata, so this will be preserved.
    chunked_documents = text_splitter.split_documents(all_pages_from_loader)
    logger.info(f"Split loaded pages into {len(chunked_documents)} final chunks.")
    
    return chunked_documents


def generate_stable_id(text_content, source_path):
    """Generates a stable ID based on content and source path."""
    hasher = hashlib.md5()
    hasher.update(str(source_path).encode('utf-8'))
    hasher.update(text_content.encode('utf-8'))
    return hasher.hexdigest()


def ingest_into_mongodb(
    chunked_langchain_docs: list,
    clear_collection: bool = False,
    batch_size: int = 100
):
    if not chunked_langchain_docs:
        logger.info("No documents to ingest into MongoDB.")
        return
    if mongo_kb_collection is None:
        logger.error("MongoDB knowledge base collection is not initialized. Cannot ingest.")
        return
    
    if clear_collection:
        try:
            logger.info(f"Clearing existing documents from MongoDB collection: {MONGO_KB_COLLECTION_NAME}")
            delete_result = mongo_kb_collection.delete_many({})
            logger.info(f"Successfully deleted {delete_result.deleted_count} documents.")
        except Exception as e:
            logger.error(f"Could not clear MongoDB collection {MONGO_KB_COLLECTION_NAME}: {e}", exc_info=True)

    num_chunks = len(chunked_langchain_docs)
    logger.info(f"Preparing to ingest {num_chunks} chunks into MongoDB collection '{MONGO_KB_COLLECTION_NAME}'.")


    # --- Important Note for User ---
    logger.warning("="*80)
    logger.warning("IMPORTANT: Before running queries, ensure you have created a VECTOR SEARCH INDEX in MongoDB Atlas on the "
                   f"'{MONGO_KB_COLLECTION_NAME}' collection for the 'embedding_vector' field.")
    logger.warning(f"The index should use 'cosine' similarity and have {EMBEDDING_DIMENSION} dimensions.")
    logger.warning("Example Atlas Index Definition (JSON):")
    logger.warning(json.dumps({
        "fields": [
            {
                "type": "vector",
                "path": "embedding_vector",
                "numDimensions": EMBEDDING_DIMENSION,
                "similarity": "cosine"
            }
            # You can add other fields to index for filtering if needed
            # {
            #   "type": "filter",
            #   "path": "source_document"
            # }
        ]
    }, indent=2))
    logger.warning("="*80)
    # --- End Important Note ---

    for i in range(0, num_chunks, batch_size):
        batch_of_langchain_docs = chunked_langchain_docs[i:i + batch_size]
        documents_for_mongo_batch = []
        texts_to_embed = [doc.page_content for doc in batch_of_langchain_docs]
        try:
            logger.debug(f"Generating embeddings for batch of {len(texts_to_embed)} texts...")
            batch_embeddings = embedding_model.encode(texts_to_embed)
            logger.debug("Embeddings generated for batch.")
        except Exception as e:
            logger.error(f"Error generating embeddings for a batch: {e}", exc_info=True)
            continue # Skip this batch

        for idx, lc_doc in enumerate(batch_of_langchain_docs):
            text_chunk = lc_doc.page_content
            embedding_vector = batch_embeddings[idx].tolist() # Convert numpy array to list for MongoDB

            # Generate a stable ID
            source_file_path = lc_doc.metadata.get("source", "unknown_source")
            chunk_id = generate_stable_id(text_chunk, source_file_path)

            metadata_to_store = {
                "source_document": Path(source_file_path).name,
                # Add other relevant metadata from lc_doc.metadata
            }
            if 'page' in lc_doc.metadata:
                metadata_to_store['page_number'] = lc_doc.metadata['page']
            if 'start_index' in lc_doc.metadata:
                 metadata_to_store['start_index_in_doc'] = lc_doc.metadata['start_index']


            mongo_doc = {
                "_id": chunk_id, # Using content-based hash as ID for potential upsert behavior
                "text_chunk": text_chunk,
                "embedding_vector": embedding_vector,
                "metadata": metadata_to_store
            }
            documents_for_mongo_batch.append(mongo_doc)

        if not documents_for_mongo_batch:
            logger.info(f"Skipping empty MongoDB batch {i//batch_size + 1} (all chunks might have had issues).")
            continue

        try:
            logger.info(f"Adding MongoDB batch {i//batch_size + 1}/{(num_chunks + batch_size -1)//batch_size} with {len(documents_for_mongo_batch)} documents...")
            # Using insert_many. For upsert, you'd need a loop with UpdateOne and upsert=True
            mongo_kb_collection.insert_many(documents_for_mongo_batch, ordered=False)
            logger.info(f"Successfully added batch to MongoDB collection '{MONGO_KB_COLLECTION_NAME}'.")
        except Exception as e: # Catch BulkWriteError specifically if needed
            logger.error(f"Error adding batch to MongoDB: {e}", exc_info=True)
            logger.error(f"First document in failing batch (preview): {str(documents_for_mongo_batch[0])[:500]}...")


    logger.info(f"Finished ingesting all processable chunks. Final count in '{MONGO_KB_COLLECTION_NAME}': {mongo_kb_collection.count_documents({})}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest documents into MongoDB for Vector Search.")
    parser.add_argument("--docs_path", type=str, required=True, help="Path to the directory containing PDF documents to ingest.")
    # Collection name is now fixed as MONGO_KB_COLLECTION_NAME
    parser.add_argument("--chunk_size", type=int, default=1000, help="Target size for text chunks (default: 1000).")
    parser.add_argument("--chunk_overlap", type=int, default=150, help="Overlap between text chunks (default: 150).")
    parser.add_argument("--clear", action="store_true", help="Clear the MongoDB collection before ingesting new documents.")
    parser.add_argument("--batch_size", type=int, default=100, help="Number of documents to insert into MongoDB in a single batch (default: 100).")

    args = parser.parse_args()

    try:
        # Embedding model is initialized globally now
        langchain_document_chunks = load_and_chunk_documents(args.docs_path, args.chunk_size, args.chunk_overlap)
        
        if langchain_document_chunks:
            ingest_into_mongodb(
                langchain_document_chunks,
                clear_collection=args.clear,
                batch_size=args.batch_size
            )
    except Exception as e:
        logger.error(f"An error occurred during the ingestion process: {e}", exc_info=True)