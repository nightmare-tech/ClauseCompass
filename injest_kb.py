import os
import logging
import chromadb # Import the main chromadb library
# No need for chromadb.utils.embedding_functions if we only use the default
from pathlib import Path
from dotenv import load_dotenv
import argparse
import json
import re
import uuid
import hashlib

from langchain_community.document_loaders import PyPDFLoader 
from langchain.text_splitter import RecursiveCharacterTextSplitter

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
load_dotenv()

CHROMA_DB_PATH = "./chroma_db_store"
DEFAULT_COLLECTION_NAME = "company_internal_kb"

      
def load_and_chunk_documents(docs_path: str, chunk_size: int = 1000, chunk_overlap: int = 150):
    """
    Loads ONLY PDF documents from the specified path using PyPDFLoader and chunks them.
    """
    logger.info(f"Loading ONLY PDF documents from: {docs_path} using PyPDFLoader.")
    source_path = Path(docs_path)
    
    pdf_files = sorted(list(source_path.rglob("*.pdf")))

    if not pdf_files:
        logger.warning(f"No PDF files found in {docs_path}. Exiting document loading.")
        return []
    
    logger.info(f"Found {len(pdf_files)} PDF files to process.")
    
    all_pages = []
    for file_path in pdf_files:
        try:
            logger.info(f"Loading {file_path}...")
            loader = PyPDFLoader(str(file_path))
            # .load() returns a list of Document objects, one for each page
            pages = loader.load() 
            all_pages.extend(pages)
            logger.info(f"Successfully loaded {len(pages)} pages from {file_path}.")
        except Exception as e:
            logger.error(f"Error loading PDF file {file_path}: {e}", exc_info=False)

    if not all_pages:
        logger.warning("No pages were successfully loaded from any PDF files. Exiting.")
        return []

    logger.info(f"Loaded a total of {len(all_pages)} pages. Now splitting into smaller chunks.")
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        add_start_index=True
    )
    
    # PyPDFLoader already puts the source file path in metadata, so this will be preserved.
    chunked_documents = text_splitter.split_documents(all_pages)
    logger.info(f"Split loaded pages into {len(chunked_documents)} final chunks.")
    
    return chunked_documents

    

def ingest_into_chroma(
    collection_name: str,
    chunked_docs: list,
    clear_collection: bool = False,
    batch_size: int = 100
):
    if not chunked_docs:
        logger.info("No documents to ingest.")
        return

    logger.info(f"Initializing ChromaDB client for path: {CHROMA_DB_PATH}")
    chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)

    if clear_collection:
        try:
            logger.info(f"Attempting to delete existing collection: {collection_name}")
            chroma_client.delete_collection(name=collection_name)
            logger.info(f"Successfully deleted collection: {collection_name}")
        except Exception as e: # Catch specific exceptions if ChromaDB API changes
            logger.warning(f"Could not delete collection {collection_name} (it might not exist): {e}")

    logger.info(f"Getting or creating ChromaDB collection: {collection_name}. ChromaDB will use its default embedding function.")
    # By NOT providing embedding_function, ChromaDB uses its default (SentenceTransformer all-MiniLM-L6-v2)
    collection = chroma_client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"} # Still good to specify distance metric for consistency
    )

    num_chunks = len(chunked_docs)
    added_ids_in_current_run = set()

    for i in range(0, num_chunks, batch_size):
        batch_chunks = chunked_docs[i:i + batch_size]
        ids_for_batch = []
        documents_for_batch = []
        metadatas_for_batch = []

        for chunk_in_batch_idx, chunk in enumerate(batch_chunks):
            original_full_path = chunk.metadata.get("_original_file_path", f"unknown_path_{uuid.uuid4()}")
            
            chunk_id_str_for_hash = f"{original_full_path}::{chunk.page_content}"
            chunk_id = hashlib.md5(chunk_id_str_for_hash.encode('utf-8')).hexdigest()

            if chunk_id in added_ids_in_current_run:
                logger.warning(f"Duplicate ID generated within current run: {chunk_id} for path {original_full_path}. Content start: '{chunk.page_content[:50]}...'. Skipping this specific chunk.")
                continue
            added_ids_in_current_run.add(chunk_id)

            ids_for_batch.append(chunk_id)
            documents_for_batch.append(chunk.page_content)
            
            current_metadata = {}
            current_metadata["source"] = chunk.metadata.get("source", Path(original_full_path).name)

            for key, value in chunk.metadata.items():
                if key in ['source', '_original_file_path', 'start_index', 'filename']:
                    continue
                if isinstance(value, (str, int, float, bool)) or value is None:
                    current_metadata[key] = value
                elif isinstance(value, (dict, list, tuple)):
                    try:
                        current_metadata[key] = json.dumps(value)
                    except TypeError:
                        current_metadata[key] = str(value)
                else:
                    current_metadata[key] = str(value)
            metadatas_for_batch.append(current_metadata)

        if not ids_for_batch:
            logger.info(f"Skipping empty batch {i//batch_size + 1} (all chunks might have been duplicates).")
            continue

        logger.info(f"Adding batch {i//batch_size + 1}/{(num_chunks + batch_size -1)//batch_size} with {len(ids_for_batch)} chunks to collection '{collection_name}'...")
        try:
            # ChromaDB will automatically embed documents_for_batch using its default EF
            collection.add(
                ids=ids_for_batch,
                documents=documents_for_batch,
                metadatas=metadatas_for_batch
            )
            logger.info(f"Successfully added batch to '{collection_name}'.")
        except Exception as e:
            logger.error(f"Error adding batch to ChromaDB: {e}. First 5 IDs in batch: {ids_for_batch[:5]}...")
            if metadatas_for_batch:
                 logger.error(f"First metadata object in failing batch: {metadatas_for_batch[0]}")

    logger.info(f"Finished ingesting all processable chunks into {collection_name}. Collection count: {collection.count()}")

# --- Main execution block (argparse) ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest documents into ChromaDB using its default embedding function.")
    parser.add_argument("--docs_path", type=str, required=True, help="Path to the directory containing documents to ingest.")
    parser.add_argument("--collection_name", type=str, default=DEFAULT_COLLECTION_NAME, help=f"Name of the ChromaDB collection (default: {DEFAULT_COLLECTION_NAME}).")
    parser.add_argument("--chunk_size", type=int, default=1000, help="Target size for text chunks (default: 1000).")
    parser.add_argument("--chunk_overlap", type=int, default=150, help="Overlap between text chunks (default: 150).")
    parser.add_argument("--clear", action="store_true", help="Clear the collection before ingesting new documents.")
    parser.add_argument("--batch_size", type=int, default=100, help="Number of chunks to add to ChromaDB in a single batch (default: 100).")

    args = parser.parse_args()

    try:
        chunked_docs = load_and_chunk_documents(args.docs_path, args.chunk_size, args.chunk_overlap)
        
        if chunked_docs:
            ingest_into_chroma(
                args.collection_name,
                chunked_docs,
                # No embedding_function passed here, relying on default
                clear_collection=args.clear,
                batch_size=args.batch_size
            )
    except Exception as e:
        logger.error(f"An error occurred during the ingestion process: {e}", exc_info=True)