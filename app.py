import logging
import os
import json
import tempfile
import uuid
import re
from pathlib import Path
from fastapi import FastAPI, Depends, HTTPException, status, UploadFile, File, BackgroundTasks
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from datetime import datetime, timedelta, timezone
from jose import JWTError, jwt
from passlib.context import CryptContext
from pymongo.mongo_client import MongoClient as MongoClientDB
from pymongo.server_api import ServerApi
from urllib.parse import quote_plus
from pydantic import BaseModel, Field
from typing import List, Optional
from ai21 import AI21Client
from ai21.models.chat import ChatMessage
from dotenv import load_dotenv

from sentence_transformers import SentenceTransformer
import chromadb
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# --- Logging Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()

# --- FastAPI App Initialization ---
app = FastAPI(title="ClauseCompass API", description="An AI-powered decision engine for document analysis.")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")

# --- Global State & Environment Variables ---
SESSION_VECTOR_STORES = {}  # For Temporary Mode
AI21_API_KEY_ENV = os.getenv("AI21_API_KEY")
MONGO_USER_ENV = os.getenv("USERN")
MONGO_PASS_ENV = os.getenv("PASSW")
JWT_SECRET_ENV = os.getenv("JWT_SECRET")

if not all([AI21_API_KEY_ENV, MONGO_USER_ENV, MONGO_PASS_ENV, JWT_SECRET_ENV]):
    raise ValueError("One or more required environment variables are missing.")

ai21_client = AI21Client(api_key=AI21_API_KEY_ENV)

# --- MongoDB Setup ---
escaped_username = quote_plus(MONGO_USER_ENV)
escaped_password = quote_plus(MONGO_PASS_ENV)
MONGO_URI = f"mongodb+srv://{escaped_username}:{escaped_password}@chattydb.dfuykzc.mongodb.net/?retryWrites=true&w=majority&appName=chattydb"
mongo_client_instance = MongoClientDB(MONGO_URI, server_api=ServerApi('1'))

try:
    mongo_client_instance.admin.command('ping')
    logger.info("Successfully connected to MongoDB!")
except Exception as e:
    logger.error(f"Failed to connect to MongoDB: {e}", exc_info=True)
    raise

db = mongo_client_instance["chattydb"]
users_collection = db["users"]
MONGO_KB_COLLECTION_NAME = "company_kb_vectorized"
mongo_kb_collection = db[MONGO_KB_COLLECTION_NAME]

# --- Embedding Model (Must match ingest_kb.py) ---
EMBEDDING_MODEL_NAME_APP = 'all-MiniLM-L6-v2'
try:
    app_embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME_APP)
    logger.info(f"FastAPI App: Initialized Embedding Model: {EMBEDDING_MODEL_NAME_APP}")
except Exception as e:
    logger.error(f"FastAPI App: Failed to load embedding model '{EMBEDDING_MODEL_NAME_APP}': {e}", exc_info=True)
    app_embedding_model = None

# --- Configuration Constants ---
ATLAS_VECTOR_SEARCH_INDEX_NAME = "vector_index_for_kb"
MONGO_SCORE_THRESHOLD = 0.60
OUT_OF_KB_SCOPE_MESSAGE = "I am designed to answer questions based on the provided documents. I do not have information on that topic."

# --- Security and JWT Configuration ---
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60

# --- Pydantic Models ---
class QueryRequest(BaseModel):
    query_text: str
    source_files: Optional[List[str]] = Field(default_factory=list)

class RegisterUser(BaseModel):
    userid: str
    emailid: str
    password: str

# --- Utility Functions ---
def hash_password(password: str): return pwd_context.hash(password)
def verify_password(plain: str, hashed: str): return pwd_context.verify(plain, hashed)
def create_access_token(data: dict, expires_delta: timedelta = None):
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, JWT_SECRET_ENV, algorithm=ALGORITHM)
def decode_access_token(token: str):
    try: return jwt.decode(token, JWT_SECRET_ENV, algorithms=[ALGORITHM])
    except JWTError: return None
def get_current_user(token: str = Depends(oauth2_scheme)):
    payload = decode_access_token(token)
    if payload is None: raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid authentication credentials", headers={"WWW-Authenticate": "Bearer"})
    userid: str = payload.get("sub")
    if userid is None: raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid authentication credentials (no subject)", headers={"WWW-Authenticate": "Bearer"})
    user_document = users_collection.find_one({"userid": userid})
    if user_document is None: raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")
    return user_document

def clean_and_parse_json(llm_response_str: str) -> dict:
    logger.info(f"Raw LLM Response String: {llm_response_str}")
    try:
        json_match = re.search(r'\{.*\}', llm_response_str, re.DOTALL)
        if json_match:
            clean_json_str = json_match.group(0)
            logger.info(f"Cleaned JSON String for parsing: {clean_json_str}")
            return json.loads(clean_json_str)
        else:
            raise json.JSONDecodeError("Could not find JSON object in LLM response", llm_response_str, 0)
    except json.JSONDecodeError:
        logger.error(f"LLM did not return valid JSON, even after cleaning. Raw response: {llm_response_str}", exc_info=True)
        return {"Decision": "Error", "Amount": 0, "Justification": "AI failed to generate a valid structured response."}

# --- API Endpoints ---

@app.post("/evaluate", summary="Evaluate a query against the persistent Knowledge Base")
async def evaluate_endpoint(
    query_req: QueryRequest,
    current_user: dict = Depends(get_current_user)
):
    user_query = query_req.query_text
    structured_response = {}
    used_knowledge_base = False

    if app_embedding_model is None:
        raise HTTPException(status_code=503, detail="AI embedding service is unavailable.")

    try:
        user_query_embedding = app_embedding_model.encode(user_query).tolist()

        vector_search_stage = {
            "$vectorSearch": {
                "index": ATLAS_VECTOR_SEARCH_INDEX_NAME, "path": "embedding_vector",
                "queryVector": user_query_embedding, "numCandidates": 100, "limit": 5
            }
        }
        if query_req.source_files:
            vector_search_stage["$vectorSearch"]["filter"] = { "metadata.source_document": { "$in": query_req.source_files } }
        
        vector_search_pipeline = [vector_search_stage, {"$project": {"_id": 0, "text_chunk": 1, "metadata": 1, "score": {"$meta": "vectorSearchScore"}}}]
        retrieved_mongo_results = list(mongo_kb_collection.aggregate(vector_search_pipeline))

        if retrieved_mongo_results and retrieved_mongo_results[0]['score'] >= MONGO_SCORE_THRESHOLD:
            used_knowledge_base = True
            logger.info(f"Relevant KB context found from MongoDB (top score: {retrieved_mongo_results[0]['score']:.4f}).")

            context_for_llm_str = "\n--- Provided Documents Context ---\n"
            for i, doc in enumerate(retrieved_mongo_results):
                metadata = doc.get('metadata', {})
                source_doc = metadata.get('source_document', 'N/A')
                clause_section = metadata.get('policy_section', '')
                clause_num = metadata.get('policy_clause_num', '')
                citation_ref = f"[{clause_section}, Clause {clause_num} (Source: {source_doc})]"
                context_for_llm_str += f"Citation Reference: {citation_ref}\n"
                context_for_llm_str += f"Content: {doc.get('text_chunk', '')}\n\n"
            context_for_llm_str += "--- End of Provided Documents Context ---\n"
            logger.info(f"Context provided to llm\n: {context_for_llm_str}")

            system_prompt_rag = (
                "You are a stateless, rule-based JSON generation API. Your ONLY function is to analyze a user query and provided context and return a single, valid JSON object. "
                "Your entire response MUST be the JSON object itself, with no other text.\n\n"

                "REASONING RULES:\n"
                "1.  **Analyze Query & Context:** Base your decision ONLY on the provided context and the user's query. Do NOT assume facts not explicitly stated (e.g., do not assume an 'Accident' if not mentioned).\n\n"
                "2.  **Check for Overriding Rules First:** Before approving, you MUST check for specific conditions that would reject the claim:\n"
                "    - **Definitions:** Does the situation violate a core definition (e.g., is the location a 'health spa' instead of a 'Hospital')?\n"
                "    - **Waiting Periods:** Does the claim fall within the 30-day, 24-month, or 36-month waiting periods?\n"
                "    - **Exclusions:** Is the condition explicitly excluded (e.g., 'Cosmetic Surgery', 'Hazardous Sports')?\n\n"
                "3.  **Prioritize Exceptions:** If a waiting period or exclusion applies, you MUST check for an exception. An **'Accident'** is a critical exception that overrides most waiting periods.\n\n"
                "4.  **Handle Financials:** Check for specific financial rules like **Sub-limits** (e.g., for Robotic Surgery), **Co-payments** (e.g., Zone-based), or **Deductions** (e.g., Room Rent). If these apply, the 'Decision' MUST be 'Partially Approved' or similar. If no specific amount or percentage is in the context, the 'Amount' MUST be null.\n\n"

                "JSON OUTPUT REQUIREMENTS:\n"
                "-   Return ONLY the JSON. 'Decision' must be one of 'Approved', 'Rejected', 'Partially Approved', 'Conditional'.\n"
                "-   The 'Justification' must cite the specific reason and clause from the context.\n\n"
                
                f"CONTEXT:\n{context_for_llm_str}"
            )
            final_user_query = (
                f"User Query: '{user_query}'.\n\n"
                "=== TASK ===\n"
                "Based on the query and the context provided in the system prompt, generate ONLY the raw JSON object as your response."
            )
            final_messages_for_ai21 = [
                ChatMessage(role="system", content=system_prompt_rag),
                ChatMessage(role="user", content=final_user_query)
            ]
            
            ai_api_response = ai21_client.chat.completions.create(model="jamba-mini-1.6-2025-03", messages=final_messages_for_ai21)
            structured_response = clean_and_parse_json(ai_api_response.choices[0].message.content)
        else:
            structured_response = {"Decision": "Cannot Determine", "Amount": 0, "Justification": OUT_OF_KB_SCOPE_MESSAGE}

    except Exception as e:
        logger.error(f"Error during RAG process in /evaluate: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="An error occurred while processing your request.")

    users_collection.update_one(
        {"userid": current_user["userid"]},
        {"$push": {"chat_history": {"$each": [{"role": "user", "content": user_query, "timestamp": datetime.now(timezone.utc)}, {"role": "assistant", "content": structured_response, "timestamp": datetime.now(timezone.utc), "used_knowledge_base": used_knowledge_base}]}}}
    )
    return structured_response

@app.post("/session/documents", summary="Upload documents to start a temporary RAG session")
async def upload_documents_for_session(
    files: List[UploadFile] = File(...),
    current_user: dict = Depends(get_current_user),
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    user_id = current_user["userid"]
    if user_id in SESSION_VECTOR_STORES:
        logger.info(f"Clearing previous session for user: {user_id}")
        del SESSION_VECTOR_STORES[user_id]
    if not files: raise HTTPException(status_code=400, detail="No files were uploaded.")

    chunked_documents = []; temp_files_to_clean = []
    try:
        for uploaded_file in files:
            with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.filename).suffix) as tmp:
                tmp.write(await uploaded_file.read()); temp_files_to_clean.append(tmp.name)
            
            file_extension = Path(tmp.name).suffix.lower()
            if file_extension == ".pdf": loader = PyPDFLoader(tmp.name)
            else: loader = TextLoader(tmp.name, encoding='utf-8')
            
            docs_from_file = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
            chunks = text_splitter.split_documents(docs_from_file)
            for chunk in chunks: chunk.metadata["source"] = uploaded_file.filename
            chunked_documents.extend(chunks)

        if not chunked_documents: raise HTTPException(status_code=400, detail="Could not process any uploaded documents.")

        session_chroma_client = chromadb.Client()
        collection_name = f"session_collection_{user_id}_{uuid.uuid4().hex}"
        session_collection = session_chroma_client.create_collection(name=collection_name)
        
        docs_to_embed = [chunk.page_content for chunk in chunked_documents]
        embeddings_to_add = app_embedding_model.encode(docs_to_embed).tolist()

        session_collection.add(
            ids=[f"chunk_{i}" for i in range(len(chunked_documents))],
            embeddings=embeddings_to_add, documents=docs_to_embed, metadatas=[chunk.metadata for chunk in chunked_documents]
        )
        
        SESSION_VECTOR_STORES[user_id] = session_collection
        logger.info(f"Successfully created and cached vector store for user: {user_id}")
        return {"message": f"Successfully processed {len(files)} documents. You can now query them.", "session_user": user_id}
    except Exception as e:
        logger.error(f"Error creating session vector store for user {user_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to create session knowledge base.")
    finally:
        for path in temp_files_to_clean: background_tasks.add_task(os.unlink, path)

@app.post("/session/query", summary="Query the current temporary session")
async def query_session_documents(
    query_req: QueryRequest,
    current_user: dict = Depends(get_current_user)
):
    user_id = current_user["userid"]
    user_query = query_req.query_text
    if user_id not in SESSION_VECTOR_STORES:
        raise HTTPException(status_code=404, detail="No active document session found. Please upload documents first.")

    session_collection = SESSION_VECTOR_STORES[user_id]
    try:
        retrieved_docs = session_collection.query(query_texts=[user_query], n_results=5)
        retrieved_docs_content = retrieved_docs.get('documents', [[]])[0]
        
        if not retrieved_docs_content:
            return {"Decision": "Cannot Determine", "Amount": 0, "Justification": "No relevant information found in uploaded documents for this query."}

        context_for_llm_str = "\n--- Provided Documents Context ---\n"
        for doc_text in retrieved_docs_content: context_for_llm_str += f"Context: {doc_text}\n\n"
        context_for_llm_str += "--- End of Provided Documents Context ---\n"
        
        # A new, more concise "Medium" prompt

        system_prompt_rag = (
            "You are a stateless, rule-based JSON generation API. Your ONLY function is to analyze a user query and provided context and return a single, valid JSON object. "
            "Your entire response MUST be the JSON object itself, with no other text.\n\n"

            "REASONING RULES:\n"
            "1.  **Analyze Query & Context:** Base your decision ONLY on the provided context and the user's query. Do NOT assume facts not explicitly stated (e.g., do not assume an 'Accident' if not mentioned).\n\n"
            "2.  **Check for Overriding Rules First:** Before approving, you MUST check for specific conditions that would reject the claim:\n"
            "    - **Definitions:** Does the situation violate a core definition (e.g., is the location a 'health spa' instead of a 'Hospital')?\n"
            "    - **Waiting Periods:** Does the claim fall within the 30-day, 24-month, or 36-month waiting periods?\n"
            "    - **Exclusions:** Is the condition explicitly excluded (e.g., 'Cosmetic Surgery', 'Hazardous Sports')?\n\n"
            "3.  **Prioritize Exceptions:** If a waiting period or exclusion applies, you MUST check for an exception. An **'Accident'** is a critical exception that overrides most waiting periods.\n\n"
            "4.  **Handle Financials:** Check for specific financial rules like **Sub-limits** (e.g., for Robotic Surgery), **Co-payments** (e.g., Zone-based), or **Deductions** (e.g., Room Rent). If these apply, the 'Decision' MUST be 'Partially Approved' or similar. If no specific amount or percentage is in the context, the 'Amount' MUST be null.\n\n"

            "JSON OUTPUT REQUIREMENTS:\n"
            "-   Return ONLY the JSON. 'Decision' must be one of 'Approved', 'Rejected', 'Partially Approved', 'Conditional'.\n"
            "-   The 'Justification' must cite the specific reason and clause from the context.\n\n"
            
            f"CONTEXT:\n{context_for_llm_str}"
        )

        # And still use the "Final Command" user message
        final_user_query = (
            f"User Query: '{user_query}'.\n\n"
            "=== TASK ===\n"
            "Based on the query and the context provided, generate ONLY the raw JSON object as your response."
        )
       
        final_messages_for_ai21 = [ChatMessage(role="system", content=system_prompt_rag), ChatMessage(role="user", content=final_user_query)]
        
        ai_api_response = ai21_client.chat.completions.create(model="jamba-mini-1.6-2025-03", messages=final_messages_for_ai21)
        return clean_and_parse_json(ai_api_response.choices[0].message.content)

    except Exception as e:
        logger.error(f"Error querying session for user {user_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="An error occurred during session query.")

@app.get("/documents", summary="List available documents in the persistent KB")
async def list_available_documents(current_user: dict = Depends(get_current_user)):
    try:
        distinct_files = mongo_kb_collection.distinct("metadata.source_document")
        return {"documents": distinct_files}
    except Exception as e:
        logger.error(f"Error fetching distinct documents from MongoDB: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Could not retrieve document list.")

@app.post("/register")
async def register_user_endpoint(registration_data: RegisterUser):
    if users_collection.find_one({"emailid": registration_data.emailid}): raise HTTPException(status_code=400, detail="User with this email already exists")
    if users_collection.find_one({"userid": registration_data.userid}): raise HTTPException(status_code=400, detail="User with this userid already exists.")
    
    hashed_pass = hash_password(registration_data.password)
    user_document = {
        "userid": registration_data.userid, "emailid": registration_data.emailid, 
        "password": hashed_pass, "chat_history": [],
        "created_at": datetime.now(timezone.utc)
    }
    users_collection.insert_one(user_document)
    return {"message": "User registered successfully", "userid": registration_data.userid}

@app.post("/login")
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    user = users_collection.find_one({"emailid": form_data.username})
    if not user or not verify_password(form_data.password, user["password"]):
        raise HTTPException(status_code=401, detail="Incorrect email or password", headers={"WWW-Authenticate": "Bearer"})
    token = create_access_token(data={"sub": user["userid"]})
    return {"access_token": token, "token_type": "bearer"}

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting FastAPI application with Uvicorn...")
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)