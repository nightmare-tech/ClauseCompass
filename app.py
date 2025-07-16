import logging
import os
import json
import tempfile
from pathlib import Path
from fastapi import FastAPI, Depends, HTTPException, status, UploadFile, File
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
app = FastAPI()
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")

# --- Environment Variable Loading & Validation ---
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
# Renamed for clarity, as discussed
class QueryRequest(BaseModel):
    query_text: str
    source_files: Optional[List[str]] = Field(default_factory=list)

class RegisterUser(BaseModel):
    userid: str
    emailid: str
    password: str

# --- Utility Functions (Auth) ---
# ... (hash_password, verify_password, create_access_token, decode_access_token, get_current_user remain the same) ...
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

# --- API Endpoints ---

# Renamed for clarity
@app.post("/evaluate", summary="Evaluate a query against the persistent Knowledge Base")
async def evaluate_endpoint(
    query_req: QueryRequest,
    current_user: dict = Depends(get_current_user)
):
    """
    Handles queries against the pre-ingested, persistent knowledge base in MongoDB.
    This is a stateless, transactional endpoint.
    """
    user_query = query_req.query_text
    structured_response = {}
    used_knowledge_base = False

    if app_embedding_model is None:
        logger.error("Embedding model not available. Cannot perform KB lookup.")
        raise HTTPException(status_code=503, detail="AI embedding service is unavailable.")

    try:
        user_query_embedding = app_embedding_model.encode(user_query).tolist()

        vector_search_stage = {
            "$vectorSearch": {
                "index": ATLAS_VECTOR_SEARCH_INDEX_NAME,
                "path": "embedding_vector",
                "queryVector": user_query_embedding,
                "numCandidates": 100,
                "limit": 5
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
                context_for_llm_str += f"Context Document {i+1} (Source: {doc.get('metadata', {}).get('source_document', 'N/A')}):\n{doc.get('text_chunk', '')}\n\n"
            context_for_llm_str += "--- End of Provided Documents Context ---\n"

            system_prompt_rag = (
                "You are a precise insurance claims adjudicator API. Your only function is to return a structured JSON object. Do NOT add any text before or after the JSON.\n\n"
                "TASK: Follow these steps precisely:\n"
                "1.  **Analyze the User Query:** Identify the user's condition/procedure and the circumstances (e.g., illness vs. accident).\n"
                "2.  **Review the Provided Context:** Scan all provided policy clauses for relevance to the user's condition and circumstances.\n"
                "3.  **Check for Exclusions & Waiting Periods:** First, determine if the condition is explicitly excluded or falls under a waiting period (e.g., 30-day, 24-month specified disease).\n"
                "4.  **CRUCIAL - Check for Exceptions:** If a waiting period or exclusion applies, you MUST re-scan the context to see if there is an exception to that rule (e.g., 'unless necessitated due to an Accident'). The presence of an accident in the user query is a critical exception.\n"
                "5.  **Formulate Decision:** Based on this step-by-step evaluation, make your final 'Decision'.\n"
                "6.  **Construct JSON:** Populate the JSON with your final decision. For the 'Justification', list EVERY relevant clause you considered, both for and against the decision, and cite the exact 'Citation Reference' for each.\n\n"
                "JSON SCHEMA TO FOLLOW:\n"
                "SCHEMA: {\"Decision\": \"...\", \"Amount\": ..., \"Justification\": [{\"reason\": \"...\", \"clause\": \"...\"}]}\n\n"
                "\n\n"
                f"CONTEXT:\n{context_for_llm_str}"
            )
            final_user_query = (
                f"User query: '{user_query}'.\n\n"
                "Now, generate ONLY the raw JSON object as your response."
            )

            final_messages_for_ai21 = [
                ChatMessage(role="system", content=system_prompt_rag),
                ChatMessage(role="user", content=final_user_query)
            ]

            ai_api_response = ai21_client.chat.completions.create(model="jamba-mini-1.6-2025-03", messages=final_messages_for_ai21)
            response_json_str = ai_api_response.choices[0].message.content
            logger.info(f"Raw LLM Response String: {response_json_str}")
            try:
                start_index = response_json_str.find('{')
                end_index = response_json_str.rfind('}')
                if start_index != -1 and end_index != -1:
                    # Extract the substring that is the actual JSON
                    clean_json_str = response_json_str[start_index : end_index + 1]
                    logger.info(f"Cleaned JSON String for parsing: {clean_json_str}")
                    structured_response = json.loads(clean_json_str)
                else:
                    raise json.JSONDecodeError("Could not find JSON object in LLM response", response_json_str, 0)
            except json.JSONDecodeError as e:
                logger.error(f"LLM did not return valid JSON, even after cleaning. Raw response: {response_json_str}", exc_info=True)
                structured_response = {"Decision": "Error", "Amount": 0, "Justification": "AI failed to generate a valid structured response."}

            
        else:
            structured_response = {"Decision": "Cannot Determine", "Amount": 0, "Justification": OUT_OF_KB_SCOPE_MESSAGE}

    # except json.JSONDecodeError:
    #     logger.error("LLM did not return valid JSON in /chat endpoint.")
    #     structured_response = {"Decision": "Error", "Amount": 0, "Justification": "AI failed to generate a structured response."}
    except Exception as e:
        logger.error(f"Error during RAG process in /chat: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="An error occurred while processing your request.")

    # Store transaction log
    users_collection.update_one(
        {"userid": current_user["userid"]},
        {"$push": {"chat_history": {"$each": [{"role": "user", "content": user_query, "timestamp": datetime.now(timezone.utc)}, {"role": "assistant", "content": structured_response, "timestamp": datetime.now(timezone.utc), "used_knowledge_base": used_knowledge_base}]}}}
    )

    return structured_response

@app.post("/evaluate-with-docs", summary="Evaluate a query with dynamically uploaded documents")
async def evaluate_with_uploaded_documents(
    query: str = File(...), # Use File to send with multipart
    files: List[UploadFile] = File(...),
    current_user: dict = Depends(get_current_user)
):
    """
    Handles on-the-fly RAG for user-uploaded documents.
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files were uploaded.")
    if app_embedding_model is None:
        raise HTTPException(status_code=503, detail="AI embedding service is unavailable.")

    # ... (The logic for this endpoint was already clean and does not need changes) ...
    chunked_documents = []
    temp_files = []
    try:
        # The existing logic is correct and self-contained
        for uploaded_file in files:
            with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.filename).suffix) as tmp:
                tmp.write(await uploaded_file.read())
                temp_files.append(tmp.name)
            
            file_extension = Path(tmp.name).suffix.lower()
            if file_extension == ".pdf": loader = PyPDFLoader(tmp.name)
            else: loader = TextLoader(tmp.name, encoding='utf-8')
            
            docs_from_file = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
            chunks = text_splitter.split_documents(docs_from_file)
            for chunk in chunks: chunk.metadata["source"] = uploaded_file.filename
            chunked_documents.extend(chunks)

        if not chunked_documents: raise HTTPException(status_code=400, detail="Could not process any uploaded documents.")

        temp_chroma_client = chromadb.Client()
        temp_collection = temp_chroma_client.create_collection(name="hackathon_temp_rag")
        
        docs_to_embed = [chunk.page_content for chunk in chunked_documents]
        embeddings_to_add = app_embedding_model.encode(docs_to_embed).tolist()

        temp_collection.add(
            ids=[f"{chunk.metadata.get('source', 'doc')}_chunk_{i}" for i, chunk in enumerate(chunked_documents)],
            embeddings=embeddings_to_add,
            documents=docs_to_embed,
            metadatas=[chunk.metadata for chunk in chunked_documents]
        )
        retrieved_docs = temp_collection.query(query_texts=[query], n_results=5)
        retrieved_docs_content = retrieved_docs.get('documents', [[]])[0]

        if not retrieved_docs_content: return {"Decision": "Cannot Determine", "Amount": 0, "Justification": "No relevant information found in the uploaded documents for the given query."}

        context_for_llm_str = "\n--- Provided Documents Context ---\n"
        for doc_text in retrieved_docs_content: context_for_llm_str += f"Context: {doc_text}\n\n"
        context_for_llm_str += "--- End of Provided Documents Context ---\n"

        system_prompt_rag = (
            "You are an expert insurance claims adjudicator. Your task is to make a decision based on the user's query and the provided context from the uploaded documents. "
            "Evaluate the user's details against the policy clauses found in the context. If the context is insufficient, state that. "
            "Do NOT use any external knowledge. ONLY output your final decision in a structured JSON format with keys: 'Decision', 'Amount', and 'Justification'."
            f"{context_for_llm_str}"
        )
        final_messages_for_ai21 = [ChatMessage(role="system", content=system_prompt_rag), ChatMessage(role="user", content=query)]
        
        ai_api_response = ai21_client.chat.completions.create(model="jamba-mini-1.6-2025-03", messages=final_messages_for_ai21)
        response_json_str = ai_api_response.choices[0].message.content
        return json.loads(response_json_str)

    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="Failed to get a structured response from the AI.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {e}")
    finally:
        for temp_file_path in temp_files:
            if os.path.exists(temp_file_path): os.unlink(temp_file_path)


@app.get("/documents", summary="List available documents in the persistent KB")
# ... (this endpoint is fine) ...
async def list_available_documents(current_user: dict = Depends(get_current_user)):
    try:
        distinct_files = mongo_kb_collection.distinct("metadata.source_document")
        return {"documents": distinct_files}
    except Exception as e:
        logger.error(f"Error fetching distinct documents from MongoDB: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Could not retrieve document list.")


# --- Authentication Endpoints ---
@app.post("/register")
async def register_user_endpoint(registration_data: RegisterUser):
    # ... (this endpoint is fine, but we'll apply the chat_history: [] change) ...
    if users_collection.find_one({"emailid": registration_data.emailid}): raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="User with this email already exists")
    if users_collection.find_one({"userid": registration_data.userid}): raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="User with this userid already exists.")

    hashed_pass = hash_password(registration_data.password)
    user_document = {
        "userid": registration_data.userid, 
        "emailid": registration_data.emailid, 
        "password": hashed_pass, 
        "chat_history": [], # CORRECT: Initialize as empty
        "created_at": datetime.now(timezone.utc) # Good practice
    }
    users_collection.insert_one(user_document)
    return {"message": "User registered successfully", "userid": registration_data.userid}


@app.post("/login")
# ... (this endpoint is fine) ...
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    user = users_collection.find_one({"emailid": form_data.username})
    if not user or not verify_password(form_data.password, user["password"]):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Incorrect email or password", headers={"WWW-Authenticate": "Bearer"})
    token = create_access_token(data={"sub": user["userid"]})
    return {"access_token": token, "token_type": "bearer"}


# --- Main Guard ---
if __name__ == "__main__":
    import uvicorn
    logger.info("Starting FastAPI application with Uvicorn...")
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)