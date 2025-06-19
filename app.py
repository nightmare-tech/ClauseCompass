import logging
import os
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from datetime import datetime, timedelta, timezone # Added timezone
from jose import JWTError, jwt
from passlib.context import CryptContext
from pymongo.mongo_client import MongoClient as MongoClientDB
from pymongo.server_api import ServerApi
from urllib.parse import quote_plus
from pydantic import BaseModel
from ai21 import AI21Client
from ai21.models.chat import ChatMessage
from dotenv import load_dotenv
import json
# from starlette.types import HTTPExceptionHandler # Unused, can be removed
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()

app = FastAPI()
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")

# --- Environment Variable Loading & Validation ---
AI21_API_KEY_ENV = os.getenv("AI21_API_KEY")
MONGO_USER_ENV = os.getenv("USERN")
MONGO_PASS_ENV = os.getenv("PASSW")
JWT_SECRET_ENV = os.getenv("JWT_SECRET")

if not AI21_API_KEY_ENV:
    raise ValueError("AI21_API_KEY not found in environment variables.")
if not MONGO_USER_ENV:
    raise ValueError("MongoDB username (USERN) not found in environment variables.")
if not MONGO_PASS_ENV:
    raise ValueError("MongoDB password (PASSW) not found in environment variables.")
if not JWT_SECRET_ENV:
    raise ValueError("JWT_SECRET not found in environment variables.")

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
MONGO_KB_COLLECTION_NAME = "company_kb_vectorized" # Same as in ingest_kb.py
mongo_kb_collection = db[MONGO_KB_COLLECTION_NAME]
EMBEDDING_MODEL_NAME_APP = 'all-MiniLM-L6-v2'
try:
    app_embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME_APP)
    logger.info(f"FastAPI App: Initialized Sentence Transformer embedding model: {EMBEDDING_MODEL_NAME_APP}")
except Exception as e:
    logger.error(f"FastAPI App: Failed to load embedding model '{EMBEDDING_MODEL_NAME_APP}': {e}", exc_info=True)
    app_embedding_model = None

ATLAS_VECTOR_SEARCH_INDEX_NAME = "vector_index_for_kb"
MONGO_SCORE_THRESHOLD = 0.5
OUT_OF_KB_SCOPE_MESSAGE = "I am designed to answer questions based on Company XYZ's internal documents. I do not have information on that topic."

# --- Security and JWT Configuration ---
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60

class ChatRequest(BaseModel):
    message: str

class RegisterUser(BaseModel):
    userid: str
    emailid: str
    password: str

# --- Utility Functions (Auth) ---
def hash_password(password: str):
    return pwd_context.hash(password)

def verify_password(plain: str, hashed: str):
    return pwd_context.verify(plain, hashed)

def create_access_token(data: dict, expires_delta: timedelta = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        # Use the global constant for default expiration
        expire = datetime.now(timezone.utc) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, JWT_SECRET_ENV, algorithm=ALGORITHM)

def decode_access_token(token: str):
    try:
        return jwt.decode(token, JWT_SECRET_ENV, algorithms=[ALGORITHM])
    except JWTError:
        return None

def get_current_user(token: str = Depends(oauth2_scheme)): # Renamed from get_current_active_user for consistency
    payload = decode_access_token(token)
    if payload is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    userid: str = payload.get("sub")
    if userid is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials (no subject)",
            headers={"WWW-Authenticate": "Bearer"},
        )
    user_document = users_collection.find_one({"userid": userid})
    if user_document is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")
    return user_document

# --- API Endpoints ---
@app.post("/chat")
async def chat_endpoint( # Changed to async for good practice, though AI21 might be sync
    chat_req: ChatRequest,
    current_user: dict = Depends(get_current_user)
):
    user_chat_history_from_db = current_user.get("chat_history", [])
    if not isinstance(user_chat_history_from_db, list) or not user_chat_history_from_db:
        system_prompt_timestamp = datetime.now(timezone.utc).isoformat()
        # This initial system message in history is more for context display;
        # the operational system prompt for RAG is constructed dynamically.
        user_chat_history_from_db = [
            {
                "role": "system",
                "content": "Welcome to Company XYZ support. I can answer questions based on our internal documents.",
                "timestamp": system_prompt_timestamp
            }
        ]

    user_query = chat_req.message
    ai_response_content = ""
    used_knowledge_base = False
    final_messages_for_ai21 = []

    historical_ai21_messages = []
    for msg_dict in user_chat_history_from_db:
        role = msg_dict.get("role", "user") # Default to user if role somehow missing
        content = str(msg_dict.get("content", "")) # Default to empty string if content missing
        historical_ai21_messages.append(ChatMessage(role=role, content=content))

    if app_embedding_model is None:
        logger.error("FastAPI App: Embedding model not available. Cannot perform KB lookup.")
        ai_response_content = OUT_OF_KB_SCOPE_MESSAGE
    else:
        try:
            logger.debug(f"Generating embedding for user query: '{user_query}'")
            user_query_embedding = app_embedding_model.encode(user_query).tolist() # Generate and convert to list
            # MongoDB Atlas Vector Search Pipeline
            vector_search_pipeline = [
                {
                    "$vectorSearch": {
                        "index": ATLAS_VECTOR_SEARCH_INDEX_NAME, # The name of your Atlas Vector Index
                        "path": "embedding_vector",         # Field in MongoDB containing the vectors
                        "queryVector": user_query_embedding,
                        "numCandidates": 100, # Number of candidates to consider
                        "limit": 3            # Number of top results to return
                    }
                },
                { # Project to get relevant fields and the search score
                    "$project": {
                        "_id": 0, # Exclude the default MongoDB _id
                        "text_chunk": 1,
                        "metadata": 1, # Contains source_document, page_number etc.
                        "score": {"$meta": "vectorSearchScore"}
                    }
                }
            ]
            logger.debug(f"Executing MongoDB Vector Search against '{MONGO_KB_COLLECTION_NAME}' "
                         f"with index '{ATLAS_VECTOR_SEARCH_INDEX_NAME}'.")

            retrieved_mongo_results = list(mongo_kb_collection.aggregate(vector_search_pipeline))
            logger.info(f"MongoDB Vector Search results count: {len(retrieved_mongo_results)}")
            # if retrieved_mongo_results:
            #     logger.debug(f"MongoDB Vector Search top result: {retrieved_mongo_results[0]}")


            if retrieved_mongo_results and retrieved_mongo_results[0]['score'] >= MONGO_SCORE_THRESHOLD:
                used_knowledge_base = True
                logger.info(f"Relevant KB context found from MongoDB (top score: {retrieved_mongo_results[0]['score']:.4f}). Augmenting prompt.")

                context_for_llm_str = "\n--- Provided Company XYZ Documents Context (from MongoDB) ---\n"
                for i, doc in enumerate(retrieved_mongo_results):
                    text_chunk = doc.get("text_chunk", "Error: Text chunk missing")
                    source_info = doc.get("metadata", {}).get("source_document", "Unknown source")
                    page_info = doc.get("metadata", {}).get("page_number", "")
                    context_for_llm_str += f"Context Document {i+1} (Source: {source_info}{f', Page: {page_info}' if page_info else ''}):\n{text_chunk}\n\n"
                context_for_llm_str += "--- End of Provided Documents Context ---\n"
                # logger.debug(f"Constructed context_for_llm_str (first 200 chars): {context_for_llm_str[:200]}")

                system_prompt_rag = (
                    "You are an AI assistant for Company XYZ. Your task is to answer the user's question using the information found in the 'Provided Company XYZ Documents Context' AND the CHAT HISTORY provided below ONLY. "
                    "Synthesize an answer based on these documents. "
                    "If the documents or the CHAT HISTORY do not contain enough information to answer the question, explicitly state: "
                    f"'{OUT_OF_KB_SCOPE_MESSAGE}'. "
                    "Do not use any external knowledge. Do not discuss other companies."
                    # The RAG payload now focuses on current query + context from DB
                    # History is handled in the AI21 call if their model supports it well with system prompts.
                    # For a cleaner RAG call, often history is omitted or summarized if too long.
                    # Let's send history for now as per previous logic.
                    f"CHAT HISTORY START\n\n{json.dumps([h.model_dump() if hasattr(h, 'model_dump') else h.dict() for h in historical_ai21_messages])}\n\nCHAT HISTORY END\n"
                    f"{context_for_llm_str}"
                )
                final_messages_for_ai21.append(ChatMessage(role="system", content=system_prompt_rag))
                # final_messages_for_ai21.extend(historical_ai21_messages) # History is now IN the system prompt for AI21
                final_messages_for_ai21.append(ChatMessage(role="user", content=user_query))

                # Log payload before sending
                # try: payload_log = [msg.model_dump() for msg in final_messages_for_ai21]
                # except AttributeError: payload_log = [msg.dict() for msg in final_messages_for_ai21]
                # logger.debug(f"Final messages for AI21 (MongoDB RAG path) PAYLOAD: {payload_log}")

                ai_api_response = ai21_client.chat.completions.create(
                    model="jamba-mini-1.6-2025-03",
                    messages=final_messages_for_ai21
                )
                ai_response_content = ai_api_response.choices[0].message.content
            else:
                log_score = retrieved_mongo_results[0]['score'] if retrieved_mongo_results else 'N/A'
                logger.info(f"KB context from MongoDB not found or below relevance threshold (best score: {log_score}). Responding with out-of-scope message.")
                ai_response_content = OUT_OF_KB_SCOPE_MESSAGE

        except Exception as e:
            logger.error(f"Error during MongoDB Vector Search or AI21 call: {e}", exc_info=True)
            ai_response_content = "Sorry, I encountered an error trying to process your request."

    # --- Store interaction in MongoDB (users_collection) ---
    # ... (same as before, store db_user_message and db_ai_reply_message) ...
    current_timestamp_iso = datetime.now(timezone.utc).isoformat()
    db_user_message = { "role": "user", "content": user_query, "timestamp": current_timestamp_iso }
    db_ai_reply_message = {
        "role": "assistant",
        "content": ai_response_content,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "used_knowledge_base": used_knowledge_base
    }
    try:
        users_collection.update_one(
            {"userid": current_user["userid"]},
            {"$push": {"chat_history": {"$each": [db_user_message, db_ai_reply_message]}}}
        )
    except Exception as e:
        logger.error(f"Failed to update chat history in MongoDB for user {current_user['userid']}: {e}", exc_info=True)

    return {"message": "Chat interaction processed", "response": ai_response_content}

@app.post("/register")
async def register_user_endpoint(registration_data: RegisterUser): # Changed to async
    if not registration_data.userid or not registration_data.emailid or not registration_data.password:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Userid, emailid, and password are required."
        )
    if users_collection.find_one({"emailid": registration_data.emailid}):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="User with this email already exists")
    if users_collection.find_one({"userid": registration_data.userid}):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="User with this userid already exists.")

    hashed_pass = hash_password(registration_data.password)
    current_time_utc = datetime.now(timezone.utc)
    initial_system_message_timestamp = current_time_utc.isoformat()

    user_document = {
        "userid": registration_data.userid,
        "emailid": registration_data.emailid,
        "password": hashed_pass,
        "chat_history": [
            {
                "role": "system",
                "content": "Welcome to Company XYZ support. I can answer questions based on our internal documents.",
                "timestamp": initial_system_message_timestamp
            }
        ],
    }
    users_collection.insert_one(user_document)
    return {"message": "User registered successfully", "userid": registration_data.userid}

@app.post("/login")
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()): # Changed to async
    user = users_collection.find_one({"emailid": form_data.username})
    if not user or not verify_password(form_data.password, user["password"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    token = create_access_token(data={"sub": user["userid"]}, expires_delta=access_token_expires)
    return {"access_token": token, "token_type": "bearer"}

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting FastAPI application with Uvicorn...")
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True) # Use string for reload

    