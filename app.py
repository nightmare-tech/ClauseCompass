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
# from starlette.types import HTTPExceptionHandler # Unused, can be removed
import chromadb

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

# --- ChromaDB Configuration ---
CHROMA_DB_PATH = "./chroma_db_store"
COMPANY_KB_COLLECTION_NAME = "company_internal_kb"
DISTANCE_THRESHOLD_COSINE = 1
OUT_OF_KB_SCOPE_MESSAGE = "I am designed to answer questions based on Company XYZ's internal documents. I do not have information on that topic."
# embedding_function_to_use = None # Not needed if relying on Chroma's default for this app
company_kb_collection = None

try:
    chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    logger.info(f"ChromaDB PersistentClient initialized at {CHROMA_DB_PATH}.")
    # IMPORTANT: Ensure this matches how injest_kb.py creates/accesses the collection
    # If injest_kb.py does NOT pass an embedding_function, Chroma uses its default.
    # So, here too, we should NOT pass embedding_function to use the same default.
    company_kb_collection = chroma_client.get_or_create_collection(
        name=COMPANY_KB_COLLECTION_NAME,
        # embedding_function=embedding_function_to_use, # OMITTED to use Chroma's default
        metadata={"hnsw:space": "cosine"}
    )
    logger.info(f"ChromaDB collection '{COMPANY_KB_COLLECTION_NAME}' loaded/created (using default EF).")
    logger.info(f"Collection count: {company_kb_collection.count()}")
except Exception as e:
    logger.error(f"Failed to initialize ChromaDB client or collection: {e}", exc_info=True)
    # company_kb_collection remains None, will be handled in /chat

# --- Security and JWT Configuration ---
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 # Now used by create_access_token

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
    # final_messages_for_ai21 is built inside the conditional logic now

    historical_ai21_messages = []
    for msg_dict in user_chat_history_from_db:
        role = msg_dict.get("role", "user") # Default to user if role somehow missing
        content = msg_dict.get("content", "") # Default to empty string if content missing
        if not isinstance(content, str): # Ensure content is a string
            logger.warning(f"Historical message content was not a string ({type(content)}), converting. Content: {content}")
            content = str(content)
        historical_ai21_messages.append(ChatMessage(role=role, content=content))

    if company_kb_collection is None:
        logger.error("ChromaDB collection is not available for RAG. Responding with out-of-scope message.")
        ai_response_content = OUT_OF_KB_SCOPE_MESSAGE
    else:
        try:
            logger.debug(f"Querying ChromaDB collection '{COMPANY_KB_COLLECTION_NAME}' for: '{user_query}'")
            retrieved_kb_results = company_kb_collection.query(
                query_texts=[user_query],
                n_results=3,
                include=["documents", "distances"]
            )
            logger.info(f"ChromaDB results: {retrieved_kb_results}")

            retrieved_docs_list = retrieved_kb_results.get('documents', []) # Default to empty list
            retrieved_distances_list = retrieved_kb_results.get('distances', []) # Default to empty list

            # Ensure these are lists of lists as expected, and access the first sub-list
            current_query_docs = retrieved_docs_list[0] if retrieved_docs_list else []
            current_query_distances = retrieved_distances_list[0] if retrieved_distances_list else []

            if current_query_docs and current_query_distances and (current_query_distances[0] < DISTANCE_THRESHOLD_COSINE):
                used_knowledge_base = True
                logger.info(f"Relevant KB context found (distance: {current_query_distances[0]:.4f}). Augmenting prompt.")
                logger.debug(f"Retrieved docs content for RAG: {current_query_docs}")

                context_for_llm_str = "\n--- Provided Company XYZ Documents Context ---\n"
                for i, doc_text in enumerate(current_query_docs):
                    if not isinstance(doc_text, str): # Ensure doc_text is a string
                        logger.warning(f"Retrieved document text for RAG (index {i}) is not a string, converting: {type(doc_text)}")
                        doc_text = str(doc_text)
                    context_for_llm_str += f"Context Document {i+1}:\n{doc_text}\n\n"
                context_for_llm_str += "--- End of Provided Documents Context ---\n"
                logger.debug(f"Constructed context_for_llm_str (first 200 chars): {context_for_llm_str[:200]}")

                system_prompt_rag = (
                    "You are an AI assistant for Company XYZ. Your task is to answer the user's question using ONLY the information found in the 'Provided Company XYZ Documents Context' below. "
                    "Synthesize an answer based on these documents. "
                    "If the documents do not contain enough information to answer the question, explicitly state: "
                    f"'{OUT_OF_KB_SCOPE_MESSAGE}'. "
                    "Do not use any external knowledge. Do not discuss other companies."
                    f"{context_for_llm_str}"
                )                

                # In the RAG positive path (if current_query_docs and ...):

                # ... (system_prompt_rag is constructed with context_for_llm_str) ...
                logger.debug(f"Constructed system_prompt_rag (first 200 chars): {system_prompt_rag[:200]}")

                # --- REVISED PAYLOAD CONSTRUCTION ---
                final_messages_for_ai21 = [
                    ChatMessage(role="system", content=system_prompt_rag),
                    ChatMessage(role="user", content=user_query)
                ]
                logger.debug("Built a clean RAG payload (System Prompt + User Query). Chat history was intentionally excluded for this call.")                                
                # Log the exact payload before sending
                try:
                    payload_log = [msg.model_dump() for msg in final_messages_for_ai21] # Pydantic v2
                except AttributeError:
                    payload_log = [msg.dict() for msg in final_messages_for_ai21] # Pydantic v1
                logger.debug(f"Final messages for AI21 (RAG path) PAYLOAD: {payload_log}")

                ai_api_response = ai21_client.chat.completions.create(
                    model="jamba-mini-1.6-2025-03",
                    messages=final_messages_for_ai21
                )
                ai_response_content = ai_api_response.choices[0].message.content            
            else:
                log_distance = current_query_distances[0] if current_query_distances else 'N/A'
                logger.info(f"KB context not found or below relevance threshold (best distance: {log_distance}). Responding with out-of-scope message.")
                ai_response_content = OUT_OF_KB_SCOPE_MESSAGE
        except Exception as e:
            logger.error(f"Error during ChromaDB query or AI21 call with RAG: {e}", exc_info=True)
            ai_response_content = "Sorry, I encountered an error trying to process your request."

    # --- Store interaction in MongoDB ---
    current_timestamp_iso = datetime.now(timezone.utc).isoformat()
    db_user_message = {
        "role": "user",
        "content": user_query,
        "timestamp": current_timestamp_iso
    }
    db_ai_reply_message = {
        "role": "assistant",
        "content": ai_response_content, # This will be the error message if an exception occurred above
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
        "created_at": current_time_utc,
        "updated_at": current_time_utc
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