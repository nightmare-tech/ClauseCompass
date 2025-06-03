from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from datetime import datetime, timedelta
from jose import JWTError, jwt
from passlib.context import CryptContext
from pymongo.message import update
from pymongo.mongo_client import MongoClient as MongoClientDB, message
from pymongo.server_api import ServerApi
from urllib.parse import quote_plus
from pydantic import BaseModel
from ai21 import AI21Client
from ai21.models.chat import ChatMessage
from dotenv import load_dotenv
import os
from starlette.types import HTTPExceptionHandler

load_dotenv() 

app = FastAPI()
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")


api_key = os.getenv("AI21_API_KEY")

username = os.getenv("USERN")
password = os.getenv("PASSW")

if not api_key:
    raise ValueError("API key not found in environment variables.")
elif not username:
    raise ValueError("username not in env")
elif not password:
    raise ValueError("password not in env")


client = AI21Client(api_key=api_key)


uri = f"mongodb+srv://{username}:{password}@chattydb.dfuykzc.mongodb.net/?retryWrites=true&w=majority&appName=chattydb"
mongo_client = MongoClientDB(uri, server_api=ServerApi('1'))
db = mongo_client["chattydb"]
users_col = db["users"]

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

SECRET_KEY = os.getenv("JWT_SECRET")

if not SECRET_KEY:
    raise ValueError("secret key not in env")

ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60

class ChatRequest(BaseModel):
    message: str

class RegisterUser(BaseModel):
    userid: str
    emailid: str
    password: str

def hash_password(password: str):
    return pwd_context.hash(password)

def verify_password(plain: str, hashed: str):
    return pwd_context.verify(plain, hashed)

def create_token(data: dict, expires_delta: timedelta = None):
    to_encode = data.copy()
    expire = datetime.now() + (expires_delta or timedelta(minutes=10))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def decode_token(token: str): 
    try:
        return jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    except JWTError:
        return None

def get_current_user(token: str = Depends(oauth2_scheme)):
    payload = decode_token(token)
    if not payload:
        raise HTTPException(status_code=401, detail="Invalid Token")
    userid = payload.get("sub")
    user = users_col.find_one({"userid": userid})

    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    return user
    

@app.post("/chat")
def chat(chat_req: ChatRequest, user_document=Depends(get_current_user)): # user_document is the dict from DB

    # chat_history_from_db is a list of dictionaries
    chat_history_from_db = user_document.get("chat_history", [])

    if not isinstance(chat_history_from_db, list): # Basic check
        # This case should ideally be handled by ensuring chat_history is always initialized as a list
        chat_history_from_db = []

    # 1. Prepare messages for AI21 client (list of ChatMessage objects)
    messages_for_ai = []
    for msg_dict in chat_history_from_db:
        content_for_ai = msg_dict.get("content", "")
        role_for_ai = msg_dict.get("role")
        if not role_for_ai: continue # Skip if role is missing

        # Append timestamp to content for historical user/assistant messages, as per system prompt
        # System prompt itself is passed as is.
        if role_for_ai != "system" and "timestamp" in msg_dict:
            content_for_ai = f"{content_for_ai} (timestamp: {msg_dict['timestamp']})"
        
        messages_for_ai.append(ChatMessage(role=role_for_ai, content=content_for_ai))

    # 2. Current user's message (as ChatMessage object for AI)
    # The system prompt implies AI should see timestamps in content
    current_time_iso = datetime.now().isoformat()
    current_user_message_content_for_ai = f"{chat_req.message} (timestamp: {current_time_iso})"
    current_user_ai_message_object = ChatMessage(role="user", content=current_user_message_content_for_ai)
    messages_for_ai.append(current_user_ai_message_object)

    # 3. Call AI21 API
    try:
        ai_response_obj = client.chat.completions.create( # Renamed to avoid conflict
            model="jamba-mini-1.6-2025-03",
            messages=messages_for_ai # This is now list[ChatMessage]
        )
    except Exception as e:
        # Log the error for server-side debugging
        print(f"AI21 API Error: {e}") # Consider using proper logging
        raise HTTPException(status_code=503, detail="Error communicating with AI service.")

    ai_reply_content = ai_response_obj.choices[0].message.content

    # 4. Prepare messages for DB storage (as dictionaries)
    # User's message for DB (store original content + separate timestamp)
    db_user_message_to_store = {
        "role": "user",
        "content": chat_req.message, # Original message without appended timestamp
        "timestamp": current_time_iso # Timestamp of this message
    }

    # AI's reply for DB
    db_ai_reply_to_store = {
        "role": "assistant",
        "content": ai_reply_content,
        "timestamp": datetime.now().isoformat() # Timestamp of AI's reply
    }

    # 5. Update DB atomically using the correct field name "chat_history"
    users_col.update_one(
        {"userid": user_document["userid"]},
        {"$push": {"chat_history": {"$each": [db_user_message_to_store, db_ai_reply_to_store]}}}
    )

    return {"message": "Chat stored", "response": ai_reply_content}

@app.post("/register")
def register_endpoint(reg_data: RegisterUser):
    if users_col.find_one({"emailid": reg_data.emailid}):
        raise HTTPException(status_code=400, detail="Account with this email already exists")
    elif users_col.find_one({"userid": reg_data.userid}):
        raise HTTPException(status_code=400, detail="Account with this userid already exists")
    
    initial_system_message_timestamp = datetime.now().isoformat()
    user_doc = {
        "userid": reg_data.userid,
        "emailid": reg_data.emailid,
        "password": hash_password(reg_data.password),
        "chat_history": [
            {
                "role": "system",
                "content": "You are a helpful assistant. Keep in mind that message history has timestamps. Each message content will include '(timestamp: YYYY-MM-DDTHH:MM:SS.ffffff)' for your time awareness.",
                "timestamp": initial_system_message_timestamp
            }
        ],
    }

    users_col.insert_one(user_doc)
    return {"message": "User Registered successfully", "userid": reg_data.userid}


@app.post("/login")
def login(form_data: OAuth2PasswordRequestForm = Depends()):
    user = users_col.find_one({"emailid": form_data.username})
    if not user or not verify_password(form_data.password, user["password"]):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    token = create_token(data={"sub": user["userid"]})
    return {"access_token": token, "token_type": "bearer"}


