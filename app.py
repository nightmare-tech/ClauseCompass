from fastapi import FastAPI, Request
from pydantic import BaseModel
from ai21 import AI21Client
from ai21.models.chat import ChatMessage
from dotenv import load_dotenv
import os

load_dotenv() 

api_key = os.getenv("AI21_API_KEY")

if not api_key:
    raise ValueError("API key not found in environment variables.")

client = AI21Client(api_key=api_key)

app = FastAPI()

class ChatRequest(BaseModel):
    message: str

chat_history = [
    ChatMessage(role="system", content="You are a helpful assistant.")
]

@app.post("/chat")
def chat_endpoint(chat_req: ChatRequest):
    chat_history.append(ChatMessage(role="user", content=chat_req.message))

    response = client.chat.completions.create(
        model="jamba-mini-1.6-2025-03",
        messages=chat_history
    )

    reply = response.choices[0].message.content
    chat_history.append(ChatMessage(role="assistant", content=reply))

    return {"response": reply}

