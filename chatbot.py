from ai21 import AI21Client
from ai21.models.chat import ChatMessage
import json
with open("config.json") as f:
    api_key = json.load(f)["api_key"]


client = AI21Client(api_key)

# Start with a system prompt and empty message history
messages = [
    ChatMessage(role="system", content="You are a helpful assistant.")
]

print("AI21 Chatbot is ready! Type 'exit' to quit.\n")

while True:
    user_input = input("You: ").strip()

    if user_input.lower() == "exit":
        print("Exiting chat.")
        print(messages)
        break

    # Add user's message to history
    messages.append(ChatMessage(role="user", content=user_input))

    # Get model response
    response = client.chat.completions.create(
        model="jamba-mini-1.6-2025-03",
        messages=messages
    )

    # Extract assistant's message and print it
    ai_reply = response.choices[0].message.content.strip()
    print("AI21:", ai_reply)

    # Add assistant's reply to message history
    messages.append(ChatMessage(role="assistant", content=ai_reply))
