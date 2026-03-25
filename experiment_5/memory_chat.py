from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from dotenv import load_dotenv
import os

load_dotenv()

# ─────────────────────────────────────────
# Setup
# ─────────────────────────────────────────
chat = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model="llama-3.3-70b-versatile",
    temperature=0.7
)

# ─────────────────────────────────────────
# Conversation History (our manual memory)
# ─────────────────────────────────────────
conversation_history = [
    SystemMessage(content="""You are a helpful English grammar teacher.
You remember everything the student tells you.
Keep your answers short and friendly.""")
]


# ─────────────────────────────────────────
# Chat function with memory
# ─────────────────────────────────────────
def chat_with_memory(user_message: str):
    # Step 1 — Add user message to history
    conversation_history.append(
        HumanMessage(content=user_message)
    )

    # Step 2 — Send FULL history to AI
    response = chat.invoke(conversation_history)

    # Step 3 — Add AI reply to history
    conversation_history.append(
        AIMessage(content=response.content)
    )

    return response.content


# ─────────────────────────────────────────
# Test the memory
# ─────────────────────────────────────────
print("=" * 50)
print("Testing Conversation Memory")
print("=" * 50)

# Message 1
print("\nYou: My name is Showmik and I am learning English grammar.")
reply = chat_with_memory("My name is Showmik and I am learning English grammar.")
print(f"AI: {reply}")

# Message 2
print("\nYou: What is the difference between past simple and past continuous?")
reply = chat_with_memory("What is the difference between past simple and past continuous?")
print(f"AI: {reply}")

# Message 3 — testing memory!
print("\nYou: Can you give me an example using my name?")
reply = chat_with_memory("Can you give me an example using my name?")
print(f"AI: {reply}")

# Message 4 — testing memory again!
print("\nYou: What was the first thing I told you?")
reply = chat_with_memory("What was the first thing I told you?")
print(f"AI: {reply}")


