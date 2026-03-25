from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import os

load_dotenv()

# ─────────────────────────────────────────
# Setup LangChain with Groq
# ─────────────────────────────────────────
chat = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model="llama-3.3-70b-versatile",
    temperature=0.7
)


# ─────────────────────────────────────────
# PART 1 — Basic Message
# ─────────────────────────────────────────
print("=" * 50)
print("PART 1 — Basic Message")
print("=" * 50)

messages = [
    SystemMessage(content="You are a helpful assistant. Keep answers short."),
    HumanMessage(content="What is LangChain in one sentence?")
]

response = chat.invoke(messages)
print(response.content)


# ─────────────────────────────────────────
# PART 2 — Prompt Template
# ─────────────────────────────────────────
print("\n" + "=" * 50)
print("PART 2 — Prompt Template")
print("=" * 50)

# Create a reusable template
template = ChatPromptTemplate.from_messages([
    ("system", "You are an expert in {subject}. Keep answers simple."),
    ("human", "Explain {topic} in 2 sentences.")
])

# Use template with different values
prompt = template.invoke({
    "subject": "Python programming",
    "topic": "decorators"
})

response = chat.invoke(prompt)
print(response.content)


# ─────────────────────────────────────────
# PART 3 — Chaining (The | operator)
# ─────────────────────────────────────────
print("\n" + "=" * 50)
print("PART 3 — Chaining")
print("=" * 50)

# Chain template directly to chat model
chain = template | chat

# Run the chain
response = chain.invoke({
    "subject": "English grammar",
    "topic": "past tense"
})
print(response.content)



