from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import SystemMessage
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

# Output parser — converts AI response to clean string
parser = StrOutputParser()


# ─────────────────────────────────────────
# CHAIN 1 — Simple Chain
# ─────────────────────────────────────────
print("=" * 50)
print("CHAIN 1 — Simple Chain")
print("=" * 50)

# Template → Chat → Parser
simple_template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Keep answers to 2 sentences."),
    ("human", "{question}")
])

simple_chain = simple_template | chat | parser

result = simple_chain.invoke({
    "question": "What is Python?"
})
print(result)


# ─────────────────────────────────────────
# CHAIN 2 — Sequential Chain
# (output of chain 1 becomes input of chain 2)
# ─────────────────────────────────────────
print("\n" + "=" * 50)
print("CHAIN 2 — Sequential Chain")
print("=" * 50)

# First chain — explain a topic
explain_template = ChatPromptTemplate.from_messages([
    ("system", "You are a teacher. Explain in 2 sentences only."),
    ("human", "Explain {topic}")
])

# Second chain — make it simpler
simplify_template = ChatPromptTemplate.from_messages([
    ("system", "You are an expert at simplifying text for 10 year olds."),
    ("human", "Make this even simpler: {explanation}")
])

# Build both chains
explain_chain = explain_template | chat | parser
simplify_chain = simplify_template | chat | parser

# Run sequentially — manually passing output
explanation = explain_chain.invoke({"topic": "machine learning"})
print(f"Original explanation:\n{explanation}")

simplified = simplify_chain.invoke({"explanation": explanation})
print(f"\nSimplified version:\n{simplified}")


# ─────────────────────────────────────────
# CHAIN 3 — Grammar Chain
# (similar to Grammar Guru but using chains!)
# ─────────────────────────────────────────
print("\n" + "=" * 50)
print("CHAIN 3 — Grammar Chain")
print("=" * 50)

grammar_template = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are an English grammar expert.
When given a text:
1. Correct all grammar mistakes
2. List each mistake briefly
Keep response short and clear."""
    ),
    ("human", "Check this text: {text}")
])

grammar_chain = grammar_template | chat | parser

result = grammar_chain.invoke({
    "text": "I are going to school yesterday and buyed a book."
})
print(result)