from langchain_groq import ChatGroq
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from dotenv import load_dotenv
import os

load_dotenv()

# ─────────────────────────────────────────
# SETUP — Load, Split, Embed, Store
# ─────────────────────────────────────────
print("Setting up RAG system...")

# Load document
loader = TextLoader("experiment_8/knowledge_base.txt")
documents = loader.load()

# Split into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=20
)
chunks = text_splitter.split_documents(documents)

# Create embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Store in ChromaDB
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="experiment_8/chroma_db"
)

# Setup AI model
chat = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model="llama-3.3-70b-versatile",
    temperature=0.3
)

print("RAG system ready!\n")


# ─────────────────────────────────────────
# CONVERSATION HISTORY (memory!)
# ─────────────────────────────────────────
conversation_history = []


# ─────────────────────────────────────────
# RAG CHATBOT FUNCTION
# ─────────────────────────────────────────
def chat_with_rag(user_message: str):

    # Step 1 — Find relevant chunks
    relevant_chunks = vectorstore.similarity_search(user_message, k=5)
    context = "\n\n".join([chunk.page_content for chunk in relevant_chunks])

    # FOR DEBUGING PURPOSE-
    # print("\n--- All Chunks ---")
    # for i, chunk in enumerate(chunks):
    #     print(f"Chunk {i+1}: {chunk.page_content[:100]}")
    # print("---\n")

    # Step 2 — Build conversation history text
    history_text = ""
    for msg in conversation_history:  # keep ALL history
        if isinstance(msg, HumanMessage):
            history_text += f"User: {msg.content}\n"
        elif isinstance(msg, AIMessage):
            history_text += f"Assistant: {msg.content}\n"

    # Step 3 — Build prompt with context + history
    template = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful assistant who answers questions 
        about Showmik Debnath based on the provided context.
        
        If the answer is not in the context, say 
        "I don't have that information."
        
        Keep answers concise and friendly.

        Previous conversation:
        {history}
        
        Context information:
        {context}
        """),
        ("human", "{question}")
    ])

    # Step 4 — Create chain and get answer
    chain = template | chat | StrOutputParser()
    answer = chain.invoke({
        "context": context,
        "history": history_text,
        "question": user_message
    })

    # Step 5 — Save to conversation history
    conversation_history.append(HumanMessage(content=user_message))
    conversation_history.append(AIMessage(content=answer))

    return answer


# ─────────────────────────────────────────
# CHAT LOOP
# ─────────────────────────────────────────
print("=" * 50)
print("RAG Chatbot — Ask about Showmik Debnath!")
print("Type 'exit' to quit")
print("=" * 50)

while True:
    user_input = input("\nYou: ").strip()

    if not user_input:
        continue

    if user_input.lower() == "exit":
        print("Goodbye! 👋")
        break

    answer = chat_with_rag(user_input)
    print(f"\n🤖 Bot: {answer}")






