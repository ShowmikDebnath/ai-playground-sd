from langchain_groq import ChatGroq
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os

load_dotenv()

# ─────────────────────────────────────────
# STEP 1 — Load the document
# ─────────────────────────────────────────
print("Step 1: Loading document...")

loader = TextLoader("experiment_7/sample_document.txt")
documents = loader.load()

print(f"Loaded {len(documents)} document(s)")


# ─────────────────────────────────────────
# STEP 2 — Split into chunks
# ─────────────────────────────────────────
print("\nStep 2: Splitting into chunks...")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,      # each chunk max 200 characters
    chunk_overlap=20     # overlap to keep context
)

chunks = text_splitter.split_documents(documents)

print(f"Created {len(chunks)} chunks")
print(f"\nFirst chunk preview:\n{chunks[0].page_content}")


# ─────────────────────────────────────────
# STEP 3 — Create embeddings
# ─────────────────────────────────────────
print("\nStep 3: Creating embeddings...")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

print("Embeddings model loaded!")


# ─────────────────────────────────────────
# STEP 4 — Store in Vector Database
# ─────────────────────────────────────────
print("\nStep 4: Storing in ChromaDB...")

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="experiment_7/chroma_db"
)

print("Stored successfully!")


# ─────────────────────────────────────────
# STEP 5 — Setup the AI model
# ─────────────────────────────────────────
chat = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model="llama-3.3-70b-versatile",
    temperature=0.3
)


# ─────────────────────────────────────────
# STEP 6 — RAG Function (the magic happens here!)
# ─────────────────────────────────────────
def ask_question(question: str):
    # 6a — Search for similar chunks
    relevant_chunks = vectorstore.similarity_search(question, k=2)

    # 6b — Combine chunks into context
    context = "\n\n".join([chunk.page_content for chunk in relevant_chunks])

    # 6c — Build prompt with context
    template = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful assistant. Answer the question 
        using ONLY the context provided below. If the answer is not 
        in the context, say "I don't have that information."
        
        Context:
        {context}"""),
        ("human", "{question}")
    ])

    # 6d — Create chain and get answer
    chain = template | chat | StrOutputParser()
    answer = chain.invoke({
        "context": context,
        "question": question
    })

    return answer, relevant_chunks


# ─────────────────────────────────────────
# STEP 7 — Test it!
# ─────────────────────────────────────────
print("\n" + "=" * 50)
print("Testing RAG System")
print("=" * 50)

questions = [
    "Who built Grammar Guru?",
    "What technology is used for the backend?",
    "What is the maximum text length allowed?",
    "Does Grammar Guru have a mobile app?",
]

for q in questions:
    print(f"\n❓ Question: {q}")
    answer, chunks_used = ask_question(q)
    print(f"💬 Answer: {answer}")














