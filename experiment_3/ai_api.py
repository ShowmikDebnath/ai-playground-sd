from fastapi import FastAPI
from pydantic import BaseModel
from groq import Groq
from dotenv import load_dotenv
import os

# Load API key
load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# FastAPI app
app = FastAPI(
    title="Bazar AI API",
    description="AI powered API using Groq + LLaMA",
    version="1.0.0"
)

# ─────────────────────────────────────────
# Schemas
# ─────────────────────────────────────────
class ChatRequest(BaseModel):
    message: str

class SummarizeRequest(BaseModel):
    text: str

class SentimentRequest(BaseModel):
    text: str


# ─────────────────────────────────────────
# Endpoint 1 — Chat
# ─────────────────────────────────────────
@app.post("/chat")
def chat(request: ChatRequest):
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant. Keep answers clear and concise."
            },
            {
                "role": "user",
                "content": request.message
            }
        ]
    )
    return {
        "message": request.message,
        "response": response.choices[0].message.content
    }


# ─────────────────────────────────────────
# Endpoint 2 — Summarize
# ─────────────────────────────────────────
@app.post("/summarize")
def summarize(request: SummarizeRequest):
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": "You are a summarization expert. Summarize the given text in 2-3 sentences only. Be concise."
            },
            {
                "role": "user",
                "content": f"Summarize this: {request.text}"
            }
        ]
    )
    return {
        "original": request.text,
        "summary": response.choices[0].message.content
    }


# ─────────────────────────────────────────
# Endpoint 3 — Sentiment Analysis
# ─────────────────────────────────────────
@app.post("/sentiment")
def sentiment(request: SentimentRequest):
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": "You are a sentiment analysis expert. Reply with only one word: 'positive', 'negative', or 'neutral'. Nothing else."
            },
            {
                "role": "user",
                "content": f"Analyze sentiment: {request.text}"
            }
        ]
    )
    return {
        "text": request.text,
        "sentiment": response.choices[0].message.content
    }


# ─────────────────────────────────────────
# Root
# ─────────────────────────────────────────
@app.get("/")
def root():
    return {"message": "Welcome to SD AI API"}