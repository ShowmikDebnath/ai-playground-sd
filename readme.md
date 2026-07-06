# 🤖 AI Playground — SD25.7

A hands-on AI experimentation project built while learning 
LLM APIs, Prompt Engineering, LangChain, and RAG systems.

---

## 🛠️ Tech Stack

- **Python 3.11**
- **Groq API** — fast LLM inference
- **LLaMA 3.3** — Meta's open source AI model
- **FastAPI** — web framework
- **LangChain** — AI orchestration framework
- **ChromaDB** — vector database
- **HuggingFace** — embedding models
- **Uvicorn** — ASGI server
- **python-dotenv** — environment variables

---

## 📁 Project Structure

ai-playground-sd/
│
├── experiment_1/
│   └── basic_chat.py           # Basic AI chat in terminal
│
├── experiment_2/
│   └── prompt_engineering.py   # Prompt engineering techniques
│
├── experiment_3/
│   └── ai_api.py               # AI powered FastAPI endpoints
│
├── experiment_4/
│   └── langchain_basics.py     # LangChain fundamentals
│
├── experiment_5/
│   └── memory_chat.py          # Conversation memory
│
├── experiment_6/
│   └── chains.py               # LangChain chaining
│
├── experiment_7/
│   ├── rag_basics.py           # RAG fundamentals
│   └── sample_document.txt     # Sample document for RAG
│
├── experiment_8/
│   ├── rag_chatbot.py          # RAG chatbot with memory
│   └── knowledge_base.txt      # Knowledge base document
│
├── groq-llama-notes.txt        # Personal notes on Groq & LLaMA
├── requirements.txt
└── README.md

---

## 🚀 Getting Started

**1. Clone the repo**
```bash
git clone https://github.com/ShowmikDebnath/ai-playground-sd.git
cd ai-playground-sd
```

**2. Create virtual environment**
```bash
python3.11 -m venv myenv
source myenv/bin/activate
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Create `.env` file**

GROQ_API_KEY=your-groq-api-key-here

Get your free API key at: console.groq.com

---

## 🧪 Experiments

### Experiment 1 — Basic AI Chat
Simple terminal chat using Groq + LLaMA 3.3
```bash
python experiment_1/basic_chat.py
```

### Experiment 2 — Prompt Engineering
Three prompt engineering techniques:
- Zero-shot prompting
- Few-shot prompting
- System prompt
```bash
python experiment_2/prompt_engineering.py
```

### Experiment 3 — AI Powered FastAPI
REST API endpoints powered by AI:

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Welcome message |
| POST | `/chat` | AI chat endpoint |
| POST | `/summarize` | Text summarizer |
| POST | `/sentiment` | Sentiment analysis |

```bash
uvicorn experiment_3.ai_api:app --reload
```
Then open: http://127.0.0.1:8000/docs

---

### Experiment 4 — LangChain Basics
LangChain fundamentals including:
- Chat models
- Message types (SystemMessage, HumanMessage, AIMessage)
- Prompt templates
- Chaining with `|` operator
```bash
python experiment_4/langchain_basics.py
```

---

### Experiment 5 — Conversation Memory
AI that remembers previous messages:
- Manual conversation history
- Growing message list
- Context-aware responses
```bash
python experiment_5/memory_chat.py
```

---

### Experiment 6 — Chains
Connecting multiple AI steps together:
- Simple chains
- Sequential chains
- Output parsers
```bash
python experiment_6/chains.py
```

---

### Experiment 7 — RAG Basics
Retrieval Augmented Generation fundamentals:
- Document loading
- Text splitting into chunks
- HuggingFace embeddings
- ChromaDB vector storage
- Semantic search
```bash
python experiment_7/rag_basics.py
```

**How RAG works:**

Document → Split → Embed → Store in ChromaDB
Question → Embed → Search → Get relevant chunks
chunks + question → AI → Answer ✅

---

### Experiment 8 — RAG Chatbot
Full RAG chatbot combining memory + RAG:
- Answers questions about custom documents
- Remembers conversation history
- Says "I don't know" when answer not in document
- Combines RAG + Memory + Chains + Prompt Engineering
```bash
python experiment_8/rag_chatbot.py
```

---

## 📝 What I Learned

- Calling LLM APIs from Python using Groq
- Prompt Engineering techniques (zero-shot, few-shot, system prompts)
- Building AI powered REST APIs with FastAPI
- LangChain fundamentals and chaining
- Conversation memory management
- RAG system architecture
- Text embeddings and semantic search
- Vector databases with ChromaDB
- Combining RAG + Memory for intelligent chatbots

---

## 👨‍💻 Author

**Showmik Debnath**
- GitHub: [@ShowmikDebnath](https://github.com/ShowmikDebnath)
- LinkedIn: [showmikdebnath](https://linkedin.com/in/showmikdebnath)

