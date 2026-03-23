# 🤖 AI Playground — SD25.7

A hands-on AI experimentation project built while learning 
LLM APIs, Prompt Engineering, and AI-powered FastAPI endpoints.

---

## 🛠️ Tech Stack

- **Python 3.11**
- **Groq API** — fast LLM inference
- **LLaMA 3.3** — Meta's open source AI model
- **FastAPI** — web framework
- **Uvicorn** — ASGI server
- **python-dotenv** — environment variables

---

## 📁 Project Structure
```
ai-playground-sd/
│
├── experiment_1/
│   └── basic_chat.py        # Basic AI chat in terminal
│
├── experiment_2/
│   └── prompt_engineering.py # Prompt engineering techniques
│
├── experiment_3/
│   └── ai_api.py            # AI powered FastAPI endpoints
│
├── groq-llama-notes.txt     # Personal notes on Groq & LLaMA
├── requirements.txt
└── README.md
```

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
```
GROQ_API_KEY=your-groq-api-key-here
```
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

## 📝 What I Learned

- How to call LLM APIs from Python
- Prompt Engineering techniques
- Building AI powered REST APIs
- Combining FastAPI with AI models

---

## 👨‍💻 Author

**Showmik Debnath**
- GitHub: [@ShowmikDebnath](https://github.com/ShowmikDebnath)
- LinkedIn: [showmikdebnath](https://linkedin.com/in/showmikdebnath)
