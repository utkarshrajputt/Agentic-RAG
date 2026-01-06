# Intent-Aware RAG Agent

An intelligent Retrieval-Augmented Generation (RAG) system with intent classification that dynamically decides when to retrieve documents, perform calculations, or answer directly.

## 🌟 Features

- **Intent Classification**: Pre-classifies queries as general, document_search, or calculation
- **Smart Routing**: Uses appropriate tools based on classified intent
- **Document Retrieval**: Semantic search using FAISS and sentence transformers
- **Calculator Tool**: Safe evaluation of mathematical expressions
- **Confidence Checking**: Returns "I don't know" when uncertain
- **Local-First**: FAISS vector store runs entirely on your machine
- **Streamlit UI**: Clean, minimal web interface

## 🏗️ Architecture

```
┌─────────────────┐
│   User Query    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Intent Classifier│
│  (Gemini LLM)   │
└────────┬────────┘
         │
    ┌────┴─────┬──────────┐
    │          │          │
    ▼          ▼          ▼
┌────────┐ ┌────────┐ ┌────────┐
│General │ │Document│ │  Calc  │
│ Answer │ │ Search │ │  Tool  │
└────────┘ └────┬───┘ └────┬───┘
              │          │
              ▼          ▼
           ┌──────────────┐
           │  FAISS Index │
           │  (384-dim)   │
           └──────────────┘
```

## 📋 Tech Stack

| Component | Technology |
|-----------|-----------|
| **LLM** | Google Gemini 2.0 Flash Lite |
| **Embeddings** | sentence-transformers (all-MiniLM-L6-v2) |
| **Vector Store** | FAISS (local) |
| **Framework** | LangChain |
| **UI** | Streamlit |
| **Language** | Python 3.10+ |

## 🚀 Quick Start

### 1. Prerequisites

- Python 3.10 or higher
- Google Gemini API key ([Get one here](https://makersuite.google.com/app/apikey))

### 2. Installation

Clone the repository:
```bash
git clone https://github.com/yourusername/intent-aware-rag.git
cd intent-aware-rag
```

Create and activate virtual environment:
```powershell
# Windows PowerShell
python -m venv venv
.\venv\Scripts\Activate.ps1

# If you get execution policy error:
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

Install dependencies:
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Setup

Set your Gemini API key:

**Option A - Environment Variable (temporary):**
```powershell
$env:GOOGLE_API_KEY="your_gemini_api_key_here"
```

**Option B - .env File (persistent):**
Create a `.env` file in the project root:
```
GOOGLE_API_KEY=your_gemini_api_key_here
```

### 4. Prepare Documents

Create a `data/` folder and add your PDF or text files:
```bash
mkdir data
# Add your .pdf or .txt files to the data/ folder
```

### 5. Build Vector Index

Run the ingestion script (one-time):
```bash
python ingest.py
```

This will:
- Load documents from `data/`
- Split them into chunks (600 chars with 150 char overlap)
- Generate embeddings using sentence-transformers
- Build FAISS index and save to `index/`

### 6. Launch the App

```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501`

## 📁 Project Structure

```
.
├── agent.py          # Intent classification & agent logic
├── app.py            # Streamlit web interface
├── ingest.py         # Document ingestion & indexing
├── retriever.py      # FAISS retrieval logic
├── tools.py          # Document search & calculator tools
├── requirements.txt  # Python dependencies
├── dev.md            # Development guidelines
├── .env              # API keys (create this)
├── .gitignore        # Git ignore rules
├── data/             # Your documents (create & populate)
└── index/            # FAISS index (auto-generated)
    ├── faiss.index
    └── chunks.pkl
```

## 🔧 Configuration

### Chunking Strategy
Edit `ingest.py`:
```python
ingestion = DocumentIngestion(
    chunk_size=600,  # Characters per chunk
    model_name="all-MiniLM-L6-v2"
)
```

### Retrieval Settings
Edit `retriever.py`:
```python
results = retriever.retrieve(query, top_k=5)  # Number of chunks
```

### Model Selection
Edit `agent.py`:
```python
agent = AgenticRAG(
    index_dir="./index",
    model_name="gemini-2.0-flash-lite"  # Change model here
)
```

## 💡 Usage Examples

### Document Questions
```
Q: "What does the document say about machine learning?"
→ Uses document_search tool → Retrieves context → Synthesizes answer
```

### Calculations
```
Q: "What is 145 * 37?"
→ Uses calculator tool → Returns 5365
```

### General Knowledge
```
Q: "What is artificial intelligence?"
→ Answers directly from LLM knowledge
```

## 🧪 Testing Components

Test retriever:
```bash
python retriever.py
```

Test tools:
```bash
python tools.py
```

Test agent:
```bash
python agent.py
```

## 🔍 How It Works

### Intent Classification
The system uses an LLM to classify queries into:
- **general**: Questions answerable with general knowledge
- **document_search**: Questions requiring document retrieval
- **calculation**: Math operations

### Retrieval Process
1. Query embedding generated using `all-MiniLM-L6-v2`
2. FAISS searches for top-k similar chunks using L2 distance
3. Retrieved context passed to LLM for synthesis

### Confidence Checking
The agent checks for uncertainty indicators:
- "I don't know"
- "I'm not sure"
- "No relevant documents found"
- Very short responses (<20 chars)

If uncertain → Returns: "I don't have enough reliable information to answer that."

## 📊 Performance

- **Embedding model**: ~80MB, 384 dimensions
- **Retrieval latency**: <100ms for local FAISS search
- **LLM latency**: Varies by Gemini API response time
- **Index build time**: ~1-2 seconds per MB of documents

## 🛠️ Troubleshooting

### "GOOGLE_API_KEY not found"
- Ensure you've set the environment variable or created `.env` file
- Restart terminal/PowerShell after setting

### "FAISS index not found"
- Run `python ingest.py` first
- Ensure `data/` folder contains documents

### "ModuleNotFoundError"
- Run `pip install -r requirements.txt`
- Ensure virtual environment is activated

### Streamlit not starting
- Check if port 8501 is available
- Try: `streamlit run app.py --server.port 8502`

## 🔮 Future Enhancements

- [ ] Caching layer to reduce API calls
- [ ] Hybrid intent classifier (regex + embeddings + LLM fallback)
- [ ] Local calculation parsing (avoid LLM for math)
- [ ] Cross-encoder re-ranking
- [ ] Semantic chunking
- [ ] Hybrid search (dense + sparse)
- [ ] Query rewriting
- [ ] Multi-document support with metadata filtering
- [ ] Export conversation history
- [ ] Dark mode UI

## 📝 Development Notes

See [dev.md](dev.md) for detailed development guidelines and constraints.

## 📄 License

MIT License - feel free to use this project for learning and development.

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📧 Support

For issues or questions, please open an issue on GitHub.

---

**Note**: This project uses Google Gemini API which may have usage limits and costs. Monitor your API usage through the Google Cloud Console.
