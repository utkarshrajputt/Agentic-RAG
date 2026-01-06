# Setup Guide

Complete step-by-step setup instructions for the Intent-Aware RAG Agent.

## Prerequisites

- **Python**: Version 3.10 or higher
- **Git**: For cloning the repository
- **Google Gemini API Key**: [Get one here](https://makersuite.google.com/app/apikey)
- **Operating System**: Windows (PowerShell commands provided)

## Step-by-Step Setup

### Step 1: Clone Repository

```bash
git clone https://github.com/yourusername/intent-aware-rag.git
cd intent-aware-rag
```

### Step 2: Create Virtual Environment

```powershell
# Create virtual environment
python -m venv venv

# Activate it (PowerShell)
.\venv\Scripts\Activate.ps1

# If you get execution policy error, run:
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
# Then try activating again
```

You should see `(venv)` in your terminal prompt.

### Step 3: Install Dependencies

```bash
# Upgrade pip
pip install --upgrade pip

# Install all requirements
pip install -r requirements.txt
```

This installs:
- `langchain` and `langchain-google-genai`
- `sentence-transformers`
- `faiss-cpu`
- `streamlit`
- `PyPDF2`
- `python-dotenv`

### Step 4: Configure API Key

**Method 1: Environment Variable (Session-based)**

```powershell
$env:GOOGLE_API_KEY="your_actual_gemini_api_key"
```

Note: This only works for the current PowerShell session.

**Method 2: .env File (Persistent, Recommended)**

Create a file named `.env` in the project root:

```bash
# .env file content
GOOGLE_API_KEY=your_actual_gemini_api_key
```

The app will automatically load this on startup.

### Step 5: Prepare Documents

```bash
# Create data directory
mkdir data

# Add your documents
# Copy .pdf or .txt files into the data/ folder
```

**Supported formats:**
- PDF (`.pdf`)
- Plain text (`.txt`)

**Example documents:**
- Research papers
- Technical documentation
- Books/articles
- Company knowledge base

### Step 6: Build the Index

```bash
python ingest.py
```

**What this does:**
- Loads all documents from `data/`
- Splits them into 600-character chunks with 150-char overlap
- Generates 384-dimensional embeddings
- Builds FAISS index
- Saves to `index/faiss.index` and `index/chunks.pkl`

**Expected output:**
```
Loaded 5 documents
Created 113 chunks
Generating embeddings...
100%|████████████| 113/113
Generated embeddings with shape: (113, 384)
Built FAISS index with 113 vectors
Saved index to ./index
Ingestion complete!
```

### Step 7: Launch the App

```bash
streamlit run app.py
```

**Expected output:**
```
You can now view your Streamlit app in your browser.

  Local URL: http://localhost:8501
  Network URL: http://192.168.x.x:8501
```

Open `http://localhost:8501` in your browser.

## Verification

### Test the Components

**Test retriever:**
```bash
python retriever.py
```

**Test tools:**
```bash
python tools.py
```

**Test agent:**
```bash
python agent.py
```

### Test Queries

Try these in the Streamlit UI:

1. **Document question**: "What does the document say about [your topic]?"
2. **Calculation**: "What is 125 * 48?"
3. **General knowledge**: "What is machine learning?"

## Common Issues

### Issue: "GOOGLE_API_KEY not found"

**Solution:**
- Check if `.env` file exists and contains the key
- Verify no extra spaces or quotes in `.env`
- Restart the app after creating `.env`

### Issue: "FAISS index not found"

**Solution:**
- Run `python ingest.py` first
- Check that `data/` folder exists and contains documents
- Verify `index/` folder was created with `faiss.index` and `chunks.pkl`

### Issue: Import errors

**Solution:**
```bash
# Ensure venv is activated
.\venv\Scripts\Activate.ps1

# Reinstall requirements
pip install -r requirements.txt --force-reinstall
```

### Issue: Streamlit won't start

**Solution:**
```bash
# Check if port is already in use
# Try a different port
streamlit run app.py --server.port 8502
```

### Issue: Execution policy error (PowerShell)

**Solution:**
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

## Environment Structure

After setup, your directory should look like:

```
intent-aware-rag/
├── venv/                  # Virtual environment (not in git)
├── data/                  # Your documents (not in git)
│   ├── document1.pdf
│   └── document2.txt
├── index/                 # Generated index (not in git)
│   ├── faiss.index
│   └── chunks.pkl
├── .env                   # API keys (not in git)
├── agent.py
├── app.py
├── ingest.py
├── retriever.py
├── tools.py
├── requirements.txt
├── .gitignore
└── README.md
```

## Next Steps

1. **Add more documents**: Copy files to `data/` and run `python ingest.py` again
2. **Customize chunking**: Edit chunk size in `ingest.py`
3. **Tune retrieval**: Adjust `top_k` in `retriever.py`
4. **Change model**: Switch Gemini model in `agent.py`

## Additional Configuration

### Change Chunk Size

Edit `ingest.py`:
```python
ingestion = DocumentIngestion(
    chunk_size=800,  # Increase for more context
    model_name="all-MiniLM-L6-v2"
)
```

### Change Top-K Results

Edit retrieval call in `retriever.py` or `tools.py`:
```python
results = self.retriever.retrieve(query, top_k=10)  # More results
```

### Change LLM Model

Edit `agent.py`:
```python
def __init__(self, index_dir: str = "./index", 
             model_name: str = "gemini-2.5-flash"):  # Upgrade model
```

### Change Streamlit Port

```bash
streamlit run app.py --server.port 8080
```

## Getting Help

- Check the [README.md](README.md) for usage examples
- Review [dev.md](dev.md) for development guidelines
- Open an issue on GitHub for bugs

---

**Ready to go!** Start asking questions in the Streamlit UI.
