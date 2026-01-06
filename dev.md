
## Agentic RAG Assistant (LangChain-based)

---

## 1. Project Overview

### Project Name

**Intent-Aware Agentic RAG Assistant**

### Goal

Build a Retrieval-Augmented Generation (RAG) system enhanced with an **agent layer** that dynamically decides:

* when to answer directly,
* when to retrieve documents,
* when to use tools,
* and when to refuse answering due to low confidence.

This project focuses on **correct system design**, not UI polish or production infra.

---

## 2. Core Concepts Demonstrated

* RAG (Retrieval-Augmented Generation)
* LangChain usage
* Agentic decision-making
* Tool calling
* Intent awareness
* Hallucination prevention using confidence & fallback
* Clean, explainable architecture

---

## 3. Tech Stack (LOCKED)

Do NOT change unless necessary.

### Backend

* Python 3.10+
* LangChain

### Embeddings

* `sentence-transformers`
* Model: `all-MiniLM-L6-v2`

### Vector Store

* FAISS (local)

### LLM (pick ONE)

* OpenAI GPT-3.5 (simplest)
  OR
* Ollama with LLaMA / Mistral (fully local)

### Frontend

* Streamlit (minimal UI)

---

## 4. Project Structure

```text
agentic-rag-assistant/
│
├── ingest.py        # Document ingestion & embeddings
├── retriever.py     # Retrieval logic
├── tools.py         # Agent tools
├── agent.py         # Agent + decision logic
├── app.py           # Streamlit UI
├── requirements.txt
├── dev.md           # This file
└── README.md
```

---

## 5. High-Level Architecture

```
User Query
   ↓
Intent Classification (LLM)
   ↓
Agent Reasoning
   ├─ Direct Answer
   ├─ Document Retrieval Tool (RAG)
   └─ Calculator Tool
   ↓
Confidence Check
   ↓
Final Answer OR Fallback
```

---

## 6. User Flow (UI Perspective)

1. User opens Streamlit app
2. User types a question
3. Clicks **Ask**
4. Sees response

User does NOT know:

* what an agent is
* what RAG is
* what tools are

This is correct behavior.

---

## 7. Backend Flow (IMPORTANT)

1. Receive user query
2. Classify intent
3. Agent decides next step
4. Agent may:

   * answer directly
   * retrieve documents
   * call calculator
5. Generate answer
6. Run confidence check
7. Return final response or fallback

---

## 8. File-by-File Implementation Details

---

### ingest.py – Document Ingestion

#### Purpose

Offline step to prepare documents for retrieval.

#### Responsibilities

* Load documents (PDF or text)
* Chunk documents
* Generate embeddings
* Store embeddings in FAISS

#### Key Points

* Runs once, not per query
* Keep chunk size simple (e.g. 500–700 tokens)
* Save FAISS index locally

---

### retriever.py – Retrieval Layer

#### Purpose

Encapsulates semantic search logic.

#### Responsibilities

* Load FAISS index
* Accept query
* Retrieve top-k relevant chunks
* Return plain text

#### Key Rule

Retriever does NOT decide *when* to run.
Agent decides.

---

### tools.py – Agent Tools

You must implement **exactly two tools**.

#### Tool 1: Document Search Tool

* Wraps retriever logic
* Returns relevant context text

#### Tool 2: Calculator Tool

* Performs exact math
* Demonstrates deterministic tool delegation

#### Why Tools Exist

LLMs are probabilistic and unreliable for:

* math
* exact lookups
* deterministic logic

Agents should delegate these tasks.

---

### agent.py – Agent Logic (MOST IMPORTANT)

#### Responsibilities

* Initialize LLM
* Register tools
* Define system prompt
* Create LangChain agent
* Execute agent reasoning loop

#### System Prompt MUST Include

* When to retrieve documents
* When to use tools
* When to say “I don’t know”
* Avoid hallucination

Example ideas (do NOT copy verbatim):

* “Use document search only if external knowledge is needed”
* “Use calculator for any math”
* “If unsure, say you don’t know”

---

### app.py – Streamlit UI

#### Responsibilities

* Simple UI wrapper
* Pass user input to agent
* Display response

#### UI Rules

* One input box
* One button
* One output area
* No styling, no auth, no extra logic

---

## 9. Intent Classification (IMPORTANT UPGRADE)

Before agent execution, classify intent using LLM.

Possible intents:

* `general`
* `document_search`
* `calculation`

This can be prompt-based (no extra model needed).

Intent helps agent:

* avoid unnecessary retrieval
* reduce hallucinations
* choose tools correctly

---

## 10. Confidence & Fallback Logic

After generating an answer:

Trigger fallback if:

* Retrieved context is empty
* Agent expresses uncertainty
* Answer is unsupported

Fallback response:

> “I don’t have enough reliable information to answer that.”

This is a **feature**, not a weakness.

---

## 11. Example End-to-End Flows

### Example 1: Simple Question

**User:** “What is Artificial Intelligence?”

* Intent: general
* Agent: direct answer
* No tools used

---

### Example 2: Document Question

**User:** “What does the document say about bias and variance?”

* Intent: document_search
* Agent calls Document Search Tool
* Answer grounded in retrieved text

---

### Example 3: Tool Question

**User:** “What is 145 × 37?”

* Intent: calculation
* Agent calls Calculator Tool
* LLM explains result

---

### Example 4: Unknown Question

**User:** “What is the internal salary of Company X?”

* No context
* No confidence
* Fallback triggered

---

## 12. Non-Goals (DO NOT IMPLEMENT)

* Multi-agent systems
* AutoGPT clones
* Fine-tuning
* Cloud deployment
* Authentication
* Databases beyond FAISS
* Advanced UI

These reduce clarity and signal.

---

## 13. README Expectations (for later)

README must explain:

* Problem
* Why RAG
* Why Agent
* Architecture
* Decision flow
* Limitations
* Future improvements

README clarity matters as much as code.

---

## 14. Development Order (IMPORTANT)

1. `ingest.py`
2. `retriever.py`
3. `tools.py`
4. `agent.py`
5. `app.py`
6. README

Do NOT jump steps.

---

## 15. Final Reminder

This project is NOT about:

* showing off
* complexity
* buzzwords

This project IS about:

* correctness
* decision-making
* explainability
* maturity

---
