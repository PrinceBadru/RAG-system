# 🛡️ DocuMind AI — Enterprise Due Diligence & Audit Suite

> **A production-grade, domain-specialized Retrieval-Augmented Generation (RAG) platform** engineered for enterprise contract review, quarterly financial analysis, and corporate compliance due diligence. Combines **local vector embeddings** and a **pure-Python vector index** for data privacy with **Groq Cloud API** for serverless, zero-GPU token streaming.

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Executive_UI-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Groq](https://img.shields.io/badge/Groq-Cloud_API_(ZDR)-F55036?logo=groq&logoColor=white)](https://groq.com/)
[![Embeddings](https://img.shields.io/badge/Embeddings-Local_MiniLM--L6--v2-10B981)](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
[![LLM](https://img.shields.io/badge/LLM-Qwen_3.8--27B-8B5CF6)](https://groq.com/)
[![Tests](https://img.shields.io/badge/Tests-8%20Passed-10B981?logo=pytest&logoColor=white)](https://pytest.org/)

---

## 💼 Business Problem & Real-World Use Case

Generic "Chat with your PDF" apps are toy demonstrations: they lack business domain expertise, hallucinate without verifiable citations, lose page numbers, and pose massive corporate privacy risks.

**DocuMind AI solves real enterprise diligence bottlenecks:**
1. **Contract Review & Legal Risk (M&A / Procurement):** Instantly audits Master Services Agreements (MSAs) and vendor contracts for liability caps, consequential damage waivers, indemnification traps, and termination penalties.
2. **Financial & Earnings Intelligence (Equity Research / Corporate Finance):** Analyzes 10-K/10-Q SEC filings and earnings releases, extracting revenue, ARR, gross/operating margins, EPS guidance, and macroeconomic risk factors.
3. **Cybersecurity & Regulatory Compliance (CISO / Audit Committees):** Audits enterprise SOPs against SOC 2, ISO 27001, and GDPR Article 33 requirements, checking MFA mandates, Zero Trust controls, and 72-hour breach notification SLAs.

---

## ✨ Key Enterprise Capabilities

### ⚡ 1-Click Executive Due Diligence & Risk Matrix
Generate an audit-grade report synthesizing:
- **Executive Briefing & Metadata:** Parties, governing law, and core transaction purpose.
- **Critical Risk Heatmap Matrix:** Formatted audit table classifying issues by severity (🔴 High Risk, 🟡 Moderate Warning, 🟢 Compliant).
- **Financial & Operational Liabilities:** Explicit dollar amounts, payment terms, and delivery milestones.
- **Regulatory Assessment:** SOC 2, ISO 27001, and GDPR compliance validation.
- **Auditor Recommendations:** Strategic takeaways for corporate decision-makers before signing or investing.
- **Export to Markdown:** Instant `.md` report download for stakeholders.

### 📑 Verifiable Audit Citations & Exhibit Drawers
Real business decisions require audit trails. Every answer produced by DocuMind includes an interactive **Audit Citations & Exhibits** drawer displaying:
- Exact **Page Number** (1-indexed)
- **Chunk ID** and character count
- **Cosine Relevance Match %**
- Verifiable document excerpt block

### 🧠 Page-Aware Sliding Window Chunking (Zero Token Truncation)
Naive paragraph splitting (`\n`) frequently generates chunks exceeding the 512-token limit of MiniLM, resulting in silent truncation and lost clauses. 
- DocuMind uses a **sliding window chunker** (~1,100 characters per chunk with 180-character overlap) that respects sentence boundaries.
- Preserves exact page attribution and rich metadata across the entire document.

### 🔒 Enterprise Privacy & Zero Data Retention (ZDR)
- **100% Local Vectorization:** Text extraction, sliding chunking, and embedding generation happen strictly on local CPU/GPU using `sentence-transformers/all-MiniLM-L6-v2`.
- **Pure-Python Vector Store:** In-memory vector index with zero external databases (no Pinecone, Milvus, or external vectors exposed).
- **ZDR Cloud Inference:** Prompts transmitted via TLS 1.3 to Groq Cloud are processed strictly in volatile memory. Data is **never logged, stored, or used for model retraining**.

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                 DocuMind AI — Modern Executive Dark SaaS UI                 │
│  [Domain Selector]  [Enterprise Samples / Custom Upload]  [Risk Matrix/Chat]│
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                         ┌─────────────▼─────────────┐
                         │   Page-Aware PDF Parser   │
                         │   (pypdf + Sliding Chunk) │
                         │  Tracks Page Numbers &    │
                         │  Sanitizes Injection Tokens│
                         └─────────────┬─────────────┘
                                       │
         ┌─────────────────────────────┴─────────────────────────────┐
         │                                                           │
┌────────▼─────────────────┐                                ┌────────▼─────────────────┐
│  Local Embedding Engine  │                                │   Pure-Python Vector     │
│  all-MiniLM-L6-v2        │                                │   In-Memory Index        │
│  (384-dim L2 Normalised) │ ── Batch Vector Embeddings ──> │   Cosine Distance Scan   │
│  Runs 100% Locally       │                                │   Zero External DBs      │
└──────────────────────────┘                                └────────────┬─────────────┘
                                                                         │
                                                               Top-k Cited Exhibits
                                                                         │
                                                            ┌────────────▼─────────────┐
                                                            │   Enterprise LLM Client  │
                                                            │   (Domain System Prompts)│
                                                            └────────────┬─────────────┘
                                                                         │
                                                            ┌────────────▼─────────────┐
                                                            │   Groq Cloud API         │
                                                            │   Qwen 3.8-27B           │
                                                            │   Zero Data Retention    │
                                                            │   (Tokens Streamed Live) │
                                                            └──────────────────────────┘
```

---

## 📁 Repository Structure

```
RAG-system/
├── main.py                     # Streamlit Executive Dark SaaS Application
├── llm_client.py               # Domain prompts, Groq LLM client & Executive Audit generator
├── local_embedding.py          # Local MiniLM model, batch embedding & citation engine
├── vector_index.py             # Pure Python stdlib in-memory cosine/Euclidean vector store
├── pdf_reader.py               # Page-aware PDF parser, sliding window chunker & sanitizer
├── sample_documents/           # Pre-packaged enterprise test documents
│   ├── saas_master_services_agreement.pdf    # B2B SaaS MSA (Legal & Contracts)
│   ├── q1_corporate_earnings_report.pdf      # Q1 10-Q Earnings Release (Financials)
│   └── corporate_cybersecurity_policy.pdf    # Security & AI Governance SOP (Compliance)
├── scripts/
│   └── create_sample_pdfs.py   # Script to regenerate sample PDFs
├── tests/
│   └── test_rag_pipeline.py    # Pytest test suite (8 unit tests)
├── audit_notes/
│   └── project_improvements.md # Engineering audit & resolved architectural items
├── requirements.txt            # Python dependencies
└── .env.example                # Template for API keys
```

---

## 🚀 Quick Start Guide

### 1. Prerequisites
- Python 3.10+
- Free Groq Cloud API key from [console.groq.com/keys](https://console.groq.com/keys)

### 2. Installation
```bash
# Clone the repository
git clone https://github.com/PrinceBadru/RAG-system.git
cd RAG-system

# Create and activate a virtual environment
python -m venv .venv
# On Windows (PowerShell):
.\.venv\Scripts\Activate.ps1
# On macOS / Linux:
# source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Configure API Key
Create a `.env` file in the project root:
```env
GROQ_API_KEY=gsk_your_groq_api_key_here
```

### 4. Run the Application

Choose whichever method best fits your workflow:

#### ⚡ Method 1: 1-Click Launchers (Easiest on Windows)
- **Command Prompt / Windows Explorer:** Double-click or run [`run_app.bat`](file:///c:/Users/user/Desktop/Personal%20Projects/RAG-system/run_app.bat)
  ```cmd
  run_app.bat
  ```
- **PowerShell:** Run [`run_app.ps1`](file:///c:/Users/user/Desktop/Personal%20Projects/RAG-system/run_app.ps1)
  ```powershell
  .\run_app.ps1
  ```

#### 🛠️ Method 2: Direct Execution via Project Python (No activation needed)
Bypasses Windows PATH issues completely by invoking the virtual environment directly:
- **Windows (PowerShell / CMD):**
  ```powershell
  .\.venv\Scripts\python.exe -m streamlit run main.py
  ```
- **macOS / Linux:**
  ```bash
  ./.venv/bin/python -m streamlit run main.py
  ```

#### 💻 Method 3: Standard Activation
- **Windows (PowerShell):**
  ```powershell
  .\.venv\Scripts\Activate.ps1
  streamlit run main.py
  ```
- **Windows (Command Prompt):**
  ```cmd
  .\.venv\Scripts\activate.bat
  streamlit run main.py
  ```
- **macOS / Linux:**
  ```bash
  source .venv/bin/activate
  streamlit run main.py
  ```

Once launched, Streamlit will automatically open or display your local URL at:
```
http://localhost:8501
```

> [!TIP]
> **Troubleshooting `'streamlit' is not recognized`**:
> If you see `'streamlit' is not recognized as an internal or external command`, it means your virtual environment is not currently activated in your shell or global Python scripts are not on your Windows `PATH`. Use **Method 1 (`run_app.bat` / `.\run_app.ps1`)** or **Method 2 (`.\.venv\Scripts\python.exe -m streamlit run main.py`)** to run without needing to touch Windows environment variables.

---

## 🧪 Running Unit Tests

Run the test suite using either the direct venv Python:
```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_rag_pipeline.py -v
```
Or after activating your virtual environment:
```bash
pytest tests/test_rag_pipeline.py -v
```
All 8 unit tests validate:
- In-memory vector distance metrics and dimension mismatch guards
- Page-aware extraction and sliding-window overlap
- Disallowed LLM control token sanitization (`<|end|>`, `<|system|>`)
- Local MiniLM embedding generation and metadata preservation
- Grounded prompt construction and exhibit citations

---

## 📄 License
MIT License. Created by Prince Badru.
