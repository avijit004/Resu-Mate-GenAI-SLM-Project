# Resu-Mate: Career Doc Generator

A professional **Streamlit + Ollama + LangChain** application that generates tailored **Cold Emails** and **Cover Letters** from your resume and job descriptions using a hybrid **NLP + SLM** approach.

## 🚀 Features

### Core Functionality
- **Resume Upload**: Supports PDF and DOCX formats
- **Job Description Analysis**: Paste any job posting
- **Dual Document Generation**: Creates both Cold Email and Cover Letter
- **PDF Export**: Download polished cover letters as PDFs
- **Live Editing**: Review and refine generated content before export

### Advanced NLP Features (NEW!)

#### 1. **Semantic Matching** (TF-IDF + Embeddings)
- Uses **TF-IDF vectorization** to find keyword matches
- Optional **Ollama embeddings** (`nomic-embed-text`) for semantic similarity
- **Cosine similarity** scoring to rank resume sections
- Pre-filters most relevant sections before SLM processing

#### 2. **Hybrid Approach**
- **Step 1**: Semantic matching identifies top K resume sections (configurable)
- **Step 2**: SLM refines and extracts alignment points from filtered sections
- **Result**: More accurate, focused matching with explainable similarity scores

#### 3. **Technical Stack**
- **Traditional NLP**: TF-IDF vectorization, cosine similarity
- **Modern SLM**: Ollama (llama3.2:3b, phi3:mini) via LangChain
- **Embeddings**: Optional Ollama embeddings for semantic search

## 📋 Requirements

```bash
pip install -r requirements.txt
```

### Key Dependencies
- `streamlit` - Web UI
- `langchain-core`, `langchain-ollama` - SLM framework
- `ollama` - Local SLM server client
- `numpy`, `scikit-learn` - NLP & ML (TF-IDF, cosine similarity)
- `pypdf`, `python-docx` - Document parsing
- `reportlab` - PDF generation

## 🛠️ Setup

1. **Install Ollama** and pull models:
   ```bash
   ollama pull llama3.2:3b
   ollama pull phi3:mini
   ollama pull nomic-embed-text  # Optional: for semantic embeddings
   ```

2. **Start Ollama server** (usually runs automatically)

3. **Run the app**:
   ```bash
   streamlit run p.py
   ```

## 🏗️ Architecture

### Module Structure
```
p.py                    # Entry point
ui.py                   # Streamlit UI & user interactions
config.py               # Configuration (models, settings)
doc_processing.py       # PDF/DOCX text extraction
semantic_matching.py    # TF-IDF, embeddings, cosine similarity
llm_service.py          # SLM generation (hybrid: semantic + SLM)
pdf_export.py           # PDF generation
```

### Data Flow
1. **Input**: Resume (PDF/DOCX) + Job Description
2. **Text Extraction**: `doc_processing.py` extracts plain text
3. **Semantic Pre-filtering** (optional):
   - Split resume into sections
   - Compute TF-IDF vectors or embeddings
   - Rank by cosine similarity with job description
   - Filter top K sections above threshold
4. **SLM Alignment Extraction**:
   - Send filtered sections + job description to SLM
   - Extract 5-10 key matching points
5. **Document Generation**:
   - Generate Cold Email and Cover Letter
   - Include "Key Highlights" section with matching bullets
6. **Export**: PDF (cover letter) or TXT (both)

## ⚙️ Configuration

Edit `config.py`:

```python
DEFAULT_MODEL = "llama3.2:3b"
USE_SEMANTIC_MATCHING = True  # Enable/disable semantic pre-filtering
TOP_K_SECTIONS = 10           # Number of top sections to extract
SIMILARITY_THRESHOLD = 0.3    # Minimum similarity score (0-1)
DEFAULT_EMBEDDING_MODEL = "nomic-embed-text"
```

## 📊 NLP Techniques Used

### 1. TF-IDF Vectorization
- **Purpose**: Keyword-based matching
- **Implementation**: `sklearn.feature_extraction.text.TfidfVectorizer`
- **Features**: Unigrams + bigrams, English stop words removed
- **Output**: Sparse matrix of TF-IDF scores

### 2. Embeddings (Optional)
- **Purpose**: Semantic similarity beyond keywords
- **Model**: `nomic-embed-text` via Ollama
- **Implementation**: `langchain_ollama.OllamaEmbeddings`
- **Fallback**: TF-IDF if embeddings unavailable

### 3. Cosine Similarity
- **Purpose**: Measure similarity between vectors
- **Formula**: `cos(θ) = (A · B) / (||A|| × ||B||)`
- **Implementation**: `sklearn.metrics.pairwise.cosine_similarity`
- **Range**: 0 (no similarity) to 1 (identical)

### 4. SLM Refinement
- **Purpose**: Extract human-readable alignment points
- **Model**: llama3.2:3b (default) or phi3:mini
- **Approach**: Prompt engineering with filtered resume sections

## 🎯 Use Cases

- **Job Seekers**: Quickly tailor applications to multiple positions
- **Career Coaches**: Demonstrate resume-JD alignment
- **Recruiters**: Analyze candidate fit (reverse use case)
- **Students**: Learn NLP + SLM hybrid approaches

## 🔬 Technical Highlights

### Why Hybrid Approach?
- **Semantic matching** (TF-IDF/embeddings) is fast and explainable
- **SLM refinement** adds context understanding and natural language
- **Combined**: More accurate than either alone

### Performance
- **Semantic matching**: ~1-2 seconds (TF-IDF) or ~3-5 seconds (embeddings)
- **SLM generation**: ~10-30 seconds per document (depends on model)
- **Total**: ~20-60 seconds for full pipeline

## 📝 License

MIT License - feel free to use and modify.

## 🤝 Contributing

Contributions welcome! Areas for improvement:
- Support for more document formats
- Additional embedding models
- Multi-language support
- Batch processing for multiple jobs

---

**Built with**: Python, Streamlit, LangChain, Ollama, scikit-learn, ReportLab
