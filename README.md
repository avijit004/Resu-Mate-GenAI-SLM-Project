# Resu-Mate: Career Document Generator

<div align="center">

**A professional hybrid NLP + SLM application for generating tailored career documents**

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Latest-red.svg)](https://streamlit.io/)
[![Ollama](https://img.shields.io/badge/Ollama-Local-green.svg)](https://ollama.ai/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

*Generate personalized Cold Emails and Cover Letters using advanced semantic matching and Small Language Models*

</div>

---

## 📖 Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [Architecture](#-architecture)
- [Technology Stack](#-technology-stack)
- [Installation](#-installation)
- [Usage](#-usage)
- [Configuration](#-configuration)
- [Technical Details](#-technical-details)
- [Performance](#-performance)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🎯 Overview

**Resu-Mate** is an intelligent career document generator that leverages a **hybrid NLP + SLM approach** to create tailored job application materials. By combining traditional natural language processing techniques (TF-IDF vectorization, embeddings, cosine similarity) with modern Small Language Models (Ollama), Resu-Mate provides accurate, context-aware document generation while maintaining explainability through similarity scores.

### Problem Statement

Job seekers often struggle with:
- **Time-consuming customization**: Tailoring each application takes 30-60 minutes
- **Keyword matching**: Difficulty identifying relevant resume sections for each role
- **Consistency**: Maintaining professional tone across multiple applications
- **Privacy concerns**: Reluctance to upload sensitive documents to cloud services

### Solution

Resu-Mate addresses these challenges by:
- **Automated alignment**: Semantic matching identifies top relevant resume sections
- **Intelligent generation**: SLM creates contextually appropriate documents
- **Local processing**: All data stays on your machine (no cloud APIs)
- **Transparency**: View similarity scores and matching rationale

---

## ✨ Key Features

### Core Functionality

- 📄 **Multi-format Resume Support**: Upload PDF or DOCX files
- 💼 **Job Description Analysis**: Paste any job posting for analysis
- 📧 **Dual Document Generation**: Creates both Cold Email and Cover Letter simultaneously
- 📥 **Export Options**: Download as PDF (cover letter) or TXT (both documents)
- ✏️ **Live Editing**: Review and refine generated content before export
- 📊 **Word Count**: Real-time word count for each document

### Advanced NLP & ML Features

#### 🔍 Semantic Matching Engine
- **TF-IDF Vectorization**: Keyword-based matching using term frequency-inverse document frequency
- **Embedding-based Similarity**: Optional semantic matching via Ollama embeddings (`nomic-embed-text`)
- **Cosine Similarity Scoring**: Quantified similarity scores (0-1) for each resume section
- **Intelligent Filtering**: Pre-filters top K sections above configurable threshold

#### 🤖 Hybrid Processing Pipeline
1. **Stage 1 - Semantic Pre-filtering**: 
   - Splits resume into meaningful sections
   - Computes similarity scores (TF-IDF or embeddings)
   - Ranks and filters top matching sections
   
2. **Stage 2 - SLM Refinement**:
   - Processes filtered sections with Small Language Model
   - Extracts 5-10 human-readable alignment points
   - Generates contextually appropriate documents

#### 🎨 User Interface Enhancements
- **Semantic Toggle**: Enable/disable semantic pre-filtering via sidebar
- **Similarity Score Display**: View ranked resume sections with percentage matches
- **Dual Expanders**: Separate views for semantic scores and LLM-extracted points
- **Real-time Feedback**: See processing stages with progress indicators

---

## 🏗️ Architecture

### System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Presentation Layer                      │
│  ┌─────────────────────────────────────────────────────┐  │
│  │         Streamlit Web App (ui.py, p.py)             │  │
│  │  • Semantic toggle • Similarity scores • Live edit  │  │
│  └─────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                            ↕
┌─────────────────────────────────────────────────────────────┐
│              Application Services (Python)                  │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  │
│  │  config  │  │   doc    │  │ semantic │  │   llm    │  │
│  │   .py    │→ │processing│→ │ matching │→ │ service  │  │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘  │
│                                                             │
│  ┌──────────┐                                              │
│  │   pdf    │                                              │
│  │  export  │                                              │
│  └──────────┘                                              │
└─────────────────────────────────────────────────────────────┘
                            ↕
┌─────────────────────────────────────────────────────────────┐
│                   External Systems                          │
│  ┌─────────────────────────────────────────────────────┐  │
│  │      Ollama Server (localhost:11434)                │  │
│  │  • LLM: llama3.2:3b, phi3:mini                       │  │
│  │  • Embeddings: nomic-embed-text                      │  │
│  └─────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

### Module Structure

| Module | Responsibility | Key Functions |
|--------|---------------|---------------|
| `p.py` | Application entry point | Main execution |
| `ui.py` | Streamlit user interface | User interactions, form handling, display |
| `config.py` | Configuration management | Model settings, thresholds, flags |
| `doc_processing.py` | Document parsing | `extract_text()`, `is_ollama_online()` |
| `semantic_matching.py` | NLP processing | `get_top_matching_sections()`, `compute_tfidf_similarity()`, `compute_embedding_similarity()` |
| `llm_service.py` | SLM orchestration | `extract_alignment_points()`, `generate_stream()`, `get_llm()` |
| `pdf_export.py` | PDF generation | `create_pdf_from_text()` |

### Data Flow

```
User Input (Resume + JD)
    ↓
[doc_processing.py] Extract text from PDF/DOCX
    ↓
[semantic_matching.py] (if enabled)
    ├─ Split resume into sections
    ├─ Compute TF-IDF vectors OR embeddings
    ├─ Calculate cosine similarity scores
    └─ Filter top K sections (threshold: 0.3)
    ↓
[llm_service.py]
    ├─ Extract alignment points (hybrid: semantic + SLM)
    └─ Generate Cold Email & Cover Letter
    ↓
[pdf_export.py] Create PDF from cover letter
    ↓
User Output (Documents + Similarity Scores)
```

---

## 🛠️ Technology Stack

### Core Technologies

- **Frontend**: [Streamlit](https://streamlit.io/) - Interactive web application framework
- **LLM Framework**: [LangChain](https://www.langchain.com/) - Orchestration and prompt management
- **Local LLM**: [Ollama](https://ollama.ai/) - Local Small Language Model server
- **NLP Library**: [scikit-learn](https://scikit-learn.org/) - TF-IDF vectorization, cosine similarity
- **Document Processing**: 
  - `pypdf` - PDF text extraction
  - `python-docx` - DOCX text extraction
- **PDF Generation**: [ReportLab](https://www.reportlab.com/) - Professional PDF creation

### Models Used

| Model | Type | Purpose | Size |
|-------|------|---------|------|
| `llama3.2:3b` | Small Language Model | Document generation, alignment extraction | 3B parameters |
| `phi3:mini` | Small Language Model | Alternative lightweight option | ~3.8B parameters |
| `nomic-embed-text` | Embedding Model | Semantic similarity (optional) | 768 dimensions |

---

## 📦 Installation

### Prerequisites

- Python 3.8 or higher
- [Ollama](https://ollama.ai/) installed and running
- 8GB+ RAM recommended (for running local models)

### Step 1: Clone the Repository

```bash
git clone https://github.com/avijit004/Resu-Mate-GenAI-SLM-Project.git
cd Resu-Mate-GenAI-SLM-Project
```

### Step 2: Create Virtual Environment

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Install Ollama Models

```bash
# Required: Default SLM model
ollama pull llama3.2:3b

# Optional: Alternative lightweight model
ollama pull phi3:mini

# Optional: Embedding model for semantic matching
ollama pull nomic-embed-text
```

### Step 5: Verify Ollama is Running

```bash
# Check if Ollama server is accessible
curl http://localhost:11434
```

If you see a response, Ollama is running. If not, start Ollama (usually runs automatically on installation).

---

## 🚀 Usage

### Starting the Application

```bash
streamlit run p.py
```

The application will open in your default web browser at `http://localhost:8501`.

### Basic Workflow

1. **Upload Resume**: Click "Browse files" and select your resume (PDF or DOCX)
2. **Paste Job Description**: Copy and paste the complete job posting into the text area
3. **Optional Details**: Fill in company name, role title, your name, and recipient position (improves personalization)
4. **Configure Settings** (Sidebar):
   - Select SLM model (`llama3.2:3b` or `phi3:mini`)
   - Adjust creativity/temperature (0.0-1.5)
   - Choose writing tone (Professional, Warm, or Concise)
   - Toggle semantic matching on/off
5. **Generate**: Click "Generate drafts" button
6. **Review & Edit**: 
   - View semantic similarity scores (if enabled)
   - Review LLM-extracted alignment points
   - Edit generated documents in the text areas
   - Preview formatted output
7. **Export**: Download cover letter as PDF or both documents as TXT

### Example Usage

```python
# The application handles everything through the UI, but here's the internal flow:

from doc_processing import extract_text
from semantic_matching import get_top_matching_sections
from llm_service import extract_alignment_points, generate_stream

# 1. Extract text from resume
resume_text = extract_text(uploaded_file)

# 2. Semantic matching (optional)
top_sections = get_top_matching_sections(
    resume_text, 
    job_description, 
    top_k=10,
    use_embeddings=True
)

# 3. Extract alignment points
alignment_points, semantic_info = extract_alignment_points(
    resume_text,
    job_description,
    model_name="llama3.2:3b",
    temperature=0.7
)

# 4. Generate documents
for chunk in generate_stream(
    resume_text,
    job_description,
    doc_type="Cover Letter",
    tone="Professional & Formal",
    company="Google",
    role="Software Engineer",
    model_name="llama3.2:3b",
    temperature=0.7,
    alignment_points=alignment_points
):
    print(chunk, end="")
```

---

## ⚙️ Configuration

Edit `config.py` to customize behavior:

```python
# Default SLM model
DEFAULT_MODEL = "llama3.2:3b"

# Available models
MODEL_OPTIONS = {
    "llama3.2:3b": "llama3.2:3b",
    "phi (mini SLM: phi3:mini)": "phi3:mini",
}

# Embedding model for semantic matching
DEFAULT_EMBEDDING_MODEL = "nomic-embed-text"

# Semantic matching settings
USE_SEMANTIC_MATCHING = True      # Enable/disable semantic pre-filtering
TOP_K_SECTIONS = 10               # Number of top matching sections to extract
SIMILARITY_THRESHOLD = 0.3        # Minimum similarity score (0-1) to include a section
```

### Configuration Options Explained

| Parameter | Default | Description |
|-----------|---------|-------------|
| `DEFAULT_MODEL` | `"llama3.2:3b"` | Primary SLM model for generation |
| `USE_SEMANTIC_MATCHING` | `True` | Enable semantic pre-filtering |
| `TOP_K_SECTIONS` | `10` | Maximum sections to extract |
| `SIMILARITY_THRESHOLD` | `0.3` | Minimum cosine similarity score |
| `DEFAULT_EMBEDDING_MODEL` | `"nomic-embed-text"` | Embedding model for semantic search |

---

## 🔬 Technical Details

### NLP Techniques

#### 1. TF-IDF Vectorization

**Purpose**: Keyword-based text matching using statistical term weighting.

**Implementation**:
```python
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(
    max_features=500,
    stop_words='english',
    ngram_range=(1, 2)  # Unigrams + bigrams
)
tfidf_matrix = vectorizer.fit_transform(texts)
```

**Features**:
- Unigrams and bigrams for phrase matching
- English stop words removal
- Maximum 500 features for efficiency
- Sparse matrix representation

#### 2. Embedding-based Semantic Similarity

**Purpose**: Capture semantic meaning beyond keyword matching.

**Model**: `nomic-embed-text` (768-dimensional vectors)

**Implementation**:
```python
from langchain_ollama import OllamaEmbeddings

embeddings = OllamaEmbeddings(model="nomic-embed-text")
job_embedding = embeddings.embed_query(job_description)
resume_embeddings = embeddings.embed_documents(resume_sections)
```

**Advantages**:
- Understands synonyms and related concepts
- Better for domain-specific terminology
- Captures contextual meaning

#### 3. Cosine Similarity

**Formula**: 
```
cos(θ) = (A · B) / (||A|| × ||B||)
```

**Range**: 0 (no similarity) to 1 (identical)

**Implementation**:
```python
from sklearn.metrics.pairwise import cosine_similarity

similarities = cosine_similarity(job_vector, resume_vectors)
```

**Interpretation**:
- `> 0.7`: Very high similarity
- `0.4 - 0.7`: Moderate similarity
- `0.3 - 0.4`: Low similarity (threshold)
- `< 0.3`: Excluded

#### 4. Hybrid Processing

**Why Hybrid?**
- **Semantic matching**: Fast, explainable, quantifiable
- **SLM refinement**: Context-aware, natural language understanding
- **Combined**: Best of both worlds - accuracy + transparency

**Pipeline**:
```
Resume Sections → TF-IDF/Embeddings → Cosine Similarity → Top K Filter → SLM Processing → Final Documents
```

---

## 📊 Performance

### Processing Times

| Stage | Time | Notes |
|-------|------|-------|
| Text Extraction | ~0.5s | PDF/DOCX parsing |
| Semantic Matching (TF-IDF) | ~1-2s | Fast keyword matching |
| Semantic Matching (Embeddings) | ~3-5s | Slower but more accurate |
| SLM Alignment Extraction | ~5-10s | Depends on model size |
| Document Generation (2 docs) | ~15-30s | Cold Email + Cover Letter |
| **Total Pipeline** | **~20-60s** | End-to-end processing |

### Accuracy Improvements

- **Without semantic pre-filtering**: SLM processes entire resume → slower, less focused
- **With semantic pre-filtering**: SLM focuses on top 10 sections → **30-40% faster**, more relevant output

### Resource Usage

- **Memory**: ~2-4GB RAM (depending on model)
- **CPU**: Moderate usage during generation
- **Disk**: ~5-10GB for models (Ollama storage)

---

## 🎯 Use Cases

### Primary Users

1. **Job Seekers**
   - Quickly tailor applications to multiple positions
   - Ensure consistent professional tone
   - Save 30-45 minutes per application

2. **Career Coaches**
   - Demonstrate resume-JD alignment techniques
   - Show clients how to optimize applications
   - Educational tool for best practices

3. **Recruiters** (Reverse Use Case)
   - Analyze candidate fit based on resume
   - Identify matching skills and experiences
   - Generate candidate summaries

4. **Students & Learners**
   - Understand hybrid NLP + SLM approaches
   - Learn semantic matching techniques
   - Study prompt engineering patterns

---

## 🤝 Contributing

Contributions are welcome! Areas for improvement:

### High Priority
- [ ] Support for additional document formats (RTF, TXT)
- [ ] Multi-language support (beyond English)
- [ ] Batch processing for multiple job applications
- [ ] User profiles and saved resume templates

### Medium Priority
- [ ] Additional embedding models (OpenAI, HuggingFace)
- [ ] Custom prompt templates
- [ ] Export to DOCX format
- [ ] Integration with job boards (LinkedIn, Indeed)

### Low Priority
- [ ] Docker containerization
- [ ] REST API for programmatic access
- [ ] Analytics dashboard
- [ ] A/B testing for prompt variations

### Contribution Guidelines

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

### Permissions
- ✅ Commercial use
- ✅ Modification
- ✅ Distribution
- ✅ Private use

### Limitations
- ❌ Liability
- ❌ Warranty

---

## 🙏 Acknowledgments

- [Ollama](https://ollama.ai/) for providing local LLM infrastructure
- [LangChain](https://www.langchain.com/) for LLM orchestration framework
- [Streamlit](https://streamlit.io/) for rapid web app development
- [scikit-learn](https://scikit-learn.org/) for NLP utilities

---

## 📞 Support & Contact

- **Repository**: [Resu-Mate-GenAI-SLM-Project](https://github.com/avijit004/Resu-Mate-GenAI-SLM-Project)
- **Issues**: [GitHub Issues](https://github.com/avijit004/Resu-Mate-GenAI-SLM-Project/issues)
- **Author**: Avijit

---

<div align="center">

**Built with ❤️ using Python, Streamlit, LangChain, Ollama, and scikit-learn**

⭐ Star this repo if you find it helpful!

</div>
