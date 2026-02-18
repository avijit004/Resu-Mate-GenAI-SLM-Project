"""
Semantic Matching Module
Uses embeddings and cosine similarity to find best-matching resume sections.
"""
import re
from typing import List, Tuple

import numpy as np
from langchain_ollama import OllamaEmbeddings
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def split_into_sections(text: str) -> List[str]:
    """Split resume/job description into meaningful sections (bullets, paragraphs)."""
    # Split by bullet points, newlines, or double newlines
    sections = re.split(r"\n\s*[-•*]\s+|\n\n+", text)
    # Filter out very short sections
    return [s.strip() for s in sections if len(s.strip()) > 20]


def compute_tfidf_similarity(resume_sections: List[str], job_desc: str) -> List[Tuple[str, float]]:
    """
    Compute TF-IDF vectorization and cosine similarity between resume sections and job description.
    
    Returns:
        List of (section, similarity_score) tuples, sorted by score descending.
    """
    if not resume_sections:
        return []
    
    # Combine job description with resume sections for TF-IDF
    all_texts = [job_desc] + resume_sections
    
    # TF-IDF vectorization
    vectorizer = TfidfVectorizer(max_features=500, stop_words='english', ngram_range=(1, 2))
    tfidf_matrix = vectorizer.fit_transform(all_texts)
    
    # Job description is first vector
    job_vector = tfidf_matrix[0:1]
    resume_vectors = tfidf_matrix[1:]
    
    # Compute cosine similarity
    similarities = cosine_similarity(job_vector, resume_vectors)[0]
    
    # Pair sections with scores and sort
    scored_sections = list(zip(resume_sections, similarities))
    scored_sections.sort(key=lambda x: x[1], reverse=True)
    
    return scored_sections


def compute_embedding_similarity(
    resume_sections: List[str], 
    job_desc: str, 
    embedding_model: str = "nomic-embed-text"
) -> List[Tuple[str, float]]:
    """
    Compute semantic similarity using Ollama embeddings and cosine similarity.
    
    Args:
        resume_sections: List of resume text sections
        job_desc: Job description text
        embedding_model: Ollama embedding model name (default: nomic-embed-text)
    
    Returns:
        List of (section, similarity_score) tuples, sorted by score descending.
    """
    if not resume_sections:
        return []
    
    try:
        # Initialize Ollama embeddings
        embeddings = OllamaEmbeddings(model=embedding_model)
        
        # Get embeddings
        job_embedding = embeddings.embed_query(job_desc)
        resume_embeddings = embeddings.embed_documents(resume_sections)
        
        # Convert to numpy arrays
        job_vec = np.array(job_embedding)
        resume_vecs = np.array(resume_embeddings)
        
        # Compute cosine similarity
        similarities = cosine_similarity([job_vec], resume_vecs)[0]
        
        # Pair sections with scores and sort
        scored_sections = list(zip(resume_sections, similarities))
        scored_sections.sort(key=lambda x: x[1], reverse=True)
        
        return scored_sections
    except Exception as e:
        # Fallback to TF-IDF if embeddings fail
        return compute_tfidf_similarity(resume_sections, job_desc)


def get_top_matching_sections(
    resume_text: str,
    job_desc: str,
    top_k: int = 10,
    use_embeddings: bool = True,
    embedding_model: str = "nomic-embed-text"
) -> List[Tuple[str, float]]:
    """
    Extract top K most relevant resume sections using semantic matching.
    
    Uses embeddings (if available) or TF-IDF as fallback.
    
    Args:
        resume_text: Full resume text
        job_desc: Job description text
        top_k: Number of top sections to return
        use_embeddings: Whether to use embeddings (True) or TF-IDF (False)
        embedding_model: Ollama embedding model name
    
    Returns:
        List of (section_text, similarity_score) tuples, sorted by relevance.
    """
    # Split resume into sections
    resume_sections = split_into_sections(resume_text)
    
    if not resume_sections:
        return []
    
    # Compute similarity
    if use_embeddings:
        try:
            scored = compute_embedding_similarity(resume_sections, job_desc, embedding_model)
        except Exception:
            # Fallback to TF-IDF
            scored = compute_tfidf_similarity(resume_sections, job_desc)
    else:
        scored = compute_tfidf_similarity(resume_sections, job_desc)
    
    # Return top K
    return scored[:top_k]


def format_semantic_matches(matches: List[Tuple[str, float]]) -> str:
    """
    Format semantic matching results into a readable markdown list.
    
    Args:
        matches: List of (section, score) tuples
    
    Returns:
        Formatted markdown string with sections and similarity scores.
    """
    if not matches:
        return "- No matching sections found."
    
    lines = []
    for section, score in matches:
        # Truncate long sections
        section_short = section[:200] + "..." if len(section) > 200 else section
        # Format score as percentage
        score_pct = f"{score * 100:.1f}%"
        lines.append(f"- [{score_pct} match] {section_short}")
    
    return "\n".join(lines)
