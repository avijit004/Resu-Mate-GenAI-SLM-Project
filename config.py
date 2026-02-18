DEFAULT_MODEL = "llama3.2:3b"

MODEL_OPTIONS = {
    "llama3.2:3b": "llama3.2:3b",
    "phi (mini SLM: phi3:mini)": "phi3:mini",
}

# Embedding model for semantic matching
DEFAULT_EMBEDDING_MODEL = "nomic-embed-text"

# Semantic matching settings
USE_SEMANTIC_MATCHING = True  # Enable/disable semantic pre-filtering
TOP_K_SECTIONS = 10  # Number of top matching sections to extract
SIMILARITY_THRESHOLD = 0.3  # Minimum similarity score (0-1) to include a section

