"""
Configuration settings for the Knowledge Graph system.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# API Configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError(
        "GEMINI_API_KEY environment variable is not set. "
        "Please create a .env file with GEMINI_API_KEY=your-api-key or set it in your environment."
    )
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite")

# Chunking Configuration
CHUNK_SIZE = 20000  # tokens (large context for Gemini)
CHUNK_OVERLAP = 1000  # tokens (5% overlap for large chunks)

# Embedding Configuration
EMBEDDING_DIMENSION = 768

# Paths
# Navigate from src/core/config.py up to project root
BASE_DIR = Path(__file__).parent.parent.parent
MATTERS_DIR = BASE_DIR / "matters"

# Entity Types
ENTITY_TYPES = [
    "Person",
    "Organization",
    "Document",
    "Clause",
    "Date",
    "Money",
    "Location",
    "Reference",
    "Fact"
]

# Relation Types (directional)
RELATION_TYPES = [
    "mentioned_in",
    "party_to",
    "represents",
    "signed",
    "defined_as",
    "references",
    "related_to",
    "attributed_to",
    "about",
    "binds",
    "testified",
    "employed_by",
    "affiliated_with"
]

# Confidence Levels
CONFIDENCE_CONFIRMED = "confirmed"  # User-asserted or structural anchors
CONFIDENCE_EXTRACTED = "extracted"  # Semantic pass, no user review
CONFIDENCE_INFERRED = "inferred"    # System-suggested, awaiting confirmation

# Resolution threshold
RESOLUTION_CONFIDENCE_THRESHOLD = 0.7
