"""
Knowledge Graph System for Legal Document Discovery

This package provides tools for extracting, building, querying, and
visualizing knowledge graphs from legal documents.

Architecture:
    src/core/       - Core SDK (extraction, storage, query)
    src/api/        - REST API layer
    src/cli/        - Command-line tools
    src/visualization/ - Graph visualization export

Usage:
    # Direct SDK usage
    from src.core import KnowledgeGraph
    kg = KnowledgeGraph("my_matter")

    # Or use legacy imports (backward compatible)
    from src import KnowledgeGraph
"""

# Re-export from core for backward compatibility
from .core import KnowledgeGraph
from .core.config import (
    GEMINI_API_KEY,
    GEMINI_MODEL,
    MATTERS_DIR,
    ENTITY_TYPES,
    RELATION_TYPES,
)

__all__ = [
    'KnowledgeGraph',
    'GEMINI_API_KEY',
    'GEMINI_MODEL',
    'MATTERS_DIR',
    'ENTITY_TYPES',
    'RELATION_TYPES',
]
