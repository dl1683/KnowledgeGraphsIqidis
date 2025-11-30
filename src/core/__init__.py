"""
Core Knowledge Graph SDK

This module provides the core functionality for building and querying knowledge graphs
from legal documents. It is designed to be used as a standalone library without any
web framework dependencies.

Usage:
    from src.core import KnowledgeGraph

    kg = KnowledgeGraph("my_matter")
    kg.add_document("path/to/document.pdf")
    result = kg.query("Who are the main parties?")
"""

from .knowledge_graph import KnowledgeGraph
from .config import GEMINI_API_KEY, GEMINI_MODEL, MATTERS_DIR

__all__ = [
    'KnowledgeGraph',
    'GEMINI_API_KEY',
    'GEMINI_MODEL',
    'MATTERS_DIR',
]
