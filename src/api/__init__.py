"""
REST API layer for the Knowledge Graph system.

This module provides HTTP endpoints for accessing knowledge graph functionality.
It is decoupled from the core SDK and can be replaced with any web framework.
"""

from .server import create_app

__all__ = ['create_app']
