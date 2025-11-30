# Legal Knowledge Graph System

A comprehensive system for extracting, building, querying, and visualizing knowledge graphs from legal documents.

## Overview

This project provides an end-to-end pipeline for:
- **Document Ingestion**: Parse PDFs, DOCX, and TXT files
- **Entity Extraction**: Extract people, organizations, dates, money, facts, and more
- **Knowledge Graph Construction**: Build a property graph with typed entities and relationships
- **Natural Language Querying**: Ask questions in plain English
- **Interactive Visualization**: Explore the graph with a web-based D3.js interface

## Features

### Hybrid Extraction Pipeline
- **Structural Extraction**: Pattern-based parsing for high-confidence elements (parties, signatures, defined terms)
- **Semantic Extraction**: AI-powered extraction using Google Gemini for entities, relationships, and facts
- **Entity Resolution**: Fuzzy matching to deduplicate entities across documents

### Entity Types Supported
| Type | Description |
|------|-------------|
| Person | Individuals mentioned in documents |
| Organization | Companies, law firms, government bodies |
| Document | Contracts, filings, exhibits |
| Date | Temporal references |
| Money | Financial amounts |
| Location | Geographic references |
| Reference | Citations and cross-references |
| Fact | Obligations, allegations, key terms |
| Clause | Contract sections |

### Relationship Types
`represents`, `party_to`, `signed`, `employed_by`, `affiliated_with`, `testified`, `references`, `mentioned_in`, `about`, `binds`, `related_to`, `attributed_to`, `defined_as`

## Installation

### Prerequisites
- Python 3.10+
- Google Gemini API key

### Setup

```bash
# Clone the repository
git clone https://github.com/dl1683/KnowledgeGraphsIqidis.git
cd KnowledgeGraphsIqidis

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env and add your Gemini API key:
# GEMINI_API_KEY=your-api-key-here
```

## Usage

### 1. CLI: Batch Document Extraction

Process a folder of documents and build a knowledge graph:

```bash
# Using CLI tool (recommended)
python -m src.cli.extract --matter my_case

# Or with custom documents directory
python -m src.cli.extract --matter my_case --dir ./documents

# Or single file
python -m src.cli.extract --matter my_case --file ./document.pdf
```

### 2. CLI: Query the Knowledge Graph

```bash
# Single query
python -m src.cli.query --matter my_case "Who are the main parties?"

# Interactive mode
python -m src.cli.query --matter my_case --interactive

# List entities by type
python -m src.cli.query --matter my_case --list-entities Person

# Search entities
python -m src.cli.query --matter my_case --search "ACME"

# Show stats
python -m src.cli.query --matter my_case --stats
```

### 3. CLI: Export Data

```bash
# Export full graph to JSON
python -m src.cli.export --matter my_case --output graph.json

# Export for D3.js visualization
python -m src.cli.export --matter my_case --format d3 --output viz.json

# Export entities to CSV
python -m src.cli.export --matter my_case --format csv-entities --output entities.csv

# Export edges to CSV
python -m src.cli.export --matter my_case --format csv-edges --output edges.csv
```

### 4. Start the Visualization Server

```bash
# Default (citiom_v_gulfstream matter)
python visualization_server.py

# Custom matter
python visualization_server.py --matter my_case --port 8080
```

Then open http://localhost:5000 in your browser.

### 5. Programmatic Usage (SDK)

```python
# Recommended import
from src.core import KnowledgeGraph

# Legacy import (also works)
from src import KnowledgeGraph

# Initialize for a specific matter
kg = KnowledgeGraph("my_case")

# Add a document
kg.add_document("path/to/document.pdf")

# Query the graph
result = kg.query("Who are the main parties in this case?")
print(result.answer)

# Get entity summary
summary = kg.get_entity_summary("ACME Corporation")
print(summary)

# List all organizations
orgs = kg.list_entities(entity_type="Organization")
for org in orgs:
    print(f"- {org['name']}")
```

### 4. Natural Language Queries

The system supports various query types:

| Query Type | Example |
|------------|---------|
| Entity Search | "Who are the main parties?" |
| Relationship Query | "How is John Smith related to ACME Corp?" |
| Fact Search | "What obligations are mentioned?" |
| Timeline | "What are the key dates?" |
| Aggregation | "How many documents are in the graph?" |

### 5. Graph Editing

Edit the graph via the web interface or programmatically:

```python
# Merge duplicate entities
kg.merge_entities(keep_id="entity-123", merge_id="entity-456")

# Add a new entity
kg.add_entity(name="New Person", entity_type="Person")

# Add a relationship
kg.add_edge(source_id="entity-123", target_id="entity-789", relation="represents")
```

## Project Structure

The project follows a modular architecture with clear separation of concerns:

```
KnowledgeGraphsIqidis/
├── src/
│   ├── __init__.py               # Package exports (backward compatible)
│   │
│   ├── core/                     # Core SDK (no web dependencies)
│   │   ├── __init__.py           # Exports KnowledgeGraph class
│   │   ├── config.py             # Configuration and constants
│   │   ├── knowledge_graph.py    # Main unified interface
│   │   ├── storage/              # Database layer
│   │   │   ├── database.py       # SQLite operations
│   │   │   └── models.py         # Data models (Entity, Edge, etc.)
│   │   ├── extraction/           # Document processing
│   │   │   ├── extraction_pipeline.py
│   │   │   ├── structural_extractor.py
│   │   │   └── semantic_extractor.py
│   │   ├── parsing/              # Document parsing
│   │   │   ├── document_parser.py
│   │   │   └── chunker.py
│   │   ├── query/                # NL query engine
│   │   │   └── nl_query.py
│   │   └── embeddings/           # Vector search
│   │       └── vector_store.py
│   │
│   ├── api/                      # REST API layer
│   │   ├── __init__.py
│   │   └── server.py             # Flask Blueprint with all endpoints
│   │
│   ├── cli/                      # Command-line tools
│   │   ├── __init__.py
│   │   ├── extract.py            # Batch extraction
│   │   ├── query.py              # Query interface
│   │   └── export.py             # Data export
│   │
│   └── visualization/            # Graph export
│       └── graph_exporter.py     # D3.js data format
│
├── visualization/                # Frontend (static files)
│   └── index.html                # Interactive D3.js visualization
│
├── matters/                      # Per-case data directories
│   └── <matter_name>/
│       ├── documents/            # Source documents
│       ├── graph.db              # SQLite knowledge graph
│       └── embeddings/           # FAISS vector index
│
├── visualization_server.py       # Web server (thin wrapper)
├── requirements.txt              # Python dependencies
└── README.md
```

### Architecture Benefits

- **`src/core/`**: Pure Python SDK - can be used as a library without any web framework
- **`src/api/`**: REST API layer - can be swapped for FastAPI, Django, etc.
- **`src/cli/`**: Standalone CLI tools - no server required
- **`visualization/`**: Frontend - can be replaced with React, Vue, or any framework

## API Endpoints

The visualization server exposes these REST endpoints:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/graph` | GET | Get full graph data for visualization |
| `/api/search?q=term` | GET | Search entities by name |
| `/api/query` | POST | Execute natural language query |
| `/api/entity/<id>` | GET | Get entity details and neighborhood |
| `/api/entity/<id>` | PUT | Update entity properties |
| `/api/entity/<id>` | DELETE | Delete an entity |
| `/api/entity` | POST | Create a new entity |
| `/api/edge` | POST | Create a new relationship |
| `/api/merge` | POST | Merge two entities |
| `/api/nl-edit` | POST | Natural language graph editing |
| `/api/stats` | GET | Get graph statistics |
| `/api/entity-types` | GET | List available entity types |
| `/api/relation-types` | GET | List available relation types |

## Configuration

### Environment Variables

Copy `.env.example` to `.env` and configure:

```bash
# Required: Google Gemini API Key
GEMINI_API_KEY=your-api-key-here

# Optional: Gemini model (default: gemini-2.5-flash-lite)
GEMINI_MODEL=gemini-2.5-flash-lite
```

### Additional Settings

Edit `src/core/config.py` to customize extraction settings:

```python
# Extraction Settings
CHUNK_SIZE = 2000        # Tokens per chunk
CHUNK_OVERLAP = 200      # Token overlap between chunks
RATE_LIMIT_DELAY = 4.5   # Seconds between API calls
```

## Web Visualization Features

The interactive visualization includes:

- **Force-directed graph layout** with multiple layout options (cluster, radial, hierarchical)
- **Type filtering** - Show/hide entity types
- **Search** - Find entities by name
- **Natural language queries** - Ask questions about the graph
- **Entity details panel** - View connections and properties
- **Edit mode** - Modify entities and relationships
- **Confidence indicators** - Visual distinction for confirmed/extracted/inferred entities

## Recent Updates

### Schema-Aware Query Fallback (Latest)
- When queries return no direct results, the system now uses LLM-guided exploration
- Provides the graph schema to the LLM and gets alternative search strategies
- Significantly improves answer quality for ambiguous queries

### Visualization Improvements
- Fixed JavaScript initialization error
- Added right-click context menu for entity editing
- Improved graph loading performance

## Roadmap

- [x] **Architecture Refactoring**: Separate core backend from visualization for better portability
- [x] **CLI Tools**: Standalone command-line tools for extraction, query, and export
- [x] **Schema-Aware Queries**: LLM-guided exploration when queries return no direct results
- [ ] **FastAPI Migration**: Replace Flask with FastAPI for OpenAPI spec generation
- [ ] **Multi-matter Support**: Better UI for managing multiple cases
- [ ] **Export Formats**: GraphML, RDF, Neo4j import
- [ ] **Tests**: Comprehensive unit and integration tests

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Make your changes
4. Run tests: `python -m pytest tests/`
5. Commit: `git commit -m "Add my feature"`
6. Push: `git push origin feature/my-feature`
7. Open a Pull Request

## License

[Add your license here]

## Acknowledgments

- Built with [Google Gemini](https://deepmind.google/technologies/gemini/) for AI extraction
- Visualization powered by [D3.js](https://d3js.org/)
- Vector search using [FAISS](https://github.com/facebookresearch/faiss)
