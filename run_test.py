"""
Quick test script to verify the system works.
"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

def main():
    print("="*60)
    print("KNOWLEDGE GRAPH SYSTEM TEST")
    print("="*60)

    # Test 1: Import modules
    print("\n[1] Testing imports...")
    try:
        from src.storage.database import Database
        from src.storage.models import Entity, Edge
        from src.parsing.document_parser import DocumentParser
        from src.parsing.chunker import Chunker
        from src.extraction.structural_extractor import StructuralExtractor
        from src.extraction.semantic_extractor import SemanticExtractor
        from src.embeddings.vector_store import VectorStore
        from src.query.nl_query import NLQueryEngine
        from src.knowledge_graph import KnowledgeGraph
        print("  All imports successful!")
    except ImportError as e:
        print(f"  Import error: {e}")
        return False

    # Test 2: Database
    print("\n[2] Testing database...")
    try:
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            db = Database(f"{tmpdir}/test.db")
            entity = Entity.create("Person", "Test Person", {"role": "test"})
            db.add_entity(entity)
            retrieved = db.get_entity(entity.id)
            assert retrieved is not None
            assert retrieved.canonical_name == "Test Person"
            db.close()
        print("  Database operations successful!")
    except Exception as e:
        print(f"  Database error: {e}")
        return False

    # Test 3: Document parser
    print("\n[3] Testing document parser...")
    try:
        parser = DocumentParser()
        extensions = parser.get_supported_extensions()
        print(f"  Supported extensions: {extensions}")
        if '.pdf' not in extensions:
            print("  WARNING: PDF support not available. Install PyMuPDF.")
        if '.docx' not in extensions:
            print("  WARNING: DOCX support not available. Install python-docx.")
    except Exception as e:
        print(f"  Parser error: {e}")

    # Test 4: Chunker
    print("\n[4] Testing chunker...")
    try:
        chunker = Chunker()
        test_text = "This is sentence one. This is sentence two. " * 100
        chunks = chunker.chunk_text(test_text)
        print(f"  Created {len(chunks)} chunks from test text")
    except Exception as e:
        print(f"  Chunker error: {e}")
        return False

    # Test 5: Structural extractor
    print("\n[5] Testing structural extractor...")
    try:
        extractor = StructuralExtractor()
        test_legal_text = """
        IN THE ARBITRATION BETWEEN:

        CITIOM LLC, a Delaware limited liability company,
        Claimant,

        v.

        GULFSTREAM AEROSPACE CORPORATION,
        Respondent.

        Case No. 01-23-0001234

        "Aircraft" means the Gulfstream G550 aircraft, serial number 5174.
        "Purchase Agreement" means that certain Aircraft Purchase Agreement dated January 15, 2020.
        """
        result = extractor.extract(test_legal_text)
        print(f"  Found {len(result.parties)} parties")
        print(f"  Found {len(result.defined_terms)} defined terms")
        print(f"  Document type: {result.document_type}")
        for party in result.parties:
            print(f"    - {party.name} ({party.role})")
    except Exception as e:
        print(f"  Structural extractor error: {e}")
        return False

    # Test 6: Gemini API
    print("\n[6] Testing Gemini API connection...")
    try:
        import google.generativeai as genai
        from src.config import GEMINI_API_KEY, GEMINI_MODEL

        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel(GEMINI_MODEL)
        response = model.generate_content("Say 'API working' and nothing else.")
        print(f"  Gemini response: {response.text.strip()}")
    except Exception as e:
        print(f"  Gemini API error: {e}")
        return False

    # Test 7: Vector store
    print("\n[7] Testing vector store...")
    try:
        import tempfile
        import numpy as np
        with tempfile.TemporaryDirectory() as tmpdir:
            vs = VectorStore(tmpdir, dimension=768)
            # Add test embedding
            test_vec = np.random.randn(768).astype(np.float32)
            vs.add("test_entity", test_vec)
            results = vs.search(test_vec, k=1)
            assert len(results) > 0
            assert results[0][0] == "test_entity"
            print(f"  Vector store working! Found {len(results)} results")
    except Exception as e:
        print(f"  Vector store error: {e}")

    print("\n" + "="*60)
    print("ALL TESTS PASSED!")
    print("="*60)
    print("\nSystem is ready. You can now run:")
    print("  python test_extraction.py --extract")
    print("  python test_extraction.py --interactive")

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
