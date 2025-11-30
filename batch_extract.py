"""
Batch extraction script for processing all documents.

Features:
- Processes all documents from a directory recursively
- Skips already-processed documents
- Logs progress to file
- Handles errors gracefully
- Provides progress estimates
"""
import sys
import time
from pathlib import Path
from datetime import datetime, timedelta

sys.path.insert(0, str(Path(__file__).parent))

from src.knowledge_graph import KnowledgeGraph
from src.config import GEMINI_API_KEY

# Configuration
MATTER_NAME = "citiom_v_gulfstream"
DOCUMENTS_DIR = r"C:\Users\devan\Downloads\CITIOM v Gulfstream\documents"
LOG_FILE = "batch_extract.log"


def log(message: str):
    """Log message to console and file."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_line = f"[{timestamp}] {message}"
    print(log_line, flush=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(log_line + "\n")
        f.flush()


def get_all_documents(directory: str) -> list:
    """Get all supported documents recursively."""
    supported = ['.pdf', '.docx', '.doc']
    docs = []
    for ext in supported:
        docs.extend(Path(directory).rglob(f"*{ext}"))
    return sorted(docs)


def main():
    log("="*60)
    log("BATCH EXTRACTION STARTED")
    log("="*60)

    # Initialize knowledge graph
    log(f"Initializing knowledge graph: {MATTER_NAME}")
    kg = KnowledgeGraph(MATTER_NAME, api_key=GEMINI_API_KEY)

    # Get initial stats
    stats = kg.get_stats()
    log(f"Initial graph: {stats['entities']} entities, {stats['edges']} edges, {stats['documents']} documents")

    # Find all documents
    all_docs = get_all_documents(DOCUMENTS_DIR)
    log(f"Found {len(all_docs)} documents to process")

    # Track progress
    processed = 0
    skipped = 0
    failed = 0
    start_time = time.time()

    for i, doc_path in enumerate(all_docs):
        doc_num = i + 1
        log(f"\n[{doc_num}/{len(all_docs)}] Processing: {doc_path.name}")

        try:
            # Process document (skip_if_exists=True will skip already processed)
            result = kg.add_document(str(doc_path))

            if result:
                if "already processed" in str(result).lower():
                    skipped += 1
                    log(f"  -> Skipped (already processed)")
                else:
                    processed += 1
                    log(f"  -> Success: {result}")
            else:
                log(f"  -> No result returned")

        except Exception as e:
            failed += 1
            log(f"  -> ERROR: {str(e)[:200]}")
            # Continue with next document
            continue

        # Progress update every 5 documents
        if doc_num % 5 == 0:
            elapsed = time.time() - start_time
            avg_per_doc = elapsed / doc_num
            remaining = (len(all_docs) - doc_num) * avg_per_doc
            eta = datetime.now() + timedelta(seconds=remaining)
            log(f"\n  Progress: {doc_num}/{len(all_docs)} ({100*doc_num/len(all_docs):.1f}%)")
            log(f"  ETA: {eta.strftime('%H:%M:%S')} ({remaining/60:.1f} min remaining)")

            # Get current stats
            current_stats = kg.get_stats()
            log(f"  Current graph: {current_stats['entities']} entities, {current_stats['edges']} edges")

    # Final summary
    elapsed = time.time() - start_time
    final_stats = kg.get_stats()

    log("\n" + "="*60)
    log("BATCH EXTRACTION COMPLETE")
    log("="*60)
    log(f"Total time: {elapsed/60:.1f} minutes")
    log(f"Documents processed: {processed}")
    log(f"Documents skipped: {skipped}")
    log(f"Documents failed: {failed}")
    log(f"\nFinal graph statistics:")
    log(f"  Entities: {final_stats['entities']}")
    log(f"  Edges: {final_stats['edges']}")
    log(f"  Documents: {final_stats['documents']}")
    log(f"  Entity types: {final_stats['entities_by_type']}")

    kg.close()
    log("\nDone!")


if __name__ == "__main__":
    main()
