"""CLI script to ingest raw university documents and generate normalized dataset."""

import argparse
import logging
from pathlib import Path
import sys

# Ensure backend directory is in sys.path
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
BACKEND_DIR = REPO_ROOT / "backend"

if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.core.config import settings
from app.rag.ingestion import DocumentIngestionService

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("ingest")


def main() -> int:
    parser = argparse.ArgumentParser(description="Ingest LPU CSE-AIML raw documents.")
    parser.add_argument(
        "--input-dir",
        "-i",
        type=Path,
        default=settings.resolved_raw_data_dir,
        help="Path to raw documents directory (default: data/raw)",
    )
    parser.add_argument(
        "--output-file",
        "-o",
        type=Path,
        default=settings.resolved_processed_data_dir / "ingested_documents.json",
        help="Path to save extracted documents JSON (default: data/processed/ingested_documents.json)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable debug logging",
    )
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    input_dir = args.input_dir.resolve()
    output_file = args.output_file.resolve()

    print("=" * 70)
    print(" LPU CSE-AIML RAG CHATBOT — DOCUMENT INGESTION")
    print("=" * 70)
    print(f" Source Directory: {input_dir}")
    print(f" Output File:      {output_file}")
    print("-" * 70)

    if not input_dir.exists():
        print(f"Error: Source directory does not exist: {input_dir}", file=sys.stderr)
        return 1

    service = DocumentIngestionService()
    docs, report = service.ingest_directory(input_dir, recursive=True)

    # Save extracted documents
    if docs:
        service.save_documents(docs, output_file)
        print(f"\n Extracted documents saved to: {output_file}")

    # Display Ingestion Summary Report
    print("\n" + "=" * 70)
    print(" INGESTION SUMMARY REPORT")
    print("=" * 70)
    print(f" Total files scanned:           {report.total_files_scanned}")
    print(f" Successfully extracted files:  {len(report.successful_files)}")
    print(f" Total document units extracted: {report.total_documents_extracted}")
    print("-" * 70)
    print(" Extracted Documents by Category:")
    for cat, count in sorted(report.documents_by_category.items()):
        print(f"   • {cat.capitalize():<18}: {count} document units")

    if report.skipped_files:
        print("-" * 70)
        print(f" Skipped Files ({len(report.skipped_files)}):")
        for f in report.skipped_files:
            print(f"   - {f}")

    if report.unsupported_files:
        print("-" * 70)
        print(f" Unsupported Files ({len(report.unsupported_files)}):")
        for f in report.unsupported_files:
            print(f"   - {f}")

    if report.failed_files:
        print("-" * 70)
        print(f" Failed Files ({len(report.failed_files)}):")
        for item in report.failed_files:
            print(f"   - {item['file']}: {item['error']}")
        print("=" * 70)
        return 1

    print("=" * 70)
    print(" Ingestion completed successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
