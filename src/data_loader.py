import hashlib
import os
from pathlib import Path
from typing import List, Any, Tuple, Dict
from langchain_community.document_loaders import PyPDFLoader, TextLoader, CSVLoader
from langchain_community.document_loaders import Docx2txtLoader
from langchain_community.document_loaders.excel import UnstructuredExcelLoader
from langchain_community.document_loaders import JSONLoader


def _compute_file_hash(file_path: Path) -> str:
    """Compute SHA-256 hash of a file for deterministic document identity."""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()[:16]


def load_documents_with_report(data_dir: str) -> Tuple[List[Any], Dict[str, Any]]:
    """
    Load all supported files from data_dir, attach provenance metadata,
    and return (documents, ingestion_report).
    Supported formats: PDF, TXT, CSV, XLSX, DOCX, JSON.
    """
    data_path = Path(data_dir).resolve()
    documents: List[Any] = []
    report: Dict[str, Any] = {
        "data_dir": str(data_path),
        "total_files": 0,
        "successful_files": 0,
        "failed_files": 0,
        "file_details": [],
        "errors": [],
    }

    if not data_path.exists():
        print(f"[WARNING] Data directory does not exist: {data_dir}")
        return documents, report

    loaders_config = [
        ("PDF", "**/*.pdf", lambda p: PyPDFLoader(str(p))),
        ("TXT", "**/*.txt", lambda p: TextLoader(str(p), encoding="utf-8")),
        ("CSV", "**/*.csv", lambda p: CSVLoader(str(p), encoding="utf-8")),
        ("Excel", "**/*.xlsx", lambda p: UnstructuredExcelLoader(str(p))),
        ("Word", "**/*.docx", lambda p: Docx2txtLoader(str(p))),
        ("JSON", "**/*.json", lambda p: JSONLoader(str(p))),
    ]

    for format_name, glob_pattern, loader_factory in loaders_config:
        matching_files = sorted(list(data_path.glob(glob_pattern)))
        for file_path in matching_files:
            report["total_files"] += 1
            rel_source = file_path.name
            try:
                # Use relative path if inside data_path
                try:
                    rel_source = str(file_path.relative_to(data_path)).replace("\\", "/")
                except ValueError:
                    rel_source = file_path.name

                file_hash = _compute_file_hash(file_path)
                loader = loader_factory(file_path)
                loaded_docs = loader.load()

                # Attach provenance metadata to each loaded document
                for doc in loaded_docs:
                    doc.metadata = dict(doc.metadata or {})
                    doc.metadata["source"] = rel_source
                    doc.metadata["doc_id"] = file_hash
                    doc.metadata["file_name"] = file_path.name

                documents.extend(loaded_docs)
                report["successful_files"] += 1
                report["file_details"].append({
                    "file": rel_source,
                    "format": format_name,
                    "status": "success",
                    "doc_count": len(loaded_docs),
                    "doc_id": file_hash,
                })
                print(f"[INFO] Loaded {format_name}: '{rel_source}' ({len(loaded_docs)} sections/pages)")

            except Exception as exc:
                report["failed_files"] += 1
                err_msg = f"Failed to load {format_name} '{rel_source}': {exc}"
                report["errors"].append({"file": rel_source, "error": str(exc)})
                report["file_details"].append({
                    "file": rel_source,
                    "format": format_name,
                    "status": "error",
                    "error": str(exc),
                })
                print(f"[ERROR] {err_msg}")

    print(f"[INFO] Ingestion complete: {report['successful_files']}/{report['total_files']} files loaded successfully ({len(documents)} total sections).")
    return documents, report


def load_all_documents(data_dir: str) -> List[Any]:
    """
    Backwards-compatible wrapper: returns loaded documents.
    """
    docs, _ = load_documents_with_report(data_dir)
    return docs


if __name__ == "__main__":
    docs, rep = load_documents_with_report("data")
    print(f"Loaded {len(docs)} documents.")
    print("Report summary:", rep)