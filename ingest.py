"""
scripts/ingest.py
CLI tool to ingest PDF / DOCX / TXT files into the FAISS index.

Usage:
    python scripts/ingest.py --source docs/ --chunk-size 512 --overlap 64
"""

import argparse
from pathlib import Path
from typing import List

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    PyPDFLoader, Docx2txtLoader, TextLoader, DirectoryLoader,
)

from src.core.vector_store import VectorStore
from src.utils.config import settings


def load_directory(path: str) -> List[str]:
    loaders = {
        "**/*.pdf":  PyPDFLoader,
        "**/*.docx": Docx2txtLoader,
        "**/*.txt":  TextLoader,
    }
    docs = []
    for glob, loader_cls in loaders.items():
        loader = DirectoryLoader(path, glob=glob, loader_cls=loader_cls)
        docs.extend(loader.load())
    return docs


def chunk_documents(docs, chunk_size: int, overlap: int) -> List[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ". ", " "],
    )
    chunks = splitter.split_documents(docs)
    return [c.page_content for c in chunks]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="docs/", help="Directory with documents")
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--overlap", type=int, default=64)
    args = parser.parse_args()

    print(f"Loading documents from {args.source} …")
    docs = load_directory(args.source)
    print(f"  Loaded {len(docs)} documents")

    chunks = chunk_documents(docs, args.chunk_size, args.overlap)
    print(f"  Split into {len(chunks)} chunks")

    vs = VectorStore(settings.FAISS_INDEX_PATH, settings.EMBEDDING_MODEL)
    vs.add_documents(chunks)
    print(f"  Index saved → {settings.FAISS_INDEX_PATH}")


if __name__ == "__main__":
    main()
