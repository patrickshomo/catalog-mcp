"""
Catalog store — ChromaDB-backed semantic search over project files.

Handles indexing, chunking, and querying. The ChromaDB collection lives
in PROJECT_ROOT/.catalog/chroma/ by default.

PROJECT_ROOT is determined by the current working directory at launch time,
making this package portable across any project.

Embedding model: all-MiniLM-L6-v2 (22MB, 384-dim, runs on CPU).
"""

import hashlib
import os
import re
import time
from pathlib import Path
from typing import Optional

import chromadb
from chromadb.config import Settings

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# PROJECT_ROOT = cwd at import time (MCP servers launch from workspace root)
PROJECT_ROOT = Path.cwd().resolve()
CATALOG_DIR = PROJECT_ROOT / ".catalog"
CHROMA_DIR = CATALOG_DIR / "chroma"
COLLECTION_NAME = "project_files"

# Directories to skip during indexing
SKIP_DIRS = {
    ".git", ".venv", "venv", "__pycache__", ".hypothesis", ".pytest_cache",
    ".claude", ".kiro", "node_modules", ".chroma", ".catalog", "mlruns",
    "mlartifacts", ".superpowers", "autogluon_models", "cache", "session_data",
}

# Directories whose JSON files are result data, not knowledge
SKIP_JSON_DIRS = {"results", "signals"}

# File extensions we index
INDEXABLE_EXTENSIONS = {
    ".py", ".md", ".sh", ".yaml", ".yml", ".json", ".toml", ".txt", ".html",
}

# Max file size to index (500KB)
MAX_FILE_SIZE = 500_000

# Chunk sizes
MAX_CHUNK_CHARS = 1500
OVERLAP_CHARS = 200


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------


def _chunk_python(text: str, filepath: str) -> list[dict]:
    """Chunk Python files by top-level functions/classes, with fallback."""
    chunks = []
    pattern = re.compile(r"^((?:class |def |async def )\w+)", re.MULTILINE)
    splits = pattern.split(text)

    if len(splits) <= 1:
        return _chunk_text(text, filepath)

    header = splits[0].strip()
    if header:
        chunks.append({"text": header[:MAX_CHUNK_CHARS], "section": "module_header"})

    i = 1
    while i < len(splits):
        sig = splits[i]
        body = splits[i + 1] if i + 1 < len(splits) else ""
        block = (sig + body).strip()
        if block:
            if len(block) > MAX_CHUNK_CHARS:
                for sub in _chunk_text(block, filepath, section=sig.strip()):
                    chunks.append(sub)
            else:
                chunks.append({"text": block, "section": sig.strip()})
        i += 2

    return chunks


def _chunk_markdown(text: str, filepath: str) -> list[dict]:
    """Chunk markdown by headings."""
    chunks = []
    pattern = re.compile(r"^(#{1,3}\s+.+)$", re.MULTILINE)
    parts = pattern.split(text)

    if len(parts) <= 1:
        return _chunk_text(text, filepath)

    preamble = parts[0].strip()
    if preamble:
        chunks.append({"text": preamble[:MAX_CHUNK_CHARS], "section": "preamble"})

    i = 1
    while i < len(parts):
        heading = parts[i].strip()
        body = parts[i + 1] if i + 1 < len(parts) else ""
        block = (heading + "\n" + body).strip()
        if block:
            if len(block) > MAX_CHUNK_CHARS:
                for sub in _chunk_text(block, filepath, section=heading):
                    chunks.append(sub)
            else:
                chunks.append({"text": block, "section": heading})
        i += 2

    return chunks


def _chunk_text(text: str, filepath: str, section: str = "") -> list[dict]:
    """Generic line-based chunking with overlap."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + MAX_CHUNK_CHARS
        chunk_text = text[start:end]
        if chunk_text.strip():
            chunks.append({"text": chunk_text, "section": section or f"offset_{start}"})
        start = end - OVERLAP_CHARS
    return chunks


def chunk_file(filepath: Path, project_root: Path | None = None) -> list[dict]:
    """Read and chunk a file. Returns list of {text, section, filepath, chunk_id}."""
    root = project_root or PROJECT_ROOT
    try:
        text = filepath.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return []

    if not text.strip():
        return []

    rel = str(filepath.relative_to(root))
    ext = filepath.suffix.lower()

    if ext == ".py":
        raw_chunks = _chunk_python(text, rel)
    elif ext == ".md":
        raw_chunks = _chunk_markdown(text, rel)
    else:
        raw_chunks = _chunk_text(text, rel)

    results = []
    for i, c in enumerate(raw_chunks):
        c["filepath"] = rel
        c["chunk_id"] = hashlib.md5(
            f"{rel}::{c['section']}::{i}".encode()
        ).hexdigest()
        results.append(c)

    return results


# ---------------------------------------------------------------------------
# Store (ChromaDB wrapper)
# ---------------------------------------------------------------------------


class CatalogStore:
    """Thin wrapper around a ChromaDB collection for project file search."""

    def __init__(self, project_root: Path | None = None, chroma_dir: Path | None = None):
        self._project_root = project_root or PROJECT_ROOT
        self._chroma_dir = chroma_dir or (self._project_root / ".catalog" / "chroma")
        self._chroma_dir.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(
            path=str(self._chroma_dir),
            settings=Settings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        self._embedder = None  # lazy-loaded

    def _get_embedder(self):
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer
            self._embedder = SentenceTransformer("all-MiniLM-L6-v2")
        return self._embedder

    def _embed(self, texts: list[str]) -> list[list[float]]:
        model = self._get_embedder()
        embeddings = model.encode(texts, show_progress_bar=False)
        return embeddings.tolist()

    # -- Indexing -----------------------------------------------------------

    def index_file(self, filepath: Path) -> int:
        """Index a single file. Returns number of chunks added."""
        chunks = chunk_file(filepath, self._project_root)
        if not chunks:
            return 0

        rel = str(filepath.relative_to(self._project_root))
        self._delete_file(rel)

        ids = [c["chunk_id"] for c in chunks]
        texts = [c["text"] for c in chunks]
        ext = filepath.suffix.lower()
        metadatas = [
            {"filepath": c["filepath"], "section": c["section"], "ext": ext}
            for c in chunks
        ]
        embeddings = self._embed(texts)

        self._collection.add(
            ids=ids,
            documents=texts,
            metadatas=metadatas,
            embeddings=embeddings,
        )
        return len(chunks)

    def index_project(
        self,
        root: Path | None = None,
        extensions: set | None = None,
        skip_dirs: set | None = None,
        force: bool = False,
    ) -> dict:
        """Walk the project tree and index all matching files."""
        root = root or self._project_root
        extensions = extensions or INDEXABLE_EXTENSIONS
        skip_dirs = skip_dirs or SKIP_DIRS

        stats = {"files": 0, "chunks": 0, "skipped": 0, "errors": 0}
        t0 = time.time()

        if force:
            try:
                self._client.delete_collection(COLLECTION_NAME)
            except Exception:
                pass
            self._collection = self._client.get_or_create_collection(
                name=COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"},
            )

        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [
                d for d in dirnames
                if d not in skip_dirs and not d.startswith(".")
            ]

            for fname in filenames:
                fpath = Path(dirpath) / fname
                if fpath.suffix.lower() not in extensions:
                    stats["skipped"] += 1
                    continue
                if fpath.stat().st_size > MAX_FILE_SIZE:
                    stats["skipped"] += 1
                    continue
                if fpath.suffix.lower() == ".json":
                    rel_dir = str(Path(dirpath).relative_to(root))
                    if any(rel_dir.startswith(sd) for sd in SKIP_JSON_DIRS):
                        stats["skipped"] += 1
                        continue

                try:
                    n = self.index_file(fpath)
                    stats["files"] += 1
                    stats["chunks"] += n
                except Exception:
                    stats["errors"] += 1

        stats["elapsed_sec"] = round(time.time() - t0, 1)
        return stats

    def _delete_file(self, rel_path: str):
        """Remove all chunks for a given file path."""
        try:
            results = self._collection.get(where={"filepath": rel_path})
            if results["ids"]:
                self._collection.delete(ids=results["ids"])
        except Exception:
            pass

    # -- Search -------------------------------------------------------------

    def search(
        self,
        query: str,
        n_results: int = 10,
        filepath_filter: str | None = None,
        extension_filter: str | None = None,
    ) -> list[dict]:
        """Semantic search. Returns list of {filepath, section, score, snippet}."""
        embedding = self._embed([query])

        where_filter = None
        if extension_filter:
            ext = extension_filter if extension_filter.startswith(".") else f".{extension_filter}"
            where_filter = {"ext": ext}

        try:
            results = self._collection.query(
                query_embeddings=embedding,
                n_results=n_results * 3 if filepath_filter else n_results,
                where=where_filter,
                include=["documents", "metadatas", "distances"],
            )
        except Exception:
            return []

        hits = []
        if results and results["ids"] and results["ids"][0]:
            for i, doc_id in enumerate(results["ids"][0]):
                meta = results["metadatas"][0][i]
                if filepath_filter and filepath_filter not in meta["filepath"]:
                    continue
                distance = results["distances"][0][i]
                snippet = results["documents"][0][i]
                score = round(1.0 - distance, 4)
                hits.append({
                    "filepath": meta["filepath"],
                    "section": meta.get("section", ""),
                    "score": score,
                    "snippet": snippet[:300] + "..." if len(snippet) > 300 else snippet,
                })
                if len(hits) >= n_results:
                    break

        return hits

    def stats(self) -> dict:
        """Return collection statistics."""
        count = self._collection.count()
        return {
            "total_chunks": count,
            "collection": COLLECTION_NAME,
            "chroma_dir": str(self._chroma_dir),
        }

    def list_indexed_files(self) -> list[str]:
        """Return sorted list of all indexed file paths."""
        all_meta = self._collection.get(include=["metadatas"])
        files = set()
        for m in all_meta["metadatas"]:
            files.add(m["filepath"])
        return sorted(files)
