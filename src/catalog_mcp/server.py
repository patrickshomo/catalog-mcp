"""
Project Catalog MCP server — semantic search over any codebase.

Exposes tools for searching, indexing, and inspecting the project catalog.
Uses ChromaDB with all-MiniLM-L6-v2 embeddings for semantic similarity.

The server uses the current working directory as the project root,
so it works portably in any project without configuration.

Usage:
    python -m catalog_mcp

mcp.json config (relative paths — portable):
    {
      "mcpServers": {
        "catalog": {
          "command": ".venv/bin/python",
          "args": ["-m", "catalog_mcp"]
        }
      }
    }
"""

from pathlib import Path

from mcp.server.fastmcp import FastMCP

from catalog_mcp.store import CatalogStore

# ---------------------------------------------------------------------------
# Server setup
# ---------------------------------------------------------------------------

_store: CatalogStore | None = None


def _get_store() -> CatalogStore:
    global _store
    if _store is None:
        _store = CatalogStore()
    return _store


mcp = FastMCP(
    "catalog",
    instructions=(
        "Project catalog tools for semantic search over the codebase. "
        "Use `catalog_search` to find files, functions, and documentation by meaning. "
        "Use `catalog_index` to rebuild the index after significant changes. "
        "Use `catalog_index_file` to re-index a single file after editing it. "
        "Use `catalog_stats` to check index health. "
        "Use `catalog_list_files` to see what's indexed. "
        "The index covers .py, .md, .sh, .yaml, .json, .toml, .txt, .html files. "
        "Search queries should be natural language descriptions of what you're looking for."
    ),
)


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


@mcp.tool()
async def catalog_search(
    query: str,
    n_results: int = 10,
    filepath_filter: str = "",
    extension_filter: str = "",
) -> str:
    """Search the project catalog using semantic similarity.

    Find files, functions, documentation, and code by describing what you're
    looking for in natural language. Results are ranked by relevance.

    Args:
        query: Natural language search query. Be descriptive.
               Examples: "ATR trailing stop loss implementation",
                         "how the backtest orchestrator works"
        n_results: Max number of results to return (default 10).
        filepath_filter: Only return results from files whose path contains
                         this substring. E.g. "backtesting", "src/trading".
        extension_filter: Only return files with this extension.
                          E.g. ".py", ".md", ".sh"
    """
    store = _get_store()
    results = store.search(
        query=query,
        n_results=n_results,
        filepath_filter=filepath_filter or None,
        extension_filter=extension_filter or None,
    )

    if not results:
        return "No results found. Try broadening your query or check that the index is built (use catalog_index)."

    lines = []
    for i, r in enumerate(results, 1):
        score_pct = f"{r['score'] * 100:.1f}%"
        lines.append(f"[{i}] {r['filepath']}  |  section: {r['section']}  |  relevance: {score_pct}")
        lines.append(r["snippet"])
        lines.append("")

    lines.append(f"--- {len(results)} results ---")
    return "\n".join(lines)


@mcp.tool()
async def catalog_index(force: bool = False) -> str:
    """Rebuild the project-wide semantic search index.

    Crawls all .py, .md, .sh, .yaml, .json, .toml, .txt, .html files,
    chunks them, generates embeddings, and stores in ChromaDB.

    This takes 1-3 minutes depending on project size. Run after major
    changes or when search results seem stale.

    Args:
        force: If True, wipe the existing index and rebuild from scratch.
               If False (default), incrementally update.
    """
    store = _get_store()
    stats = store.index_project(force=force)
    return (
        f"Indexing complete.\n"
        f"  Files indexed: {stats['files']}\n"
        f"  Chunks created: {stats['chunks']}\n"
        f"  Files skipped: {stats['skipped']}\n"
        f"  Errors: {stats['errors']}\n"
        f"  Time: {stats['elapsed_sec']}s"
    )


@mcp.tool()
async def catalog_index_file(filepath: str) -> str:
    """Re-index a single file after editing it.

    Use this after modifying a file to keep the search index current.
    Much faster than a full re-index.

    Args:
        filepath: Path to the file, relative to project root.
                  E.g. "src/trading/exit_evaluator.py"
    """
    store = _get_store()
    from catalog_mcp.store import PROJECT_ROOT

    fpath = Path(filepath)
    if not fpath.is_absolute():
        fpath = PROJECT_ROOT / fpath
    if not fpath.exists():
        return f"File not found: {filepath}"

    n = store.index_file(fpath)
    rel = str(fpath.relative_to(PROJECT_ROOT))
    return f"Indexed {rel}: {n} chunks"


@mcp.tool()
async def catalog_stats() -> str:
    """Show index statistics — total chunks, storage location."""
    store = _get_store()
    s = store.stats()
    return (
        f"Total chunks indexed: {s['total_chunks']}\n"
        f"Collection: {s['collection']}\n"
        f"Storage: {s['chroma_dir']}"
    )


@mcp.tool()
async def catalog_list_files(path_filter: str = "") -> str:
    """List all files currently in the search index.

    Args:
        path_filter: Optional substring to filter file paths.
                     E.g. "autoresearch" to see only auto-research files.
    """
    store = _get_store()
    files = store.list_indexed_files()
    if path_filter:
        files = [f for f in files if path_filter in f]
    if not files:
        return "No files found in index (matching filter). Run catalog_index to build."
    return f"{len(files)} files:\n" + "\n".join(f"  {f}" for f in files)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main():
    mcp.run()


if __name__ == "__main__":
    main()
