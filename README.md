# catalog-mcp

Semantic search MCP server for any project. Indexes code, docs, and configs using ChromaDB + sentence-transformers embeddings.

## Install

```bash
# In any project's venv:
uv pip install -e /path/to/catalog-mcp

# Or from git:
uv pip install git+https://github.com/youruser/catalog-mcp.git
```

## MCP Config

Add to `.kiro/settings/mcp.json` (or equivalent):

```json
{
  "mcpServers": {
    "catalog": {
      "command": ".venv/bin/python",
      "args": ["-m", "catalog_mcp"],
      "autoApprove": [
        "catalog_search",
        "catalog_stats",
        "catalog_list_files",
        "catalog_index",
        "catalog_index_file"
      ]
    }
  }
}
```

The server uses the working directory as the project root — no absolute paths needed.

## Tools

- `catalog_search` — semantic search over indexed files
- `catalog_index` — rebuild the full index
- `catalog_index_file` — re-index a single file
- `catalog_stats` — index health check
- `catalog_list_files` — list what's indexed

## How It Works

- Indexes `.py`, `.md`, `.sh`, `.yaml`, `.json`, `.toml`, `.txt`, `.html` files
- Stores embeddings in `.catalog/chroma/` (add to `.gitignore`)
- Uses `all-MiniLM-L6-v2` (384-dim, runs on CPU, ~22MB)
- Python files are chunked by function/class; markdown by heading; others by line overlap
