#!/usr/bin/env python3
"""Launch the job API (uvicorn)."""

from __future__ import annotations

import os

import uvicorn


def main() -> None:
    host = os.environ.get("VTCLONE_API_HOST", "0.0.0.0")
    port = int(os.environ.get("VTCLONE_API_PORT", "8000"))
    reload = os.environ.get("VTCLONE_API_RELOAD", "").lower() in ("1", "true", "yes")
    uvicorn.run(
        "vtclone.api:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info",
    )


if __name__ == "__main__":
    main()
