"""Shared infrastructure: config loading, HTTP client, logging, paths.

Every other package in the workspace depends on this one; nothing here may
depend on `packages.scraper`, `packages.ontology`, etc. Keep it small.
"""

from packages.common.config import Config, load_config
from packages.common.http import HttpClient, HttpError
from packages.common.logging import get_logger
from packages.common.paths import REPO_ROOT, ensure_dir, resolve

__all__ = [
    "Config",
    "HttpClient",
    "HttpError",
    "REPO_ROOT",
    "ensure_dir",
    "get_logger",
    "load_config",
    "resolve",
]
