"""
Test configuration — mock heavy dependencies so tests run without
sentence-transformers, supabase, openai, etc.

Lightweight deps (requests, bs4, lxml, dotenv) are real imports.
"""

import sys
import types
from unittest.mock import MagicMock

# Create mock modules for heavy/optional dependencies not needed in unit tests
_MOCK_MODULES = [
    "sentence_transformers",
    "supabase",
    "openai",
    "playwright",
    "playwright.sync_api",
    "bcrypt",
    "pypdf",
    "tiktoken",
]

for mod_name in _MOCK_MODULES:
    if mod_name not in sys.modules:
        mock_mod = types.ModuleType(mod_name)
        if mod_name == "sentence_transformers":
            mock_mod.SentenceTransformer = MagicMock
        elif mod_name == "supabase":
            mock_mod.create_client = MagicMock(return_value=MagicMock())
        elif mod_name == "openai":
            mock_mod.OpenAI = MagicMock
        elif mod_name == "bcrypt":
            mock_mod.hashpw = MagicMock(return_value=b"$2b$12$fakehash")
            mock_mod.gensalt = MagicMock(return_value=b"$2b$12$salt")
            mock_mod.checkpw = MagicMock(return_value=False)
        elif mod_name == "pypdf":
            mock_mod.PdfReader = MagicMock
        sys.modules[mod_name] = mock_mod

# Set required env vars before config is imported
import os

os.environ.setdefault("SUPABASE_URL", "https://test.supabase.co")
os.environ.setdefault("SUPABASE_SERVICE_KEY", "test-key")
os.environ.setdefault("OPENAI_API_KEY", "test-openai-key")
