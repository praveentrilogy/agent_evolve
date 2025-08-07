"""
Configuration constants for agent_evolve package.
"""

import os
from pathlib import Path

# Database configuration
DEFAULT_DB_PATH = ".agent_evolve/db/agent_evolve.db"

def get_db_path() -> str:
    """Get the default database path, ensuring the directory exists."""
    db_path = Path(DEFAULT_DB_PATH)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return str(db_path)

def get_absolute_db_path() -> str:
    """Get the absolute database path."""
    return str(Path(DEFAULT_DB_PATH).resolve())