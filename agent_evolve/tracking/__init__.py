"""
Agent Evolve Tracking Package

A flexible tracking decorator system that can extract data from nested structures
and track function executions, messages, and custom data across any codebase.

This package provides:
- Generic @track decorator with nested key extraction
- Operation sequencing and call chain analysis  
- Message tracking with role detection
- Multiple storage backends (memory, PostgreSQL, etc.)
- Performance optimizations with async processing
"""

from .decorator import track, track_node
from .config import TrackerConfig
from .tracker import Tracker, get_tracker
from .analysis import analyze_sequences

__version__ = "1.0.0"
__all__ = ["track", "track_node", "TrackerConfig", "Tracker", "get_tracker", "analyze_sequences"]