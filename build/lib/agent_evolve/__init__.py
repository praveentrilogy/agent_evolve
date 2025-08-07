"""
Agent Evolve Package

A comprehensive toolkit for evolving and tracking AI agents, including:
- Evolution framework with decorators and evaluation
- Generic tracking system for function calls and sequences
- Tool extraction and optimization utilities
- Auto-tracing system for complete operation monitoring
"""

from .evolve_decorator import evolve
from .tracking.decorator import track_node
from .auto_tracing import enable_auto_tracing

__version__ = "2.0.0"
__all__ = ['evolve', 'track_node', 'enable_auto_tracing']