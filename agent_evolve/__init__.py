"""
Agent Evolve Package

A comprehensive toolkit for evolving and tracking AI agents, including:
- Evolution framework with decorators and evaluation
- Generic tracking system for function calls and sequences
- Tool extraction and optimization utilities
"""

from .evolve_decorator import evolve
from .tracking.decorator import track_node

__version__ = "0.1.0"
__all__ = ['evolve', 'track_node']