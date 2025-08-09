"""
Agent Evolve Package

A comprehensive toolkit for evolving and tracking AI agents, including:
- Evolution framework with decorators and evaluation
- Generic tracking system for function calls and sequences
- Tool extraction and optimization utilities
- Safe auto-tracing system for complete operation monitoring
"""


from .tracking.decorator import track_node
from .auto_tracing import enable_auto_tracing
from .trace_tracer import enable_trace_tracing, analyze_prompts

__version__ = "2.0.0"
__all__ = ['evolve', 'track_node', 'enable_auto_tracing', 'trace_call', 'safe_trace_decorator', 'trace_operation', 'enable_trace_tracing', 'analyze_prompts']