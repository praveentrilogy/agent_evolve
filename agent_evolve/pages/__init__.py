"""
Pages module for Agent Evolve Dashboard
"""

from .prompts import render_prompts_page
from .code import render_code_page

__all__ = ['render_prompts_page', 'render_code_page']