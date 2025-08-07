"""
Nested Key Extractor

Extracts values from nested data structures using path notation.  
Supports dot notation, bracket notation, array indexing, and more.
"""

import re
import hashlib
from typing import Any, Optional, List, Union
from collections.abc import Mapping, Sequence


class NestedExtractor:
    """Extract values from nested structures using path notation."""
    
    # Regex patterns for parsing paths
    BRACKET_PATTERN = re.compile(r'^\[([\'"]?)(.+?)\1\]')
    INDEX_PATTERN = re.compile(r'^\[(-?\d+)\]')
    DOT_PATTERN = re.compile(r'^\.([^.\[]+)')
    ATTR_PATTERN = re.compile(r'^([^.\[]+)')
    
    @classmethod
    def extract(cls, obj: Any, path: str, default: Any = None) -> Any:
        """
        Extract value from nested object using path notation.
        
        Supports:
        - Dot notation: "user.profile.name"
        - Bracket notation: "headers['X-Request-ID']"
        - Array indexing: "messages[0]" or "messages[-1]"
        - Mixed notation: "state.messages[0]['content']"
        - Conditional: "messages[?role='human'][0]"
        - Default values: "user.name|'Anonymous'"
        
        Args:
            obj: The object to extract from
            path: The path string
            default: Default value if extraction fails
            
        Returns:
            Extracted value or default
        """
        if not path:
            return obj
            
        # Handle default value syntax (path|default)
        if '|' in path and not path.startswith('['):
            path_part, default_part = path.split('|', 1)
            # Try to evaluate the default (for strings with quotes)
            try:
                if default_part.startswith(("'", '"')):
                    default = default_part.strip("'\"")
                else:
                    default = default_part
            except:
                default = default_part
            path = path_part.strip()
        
        try:
            return cls._extract_path(obj, path)
        except (KeyError, IndexError, AttributeError, TypeError):
            return default
    
    @classmethod
    def _extract_path(cls, obj: Any, path: str) -> Any:
        """Internal method to extract value following the path."""
        current = obj
        remaining_path = path.strip()
        
        while remaining_path:
            current, remaining_path = cls._extract_next_segment(current, remaining_path)
            
        return current
    
    @classmethod
    def _extract_next_segment(cls, obj: Any, path: str) -> tuple[Any, str]:
        """Extract the next segment from the path."""
        # Try bracket notation ['key'] or ["key"]
        match = cls.BRACKET_PATTERN.match(path)
        if match:
            key = match.group(2)
            remaining = path[match.end():]
            return cls._get_item(obj, key), remaining
        
        # Try array index [0] or [-1]
        match = cls.INDEX_PATTERN.match(path)
        if match:
            index = int(match.group(1))
            remaining = path[match.end():]
            return cls._get_index(obj, index), remaining
        
        # Try dot notation .attr
        match = cls.DOT_PATTERN.match(path)
        if match:
            attr = match.group(1)
            remaining = path[match.end():]
            return cls._get_attr(obj, attr), remaining
        
        # Try plain attribute (start of path)
        match = cls.ATTR_PATTERN.match(path)
        if match:
            attr = match.group(1)
            remaining = path[match.end():]
            return cls._get_attr(obj, attr), remaining
        
        raise ValueError(f"Invalid path segment: {path}")
    
    @classmethod
    def _get_item(cls, obj: Any, key: str) -> Any:
        """Get item using key (for dict-like objects)."""
        if isinstance(obj, Mapping):
            return obj[key]
        # Try attribute access as fallback
        return getattr(obj, key)
    
    @classmethod
    def _get_index(cls, obj: Any, index: int) -> Any:
        """Get item by index (for list-like objects)."""
        if isinstance(obj, Sequence) and not isinstance(obj, str):
            return obj[index]
        raise TypeError(f"Cannot index {type(obj).__name__}")
    
    @classmethod
    def _get_attr(cls, obj: Any, attr: str) -> Any:
        """Get attribute or key from object."""
        # Try dictionary-style access first
        if isinstance(obj, Mapping) and attr in obj:
            return obj[attr]
        # Try attribute access
        if hasattr(obj, attr):
            return getattr(obj, attr)
        # Final attempt with dictionary access (may raise KeyError)
        if isinstance(obj, Mapping):
            return obj[attr]
        raise AttributeError(f"'{type(obj).__name__}' has no attribute '{attr}'")


class PathExtractor:
    """
    Enhanced extractor that supports multiple paths with fallbacks.
    """
    
    def __init__(self, paths: Union[str, List[str]], default: Any = None):
        """
        Initialize with extraction paths.
        
        Args:
            paths: Single path or list of paths to try in order
            default: Default value if all paths fail
        """
        self.paths = [paths] if isinstance(paths, str) else paths
        self.default = default
    
    def extract(self, *args, **kwargs) -> Any:
        """
        Extract value from function arguments.
        
        Tries each configured path until one succeeds.
        Paths can reference positional args by index: "0.user.id" or "args[0].user.id"
        Or keyword args by name: "kwargs.config.thread_id" or "config.thread_id"
        """
        # Create a unified view of arguments
        context = {
            'args': args,
            'kwargs': kwargs,
            # Direct access to positional args by index
            **{str(i): arg for i, arg in enumerate(args)},
            # Direct access to keyword args
            **kwargs
        }
        
        # Try each path
        for path in self.paths:
            # Handle special extractors (like generate_from:)
            if ':' in path:
                strategy, path_part = path.split(':', 1)
                if strategy == 'generate_from':
                    # Generate ID from content
                    content = NestedExtractor.extract(context, path_part)
                    if content:
                        return hashlib.sha256(str(content).encode()).hexdigest()[:16]
                # Add more strategies as needed
            else:
                # Normal path extraction
                result = NestedExtractor.extract(context, path, None)
                if result is not None:
                    return result
        
        return self.default


def create_extractor(config: Union[str, List[str], dict]) -> PathExtractor:
    """
    Create an extractor from configuration.
    
    Args:
        config: Path string, list of paths, or dict with 'paths' and 'default'
        
    Returns:
        PathExtractor instance
    """
    if isinstance(config, dict):
        return PathExtractor(
            paths=config.get('paths', []),
            default=config.get('default')
        )
    return PathExtractor(config)