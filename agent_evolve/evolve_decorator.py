"""
Decorator for marking functions as optimization candidates.

This decorator marks functions that should be extracted and optimized
by the evolution framework.
"""
from functools import wraps
from typing import Optional, Dict, Any, Callable


class EvolveCandidate:
    """Decorator to mark functions for evolutionary optimization"""
    
    # Class variable to track all decorated functions
    _registered_functions = []
    
    def __init__(
        self, 
        name: Optional[str] = None,
        description: Optional[str] = None,
        category: Optional[str] = None,
        metrics: Optional[list] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the optimization decorator.
        
        Args:
            name: Override name for the tool (defaults to function name)
            description: Description of what the function does
            category: Category (e.g., 'nlg', 'research', 'classification')
            metrics: List of metrics to optimize for
            metadata: Additional metadata for the evolver
        """
        self.name = name
        self.description = description
        self.category = category
        self.metrics = metrics or []
        self.metadata = metadata or {}
    
    def __call__(self, func: Callable) -> Callable:
        """Apply the decorator to a function"""
        
        # Store metadata on the function
        func._evolve_candidate = True
        func._evolve_name = self.name or func.__name__
        func._evolve_description = self.description or func.__doc__
        func._evolve_category = self.category
        func._evolve_metrics = self.metrics
        func._evolve_metadata = self.metadata
        
        # Register the function
        self._registered_functions.append({
            'function': func,
            'name': func._evolve_name,
            'description': func._evolve_description,
            'category': func._evolve_category,
            'metrics': func._evolve_metrics,
            'metadata': func._evolve_metadata,
            'module': func.__module__,
            'qualname': func.__qualname__
        })
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        
        # Preserve optimization metadata on wrapper
        wrapper._evolve_candidate = True
        wrapper._evolve_name = func._evolve_name
        wrapper._evolve_description = func._evolve_description
        wrapper._evolve_category = func._evolve_category
        wrapper._evolve_metrics = func._evolve_metrics
        wrapper._evolve_metadata = func._evolve_metadata
        
        return wrapper
    
    @classmethod
    def get_registered_functions(cls):
        """Get all functions registered for optimization"""
        return cls._registered_functions
    
    @classmethod
    def clear_registry(cls):
        """Clear the registry (useful for testing)"""
        cls._registered_functions = []


# Main decorator - only one needed
evolve = EvolveCandidate


# Example usage:
if __name__ == "__main__":
    
    # Example 1: Simple function
    @evolve(description="Generate a compelling essay on any topic")
    def generate_essay(topic: str, length: int = 500) -> str:
        """Generate an essay on the given topic"""
        return f"Essay about {topic}..."
    
    # Example 2: With specific name and metrics
    @evolve(
        name="linkedin_post_generator",
        metrics=["engagement", "professionalism", "clarity"],
        metadata={"platform": "linkedin", "max_length": 300}
    )
    def create_linkedin_post(topic: str, tone: str = "professional") -> str:
        """Create a LinkedIn post"""
        return f"LinkedIn post about {topic} in {tone} tone"
    
    # Example 3: Class method
    class ContentGenerator:
        @evolve(
            description="Generate tweets with hashtags"
        )
        def generate_tweet(self, topic: str) -> str:
            """Generate a tweet about the topic"""
            return f"Tweet about {topic} #AI #Tech"
    
    # Show registered functions
    print("Registered functions:")
    for func_info in EvolveCandidate.get_registered_functions():
        print(f"- {func_info['name']}: {func_info['description']}")
        print(f"  Category: {func_info['category']}")
        print(f"  Module: {func_info['module']}")
        print(f"  Qualname: {func_info['qualname']}")
        print()