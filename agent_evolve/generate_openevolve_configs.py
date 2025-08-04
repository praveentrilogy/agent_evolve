#!/usr/bin/env python3
"""
OpenEvolve Configuration Generator

This script generates OpenEvolve configuration files for each tool-evaluator pair
in the evolution framework. Each configuration is optimized for the specific tool's
characteristics and evaluation metrics.

Usage:
    python generate_openevolve_configs.py [tools_directory]
"""

import os
import sys
import yaml
from pathlib import Path
from typing import Dict, List, Any


class OpenEvolveConfigGenerator:
    """Generates OpenEvolve configuration files for tool-evaluator pairs"""
    
    def __init__(self, tools_dir: str = "evolution/tools"):
        self.tools_dir = Path(tools_dir)
    
    def generate_configs(self):
        """Generate OpenEvolve configs for all tool-evaluator pairs"""
        if not self.tools_dir.exists():
            print(f"Error: Tools directory '{self.tools_dir}' does not exist")
            return
        
        # Find all tool directories with evaluators
        tool_dirs = self._find_tool_directories()
        
        if not tool_dirs:
            print("No tool directories with evaluators found")
            return
        
        print(f"Generating OpenEvolve configs for {len(tool_dirs)} tools...")
        
        for tool_dir in tool_dirs:
            tool_name = tool_dir.name
            print(f"Creating config for: {tool_name}")
            self._generate_single_config(tool_dir, tool_name)
        
        print(f"\nOpenEvolve configuration generation completed!")
        print(f"Configs saved alongside each tool in their respective directories")
    
    def _find_tool_directories(self) -> List[Path]:
        """Find all tool directories that have both evolve_target.py and evaluator.py"""
        tool_dirs = []
        
        for item in self.tools_dir.iterdir():
            if item.is_dir():
                tool_file = item / "evolve_target.py"
                evaluator_file = item / "evaluator.py"
                
                if tool_file.exists() and evaluator_file.exists():
                    tool_dirs.append(item)
        
        return sorted(tool_dirs)
    
    def _generate_single_config(self, tool_dir: Path, tool_name: str):
        """Generate OpenEvolve config for a single tool"""
        # Analyze tool characteristics
        tool_category = self._analyze_tool_category(tool_dir)
        metrics = self._extract_evaluator_metrics(tool_dir)
        
        # Create tool-specific configuration
        config = self._create_config_structure(tool_name, tool_category, metrics)
        
        # Save configuration file in the tool's directory
        config_file = tool_dir / "openevolve_config.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, indent=2, sort_keys=False)
        
        print(f"  → {config_file}")
    
    def _analyze_tool_category(self, tool_dir: Path) -> str:
        """Analyze tool source code to determine category"""
        tool_file = tool_dir / "evolve_target.py"
        
        try:
            with open(tool_file, 'r') as f:
                source_code = f.read().lower()
            
            # Categorize based on content analysis
            if any(keyword in source_code for keyword in ['generate', 'create', 'write']):
                if any(output in source_code for output in ['guidelines', 'content', 'text']):
                    return "natural_language_generation"
                else:
                    return "generation"
            elif any(keyword in source_code for keyword in ['research', 'search', 'query']):
                return "research"
            elif any(keyword in source_code for keyword in ['reflect', 'critique', 'analyze']):
                return "analysis"
            elif any(keyword in source_code for keyword in ['has_', 'is_', 'check_']):
                return "classification"
            else:
                return "utility"
                
        except Exception:
            return "utility"
    
    def _extract_evaluator_metrics(self, tool_dir: Path) -> List[str]:
        """Extract metrics from EVALUATION_METRICS variable in evaluator code"""
        evaluator_file = tool_dir / "evaluator.py"
        
        try:
            with open(evaluator_file, 'r') as f:
                evaluator_code = f.read()
            
            import re
            
            # Look for EVALUATION_METRICS = [...] pattern (new approach)
            metrics_pattern = r'EVALUATION_METRICS\s*=\s*\[([^\]]+)\]'
            metrics_match = re.search(metrics_pattern, evaluator_code)
            
            if metrics_match:
                # Extract the content inside the brackets
                metrics_content = metrics_match.group(1)
                # Find all quoted strings (metric names) - try both single and double quotes
                metric_matches = re.findall(r"'([^']+)'", metrics_content)
                if not metric_matches:
                    metric_matches = re.findall(r'"([^"]+)"', metrics_content)
                if metric_matches:
                    print(f"    Found EVALUATION_METRICS: {metric_matches}")
                    return metric_matches[:4]  # Limit to 4 for MAP-Elites
            
            # Fallback: Look for old METRICS = [...] pattern
            metrics_pattern_old = r'METRICS\s*=\s*\[([^\]]+)\]'
            metrics_match_old = re.search(metrics_pattern_old, evaluator_code)
            
            if metrics_match_old:
                # Extract the content inside the brackets
                metrics_content = metrics_match_old.group(1)
                # Find all quoted strings (metric names) - try both single and double quotes
                metric_matches = re.findall(r"'([^']+)'", metrics_content)
                if not metric_matches:
                    metric_matches = re.findall(r'"([^"]+)"', metrics_content)
                if metric_matches:
                    print(f"    Found METRICS: {metric_matches}")
                    return metric_matches[:4]  # Limit to 4 for MAP-Elites
            
            # Final fallback
            print(f"    No metrics variable found, using defaults")
            return ['quality', 'relevance', 'usefulness', 'clarity']
            
        except Exception as e:
            print(f"    Warning: Could not extract metrics from evaluator: {e}")
            return ['quality', 'relevance', 'usefulness', 'clarity']
    
    def _create_config_structure(self, tool_name: str, tool_category: str, metrics: List[str]) -> Dict[str, Any]:
        """Create OpenEvolve configuration structure for the tool"""
        
        # Base configuration
        config = {
            'max_iterations': self._get_iterations_for_category(tool_category),
            'random_seed': 42,
            'checkpoint_interval': 5,
            
            'llm': {
                'models': [
                    {
                        'name': 'gpt-4o',
                        'weight': 1.0
                    }
                ],
                'temperature': self._get_temperature_for_category(tool_category)
            },
            
            'database': {
                'population_size': self._get_population_size(tool_category),
                'num_islands': 3,
                'migration_interval': 25,
                'feature_dimensions': self._get_feature_dimensions(metrics)
            },
            
            'evaluator': {
                'enable_artifacts': True,
                'cascade_evaluation': tool_category in ['natural_language_generation', 'analysis'],
                'use_llm_feedback': tool_category in ['natural_language_generation', 'analysis']
            },
            
            'prompt': {
                'system_message': self._generate_system_message(tool_name, tool_category, metrics),
                'num_top_programs': 3,
                'num_diverse_programs': 2,
                'include_artifacts': True,
                'max_artifact_bytes': 4096,
                'artifact_security_filter': True
            }
        }
        
        # Add tool-specific optimizations using supported fields only
        if tool_category == 'natural_language_generation':
            # Higher bins for creative NLG tasks
            config['database']['feature_bins'] = 12
            
        elif tool_category == 'research':
            # More bins for accuracy-focused research tasks
            config['database']['feature_bins'] = 15
            
        elif tool_category == 'classification':
            # Higher precision bins for classification
            config['database']['feature_bins'] = 15
        else:
            # Default bins for other categories
            config['database']['feature_bins'] = 10
        
        return config
    
    def _get_iterations_for_category(self, category: str) -> int:
        """Get appropriate iteration count based on tool category"""
        iterations_map = {
            'natural_language_generation': 20,   # Max 20 iterations
            'research': 20,                      # Max 20 iterations
            'analysis': 20,                      # Max 20 iterations
            'classification': 20,                # Max 20 iterations
            'utility': 20,                       # Max 20 iterations
            'generation': 20
        }
        return iterations_map.get(category, 20)
    
    def _get_temperature_for_category(self, category: str) -> float:
        """Get appropriate LLM temperature based on tool category"""
        temp_map = {
            'natural_language_generation': 0.8,  # Higher creativity
            'research': 0.3,                     # Lower for factual accuracy
            'analysis': 0.5,                     # Moderate for balanced analysis
            'classification': 0.2,               # Very low for consistency
            'utility': 0.3,                      # Low for deterministic results
            'generation': 0.7
        }
        return temp_map.get(category, 0.5)
    
    def _get_population_size(self, category: str) -> int:
        """Get appropriate population size based on complexity"""
        size_map = {
            'natural_language_generation': 750,  # Larger for diverse outputs
            'research': 500,                     # Standard size
            'analysis': 600,                     # Moderate size
            'classification': 400,               # Smaller for focused tasks
            'utility': 300,                      # Minimal for simple tasks
            'generation': 500
        }
        return size_map.get(category, 400)
    
    
    def _get_feature_dimensions(self, metrics: List[str]) -> List[str]:
        """Use evaluator metrics directly as feature dimensions"""
        # Use the actual metric names from the evaluator
        dimensions = metrics[:4] if len(metrics) >= 4 else metrics
        
        # Ensure we have at least 2 dimensions for MAP-Elites
        if len(dimensions) < 2:
            dimensions.extend(['quality', 'diversity'])
        
        return dimensions[:4]  # Limit to 4 dimensions for manageable grid
    
    
    def _generate_system_message(self, tool_name: str, tool_category: str, metrics: List[str]) -> str:
        """Generate system message for code optimization based on evaluator metrics"""
        
        base_instruction = "You are an expert Python developer optimizing agent tools through evolutionary programming. Your task is to improve the tool's code and prompts to maximize performance on the evaluation metrics."
        
        # Format the actual metrics from the evaluator
        metrics_list = "\n".join([f"- {metric.replace('_', ' ').title()}" for metric in metrics])
        
        if tool_category == 'natural_language_generation':
            return f"""{base_instruction}

You are optimizing the {tool_name} tool code. Focus on improving:

CODE OPTIMIZATION PRIORITIES:
- Enhance prompt engineering for better content quality
- Improve LLM interaction patterns for more engaging output
- Optimize text generation logic for professional tone
- Refine content structure and formatting algorithms
- Enhance context understanding and response relevance

PROMPT OPTIMIZATION FOCUS:
- Craft prompts that generate more engaging content
- Include instructions for professional tone and authenticity
- Add context-awareness to prompts for better relevance
- Optimize prompt structure for coherent, creative output
- Include specific formatting and style guidelines

EVALUATION METRICS TO OPTIMIZE FOR:
{metrics_list}

Modify the tool's implementation, prompts, and logic to score higher on these specific evaluation metrics. Focus on prompt engineering improvements that directly impact the measured quality dimensions."""

        elif tool_category == 'research':
            return f"""{base_instruction}

You are optimizing the {tool_name} tool code. Focus on improving:

CODE OPTIMIZATION PRIORITIES:
- Enhance search query generation and refinement
- Improve source selection and credibility assessment
- Optimize information extraction and synthesis
- Refine result ranking and relevance scoring
- Enhance data validation and fact-checking logic

PROMPT OPTIMIZATION FOCUS:
- Craft prompts for more accurate information gathering
- Include instructions for source credibility evaluation
- Add context for comprehensive topic coverage
- Optimize prompts for objective, unbiased research
- Include specific requirements for current information

EVALUATION METRICS TO OPTIMIZE FOR:
{metrics_list}

Modify the tool's search algorithms, prompts, and validation logic to achieve higher scores on these specific evaluation metrics."""

        elif tool_category == 'analysis':
            return f"""{base_instruction}

You are optimizing the {tool_name} tool code. Focus on improving:

CODE OPTIMIZATION PRIORITIES:
- Enhance analytical reasoning and insight generation
- Improve pattern recognition and trend identification
- Optimize recommendation generation algorithms
- Refine evidence gathering and validation
- Enhance multi-perspective analysis capabilities

PROMPT OPTIMIZATION FOCUS:
- Craft prompts for deeper analytical thinking
- Include instructions for objective evaluation
- Add context for actionable insight generation
- Optimize prompts for evidence-based conclusions
- Include requirements for balanced perspectives

EVALUATION METRICS TO OPTIMIZE FOR:
{metrics_list}

Modify the tool's analysis logic, prompts, and reasoning patterns to maximize scores on these specific evaluation metrics."""

        elif tool_category == 'classification':
            return f"""{base_instruction}

You are optimizing the {tool_name} tool code. Focus on improving:

CODE OPTIMIZATION PRIORITIES:
- Enhance classification accuracy and precision
- Improve edge case handling and boundary detection
- Optimize feature extraction and pattern matching
- Refine decision criteria and threshold tuning
- Enhance consistency across similar inputs

PROMPT OPTIMIZATION FOCUS:
- Craft prompts for more precise categorization
- Include clear classification criteria and examples
- Add instructions for consistent decision-making
- Optimize prompts for edge case handling
- Include specific guidelines for boundary cases

EVALUATION METRICS TO OPTIMIZE FOR:
{metrics_list}

Modify the tool's classification logic, prompts, and decision algorithms to achieve higher scores on these specific evaluation metrics."""

        elif tool_category == 'utility':
            return f"""{base_instruction}

You are optimizing the {tool_name} tool code. Focus on improving:

CODE OPTIMIZATION PRIORITIES:
- Enhance execution efficiency and performance
- Improve error handling and graceful degradation
- Optimize algorithm complexity and resource usage
- Refine input validation and output formatting
- Enhance reliability and robustness

PROMPT OPTIMIZATION FOCUS:
- Craft prompts for efficient task completion
- Include clear instructions and constraints
- Add error handling guidance in prompts
- Optimize for reliable, consistent results
- Include specific output format requirements

EVALUATION METRICS TO OPTIMIZE FOR:
{metrics_list}

Modify the tool's implementation, prompts, and algorithms to maximize scores on these specific evaluation metrics."""

        else:  # Default/generation category
            return f"""{base_instruction}

You are optimizing the {tool_name} tool code. Focus on improving:

CODE OPTIMIZATION PRIORITIES:
- Enhance overall output quality and usefulness
- Improve accuracy and completeness of results
- Optimize performance and reliability
- Refine user experience and usability
- Enhance innovation and effectiveness

PROMPT OPTIMIZATION FOCUS:
- Craft prompts for high-quality output generation
- Include instructions for accuracy and completeness
- Add context for user value optimization
- Optimize prompts for innovative solutions
- Include specific quality and format requirements

EVALUATION METRICS TO OPTIMIZE FOR:
{metrics_list}

Modify the tool's implementation, prompts, and logic to maximize scores on these specific evaluation metrics."""


def main():
    """Main entry point"""
    if len(sys.argv) > 2:
        print("Usage: python generate_openevolve_configs.py [tools_directory]")
        print("Example: python generate_openevolve_configs.py evolution/tools")
        sys.exit(1)
    
    tools_directory = sys.argv[1] if len(sys.argv) == 2 else "evolution/tools"
    
    try:
        generator = OpenEvolveConfigGenerator(tools_directory)
        generator.generate_configs()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()