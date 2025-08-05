#!/usr/bin/env python3
"""
Training Data Generation Script for Agent Evolution Framework

This script generates training data for each extracted tool using an LLM.
The data is used by evaluators to assess tool performance.

Usage:
    python generate_training_data.py [tools_directory] [--num-samples N]
    python generate_training_data.py evolution/tools --num-samples 20
    python generate_training_data.py --num-samples 5
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Any
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage


class TrainingDataGenerator:
    """Generates training data for agent tools using LLM"""
    
    def __init__(self, model_name: str = "gpt-4o", num_samples: int = 10):
        self.model = ChatOpenAI(
            model=model_name, 
            temperature=0.7,  # Higher temperature for more diverse data
            max_tokens=4000,
            timeout=60
        )
        self.num_samples = num_samples
        self.tools_dir = Path("evolution/tools")
    
    def generate_training_data(self, tools_directory: str = None, force: bool = False, specific_tool: str = None):
        """Generate training data for all tools in the directory or a specific tool"""
        if tools_directory:
            self.tools_dir = Path(tools_directory)
        
        if not self.tools_dir.exists():
            print(f"Error: Tools directory '{self.tools_dir}' does not exist")
            return
        
        print(f"Generating training data for tools in: {self.tools_dir}")
        print(f"Number of samples per tool: {self.num_samples}")
        if force:
            print(f"Force mode: Will overwrite existing training data")
        
        # Find tool directories
        if specific_tool:
            # Generate for specific tool only
            tool_dir = self.tools_dir / specific_tool
            if not tool_dir.exists() or not (tool_dir / "evolve_target.py").exists():
                print(f"Error: Tool '{specific_tool}' not found or missing evolve_target.py")
                return
            tool_dirs = [tool_dir]
        else:
            # Find all tool directories
            tool_dirs = [d for d in self.tools_dir.iterdir() if d.is_dir() and (d / "evolve_target.py").exists()]
        
        if not tool_dirs:
            print("No tool directories found")
            return
        
        print(f"Found {len(tool_dirs)} tools to generate training data for")
        
        success_count = 0
        skip_count = 0
        fail_count = 0
        
        for tool_dir in tool_dirs:
            print(f"\n{'='*60}")
            print(f"Processing: {tool_dir.name}")
            
            # Check if training data already exists
            training_data_file = tool_dir / "training_data.json"
            if training_data_file.exists() and not force:
                print(f"  ⚠️  Training data already exists at {training_data_file}")
                print(f"  Use --force to regenerate existing data")
                skip_count += 1
                continue
            elif training_data_file.exists() and force:
                print(f"  🔄 Overwriting existing training data")
            
            # Generate training data
            if self._generate_single_training_data(tool_dir):
                success_count += 1
            else:
                fail_count += 1
        
        print(f"\n{'='*60}")
        print(f"Training data generation completed!")
        print(f"  ✅ Success: {success_count}")
        print(f"  ⏭️  Skipped: {skip_count}")
        print(f"  ❌ Failed: {fail_count}")

    def _get_source_code(self, file_path: Path) -> str:
        """Get the source code for a single tool"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e: 
            print(f"Error reading tool file: {e}")
            return None

    def _generate_single_training_data(self, tool_dir: Path) -> bool:
        """Generate training data for a single tool"""
        tool_name = tool_dir.name
        tool_file = tool_dir / "evolve_target.py"
        metadata_file = tool_dir / "metadata.json"
        training_data_file = tool_dir / "training_data.json"
        
        # Read metadata to determine tool type
        tool_type = "function"  # default
        if metadata_file.exists():
            try:
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                    if metadata.get("category") == "prompt_optimization":
                        tool_type = "prompt"
                        print(f"  Detected prompt template: {tool_name}")
                    else:
                        print(f"  Analyzing function: {tool_name}")
            except:
                print(f"  Analyzing function: {tool_name} (metadata read failed)")
        else:
            print(f"  Analyzing function: {tool_name}")
        
        # Read tool source code
        source_code = self._get_source_code(tool_file)
        if not source_code:
            return False

        print(f"  Generating {self.num_samples} training samples...")
        
        # Handle prompt templates differently
        if tool_type == "prompt":
            return self._generate_prompt_training_data(tool_dir, source_code, training_data_file)
        else:
            return self._generate_function_training_data(tool_dir, source_code, training_data_file)

    def _generate_prompt_training_data(self, tool_dir: Path, source_code: str, training_data_file: Path) -> bool:
        """Generate training data specifically for prompt templates"""
        import re
        
        # Extract the prompt template from source code
        prompt_match = re.search(r'([A-Z_]+)\s*=\s*"""(.*?)"""', source_code, re.DOTALL)
        if not prompt_match:
            print(f"  ❌ Could not extract prompt template from source")
            return False
        
        prompt_name = prompt_match.group(1)
        prompt_content = prompt_match.group(2).strip()
        
        print(f"  📝 Found prompt template: {prompt_name}")
        print(f"  📏 Prompt length: {len(prompt_content)} characters")
        
        # Generate training data specific to this prompt type
        prompt = f"""Generate training data for testing this PROMPT TEMPLATE:

PROMPT TEMPLATE:
{prompt_content}

PROMPT ANALYSIS:
This is a prompt template for intent classification that determines user intent from messages.

TASK: Generate {self.num_samples} diverse test cases for evaluating this prompt.

TRAINING DATA FORMAT:
Each test case should have:
- "user_message": A realistic user message to classify
- "expected_intent": The correct intent category this message should produce

The prompt categorizes messages as:
1. "research_and_write" - User wants to research and write content
2. "improve_and_rewrite_draft" - User wants to improve existing content  
3. "other" - User wants something else

INSTRUCTIONS:
- Generate diverse, realistic user messages
- Cover all three intent categories evenly
- Include edge cases and challenging examples
- Make messages sound natural and varied
- Ensure expected_intent matches one of the three categories exactly

EXAMPLE FORMAT:
[
    {{"user_message": "Write a LinkedIn post about AI trends", "expected_intent": "research_and_write"}},
    {{"user_message": "Make this more engaging", "expected_intent": "improve_and_rewrite_draft"}},
    {{"user_message": "What's the weather today?", "expected_intent": "other"}}
]

Generate {self.num_samples} training examples now.
Return ONLY a valid JSON array. No explanations, no markdown, just the JSON array."""

        return self._execute_llm_generation(prompt, training_data_file)
    
    def _generate_function_training_data(self, tool_dir: Path, source_code: str, training_data_file: Path) -> bool:
        """Generate training data for regular functions"""
        
        prompt = f"""Analyze this Python function and generate diverse, realistic training data for testing different scenarios.

FUNCTION CODE:
{source_code}

TASK: Generate {self.num_samples} varied and realistic training examples that represent different real-world use cases.

CRITICAL FORMAT REQUIREMENT:
Each training example MUST follow this exact structure:
{{
    "inputs": {{
        "<parameter_name>": <parameter_value>,
        ... (for each function parameter)
    }},
    "outputs": <expected_output>
}}

The "inputs" field MUST contain a dictionary with keys matching the function's parameter names EXACTLY.
The "outputs" field contains the expected return value of the function.

TRAINING DATA PHILOSOPHY:
- Focus on VARIETY and DIVERSITY of realistic scenarios
- Generate examples that represent different types of users, contexts, and use cases
- Create meaningful test cases that showcase the function's capabilities across different domains
- Avoid edge cases, error conditions, or empty/null values unless they represent common real-world scenarios
- Prioritize realistic, production-like inputs that users would actually provide

INSTRUCTIONS:
1. Extract the function's parameter names from the signature
2. For each example, create an "inputs" dict with those EXACT parameter names as keys
3. Include "outputs" field with the expected function output
4. Each example should represent a different context, industry, style, or use case
5. Make examples representative of actual user needs and scenarios

EXAMPLE for a function like def add(a: int, b: int) -> int:
[
    {{"inputs": {{"a": 5, "b": 3}}, "outputs": 8}},
    {{"inputs": {{"a": 10, "b": 20}}, "outputs": 30}}
]

EXAMPLE for a function like def generate_fibbonacci_sequence(n: int) -> List[int]:
[
    {{"inputs": {{"n": 5}}, "outputs": [0, 1, 1, 2, 3]}},
    {{"inputs": {{"n": 8}}, "outputs": [0, 1, 1, 2, 3, 5, 8, 13]}}
]

Generate {self.num_samples} diverse, realistic training examples now.
Return ONLY a valid JSON array. No explanations, no markdown, just the JSON array."""

        return self._execute_llm_generation(prompt, training_data_file)
    
    def _execute_llm_generation(self, prompt: str, training_data_file: Path) -> bool:
        """Execute LLM generation and save results"""
        try:
            response = self.model.invoke([HumanMessage(content=prompt)])
            
            # Parse the response
            response_text = response.content.strip()
            
            # Remove markdown code blocks if present
            if response_text.startswith('```json'):
                response_text = response_text[7:]
            elif response_text.startswith('```'):
                response_text = response_text[3:]
            
            if response_text.endswith('```'):
                response_text = response_text[:-3]
            
            response_text = response_text.strip()
            
            # Parse JSON
            training_data = json.loads(response_text)
            
            # Validate it's a list
            if not isinstance(training_data, list):
                print(f"  ❌ Error: Expected a list, got {type(training_data)}")
                return False
            
            # Validate we got the requested number of samples
            if len(training_data) != self.num_samples:
                print(f"  ⚠️  Warning: Got {len(training_data)} samples instead of {self.num_samples}")
            
            # Save training data
            with open(training_data_file, 'w', encoding='utf-8') as f:
                json.dump(training_data, f, indent=2)
            
            print(f"  ✅ Generated {len(training_data)} training samples")
            print(f"  💾 Saved to: {training_data_file}")
            
            # Show a sample
            if training_data:
                print(f"  📝 Sample: {json.dumps(training_data[0], indent=2)}")
            
            return True
            
        except json.JSONDecodeError as e:
            print(f"  ❌ Error parsing JSON response: {e}")
            print(f"  Response: {response.content[:200]}...")
            return False
        except Exception as e:
            print(f"  ❌ Error generating training data: {e}")
            return False


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Generate training data for agent tools",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python generate_training_data.py                    # Default: 10 samples per tool
  python generate_training_data.py --num-samples 20   # Generate 20 samples per tool
  python generate_training_data.py evolution/tools --num-samples 5
        """
    )
    
    parser.add_argument(
        'tools_directory',
        nargs='?',
        default=None,
        help='Directory containing tool subdirectories (default: evolution/tools)'
    )
    
    parser.add_argument(
        '--num-samples',
        '-n',
        type=int,
        default=10,
        help='Number of training samples to generate per tool (default: 10)'
    )
    
    parser.add_argument(
        '--force',
        '-f',
        action='store_true',
        help='Force regeneration of existing training data'
    )
    
    args = parser.parse_args()
    
    # Check for OpenAI API key
    if not os.getenv('OPENAI_API_KEY'):
        print("Error: OPENAI_API_KEY environment variable is required")
        print("Please set your OpenAI API key: export OPENAI_API_KEY=your_key_here")
        sys.exit(1)
    
    # Validate num_samples
    if args.num_samples < 1:
        print("Error: Number of samples must be at least 1")
        sys.exit(1)
    
    if args.num_samples > 100:
        print("Warning: Generating more than 100 samples per tool may be expensive and slow")
        response = input("Continue? (y/N): ")
        if response.lower() != 'y':
            sys.exit(0)
    
    # Initialize generator
    generator = TrainingDataGenerator(num_samples=args.num_samples)
    
    # Generate training data
    generator.generate_training_data(args.tools_directory, force=args.force)


if __name__ == "__main__":
    main()