#!/usr/bin/env python3
"""
Evaluator Generation Script for Agent Evolution Framework

This script uses an LLM to generate OpenEvolve-compatible evaluators
for each extracted tool. The evaluators assess tool performance across
multiple dimensions relevant to agent capabilities.

Usage:
    python generate_evaluators.py [tools_directory]
"""

import os
import sys
import json
from pathlib import Path
from typing import Dict, List, Any
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage


class EvaluatorGenerator:
    """Generates OpenEvolve evaluators for agent tools using LLM"""
    
    def __init__(self, model_name: str = "gpt-4o"):
        # Use higher token limits for models that support it
        max_tokens = 16000 if "gpt-4o" in model_name else 8000
        
        self.model = ChatOpenAI(
            model=model_name, 
            temperature=0.0,
            max_tokens=max_tokens,
            timeout=120
        )
        self.tools_dir = Path("evolution/tools")
    
    def generate_evaluators(self, tools_directory: str = None):
        """Generate evaluators for all tools in the directory"""
        if tools_directory:
            self.tools_dir = Path(tools_directory)
        
        if not self.tools_dir.exists():
            print(f"Error: Tools directory '{self.tools_dir}' does not exist")
            return
        
        print(f"Generating evaluators for tools in: {self.tools_dir}")
        
        # Find all tool directories
        tool_dirs = [d for d in self.tools_dir.iterdir() if d.is_dir() and (d / "evolve_target.py").exists()]
        
        if not tool_dirs:
            print("No tool directories found")
            return
        
        print(f"Found {len(tool_dirs)} tools to generate evaluators for")
        
        for tool_dir in tool_dirs:
            print(f"\nGenerating evaluator for: {tool_dir.name}")
            self._generate_single_evaluator(tool_dir)
        
        print("\nEvaluator generation completed!")

    def _get_source_code(self, file_path: Path):
        """Get the source code for a single tool"""
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e: 
            print(f"Error reading tool file: {e}")
            return

    def _fix_common_import_errors(self, code: str) -> str:
        """Fix common import errors and enforce raw OpenAI API usage"""
        import re
        
        # Replace langchain imports with raw OpenAI
        code = re.sub(
            r'from langchain_openai import.*',
            'import openai',
            code
        )
        code = re.sub(
            r'from langchain_core\.messages import.*',
            '',
            code
        )
        
        # Replace langchain LLM usage with raw OpenAI client
        code = re.sub(
            r'ChatOpenAI\([^)]*\)',
            'openai.OpenAI()',
            code
        )
        
        # Replace langchain message patterns with raw OpenAI API calls
        code = re.sub(
            r'llm\.generate\(\[HumanMessage\(content=([^)]+)\)\]\)',
            r'client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": \1}])',
            code
        )
        code = re.sub(
            r'llm\.invoke\(\[HumanMessage\(content=([^)]+)\)\]\)',
            r'client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": \1}])',
            code
        )
        
        # Fix response access patterns
        code = re.sub(
            r'response\[0\]\.content',
            'response.choices[0].message.content',
            code
        )
        code = re.sub(
            r'response\.content',
            'response.choices[0].message.content',
            code
        )
        
        return code

    def _validate_evaluator_code(self, code: str) -> bool:
        """Validate that the evaluator code is syntactically correct and functional"""
        try:
            # Check for required function
            if 'def evaluate(' not in code:
                print("  Missing 'def evaluate(' function")
                return False
            
            # Check syntax by compiling
            compile(code, '<string>', 'exec')
            
            # Check that evaluate function has a return statement
            import ast
            tree = ast.parse(code)
            
            # Find the evaluate function
            evaluate_func = None
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef) and node.name == 'evaluate':
                    evaluate_func = node
                    break
            
            if not evaluate_func:
                print("  Could not find evaluate function in AST")
                return False
            
            # Check that evaluate function has at least one return statement
            has_return = False
            for node in ast.walk(evaluate_func):
                if isinstance(node, ast.Return):
                    has_return = True
                    break
            
            if not has_return:
                print("  evaluate function has no return statement")
                return False
            
            print("  ✅ Evaluator code validation passed")
            return True
            
        except SyntaxError as e:
            print(f"  Syntax error in generated code: {e}")
            return False
        except Exception as e:
            print(f"  Validation error: {e}")
            return False

    def _validate_evaluator_with_llm(self, code: str) -> dict:
        """Use LLM to validate and potentially fix evaluator code"""
        validation_prompt = f"""You are a Python code validator. Analyze this evaluator code and check for common issues:

CODE TO VALIDATE:
```python
{code}
```

CHECK FOR THESE ISSUES:
1. Syntax errors (try compiling the code)
2. Import errors (wrong imports like 'Message' instead of proper OpenAI API)
3. API usage errors (using langchain instead of raw OpenAI API)
4. Missing error handling
5. Incorrect response parsing
6. Missing required functions (def evaluate)

RESPONSE FORMAT:
Return a JSON with:
- "is_valid": boolean (true if code is correct)
- "issues": list of strings describing problems found
- "suggested_fix": string containing corrected code (only if fixable), or null

If the code needs fixing, provide the complete corrected code in suggested_fix.
Use raw OpenAI API: openai.OpenAI(), client.chat.completions.create(), response.choices[0].message.content

Return ONLY JSON, no explanations."""

        try:
            response = self.model.invoke([HumanMessage(content=validation_prompt)])
            result = json.loads(response.content)
            return result
        except Exception as e:
            print(f"LLM validation failed: {e}")
            # Fallback to basic validation
            return {
                "is_valid": self._validate_evaluator_code(code),
                "issues": ["LLM validation failed, using basic validation"],
                "suggested_fix": None
            }

    def _analyze_function_and_metrics(self, tool_dir: Path) -> tuple:
        """Use LLM to analyze function and determine relevant metrics"""
        tool_name = tool_dir.name
        tool_file = tool_dir / "evolve_target.py"
        
        # Read tool source code
        source_code = self._get_source_code(tool_file)
        
        # Check if this is an intent classification task by examining training data
        training_data_file = tool_dir / "training_data.json"
        is_intent_classification = False
        training_sample = ""
        
        if training_data_file.exists():
            try:
                with open(training_data_file, 'r') as f:
                    training_data = json.load(f)
                    if training_data and isinstance(training_data[0], dict):
                        sample = training_data[0]
                        # Check for intent classification pattern
                        if "expected_intent" in sample:
                            is_intent_classification = True
                            training_sample = f"Sample training data: {sample}"
                            print(f"  🎯 Detected intent classification task")
            except Exception as e:
                print(f"  ⚠️ Could not read training data: {e}")
        
        if is_intent_classification:
            # For intent classification, always use accuracy-focused metrics
            description = "Intent classification prompt that categorizes user messages into predefined intents"
            function_type = "classification"
            metrics = ["accuracy", "clarity", "effectiveness", "consistency"]
            
            print(f"Function analysis: {description}")
            print(f"Function type: {function_type}")
            print(f"Selected metrics: {metrics}")
            
            return description, function_type, metrics
        
        print("Generating metrics for: ", tool_name)
        analysis_prompt = f"""Analyze this Python function and determine the most relevant evaluation metrics.

FUNCTION CODE:
{source_code}

{training_sample}

TASK: Provide a concise analysis and exactly 4 metrics for evaluating this function's output quality.

INSTRUCTIONS:
1. Analyze what this function does based on its name, code, and purpose
2. Determine if this is a QUANTITATIVE/ANALYTICAL function or a CREATIVE/CONTENT function
3. Choose exactly 4 metrics that are most relevant for evaluating this specific tool
4. For quantitative functions, focus on correctness, completeness, and logical soundness
5. For creative functions, focus on quality, relevance, and subjective measures

FUNCTION TYPE CLASSIFICATION:
- QUANTITATIVE/ANALYTICAL: Functions that perform calculations, data analysis, technical analysis, mathematical operations, statistical computations, financial analysis, etc.
- CREATIVE/CONTENT: Functions that generate text, creative content, marketing materials, essays, stories, etc.
- PROMPT/TEMPLATE: Pure prompt templates, system messages, instruction strings, or prompt engineering patterns without executable code.
- CLASSIFICATION: Intent classification, category classification, or classification tasks with expected outputs.

METRICS BY FUNCTION TYPE:
- Quantitative/Analytical functions: correctness, completeness, accuracy, consistency, precision, logical_soundness, data_coverage, calculation_validity
- Content/Creative functions: quality, relevance, engagement, authenticity, creativity, coherence, persuasiveness, clarity
- Prompt/Template functions: clarity, specificity, effectiveness, completeness, instruction_quality, prompt_structure, task_alignment, response_guidance
- Research tools: accuracy, depth, insight, completeness, relevance, comprehensiveness
- Classification: accuracy, precision, consistency, confidence
- Code generation: correctness, completeness, maintainability, readability

Return your response in this exact JSON format:
{{
    "function_description": "Brief description of what this function does",
    "function_type": "quantitative" or "creative" or "prompt" or "classification",
    "metrics": ["metric1", "metric2", "metric3", "metric4"]
}}

Be specific and choose metrics that truly matter for this function's output quality."""

        response = self.model.invoke([HumanMessage(content=analysis_prompt)])
        print(response.content)
        
        try:
            import re
            # Extract JSON from response
            content = response.content.strip()
            if "```json" in content:
                content = re.search(r'```json\s*([^`]+)```', content, re.DOTALL).group(1).strip()
            elif "```" in content:
                content = re.search(r'```\s*([^`]+)```', content, re.DOTALL).group(1).strip()
            
            analysis = json.loads(content)
            description = analysis.get("function_description", f"Function: {tool_name}")
            function_type = analysis.get("function_type", "creative").lower()
            metrics = analysis.get("metrics", ["quality", "relevance", "usefulness", "clarity"])
            
            print(f"Function analysis: {description}")
            print(f"Function type: {function_type}")
            print(f"Selected metrics: {metrics}")
            
            return description, function_type, metrics
            
        except Exception as e:
            print(f"Error parsing function analysis: {e}")
            print(f"Raw response: {response.content}")
            return f"Function: {tool_name}", "creative", ["quality", "relevance", "usefulness", "clarity"]

    def _generate_evaluator_file(self, tool_dir: Path, feedback: str = None):
        """Generate evaluator file for a single tool"""
        tool_name = tool_dir.name
        tool_file = tool_dir / "evolve_target.py"
        training_data_file = tool_dir / "training_data.json"
        evaluator_file = tool_dir / "evaluator.py"
        
        # Read tool source code
        source_code = self._get_source_code(tool_file)
        
        # Analyze function and get metrics
        function_description, function_type, metrics = self._analyze_function_and_metrics(tool_dir)
        
        # Read training data if it exists
        training_data = []
        if training_data_file.exists():
            training_data = json.load(open(training_data_file, 'r', encoding='utf-8'))

        prompt = {
            "role": "You are an expert in creating OpenEvolve-compatible evaluators.",
            "task": "Generate a Python evaluator file for OpenEvolve that evaluates the given tool's output quality.",
            "function_name": tool_name,
            "function_code": source_code,
            "training_data_examples": training_data[:3] if training_data else [],
            "instructions": (
                "Create a Python evaluator file that OpenEvolve can use to evaluate this tool.",
                "The evaluator must have EXACTLY this function signature: def evaluate(program) -> dict:",
                "The 'program' parameter is the file path to the tool's Python file.",
                "The function must return a dictionary with metric names as keys and float scores (0.0-1.0) as values.",
                "Choose evaluation metrics that are contextually relevant to what this specific tool does.",
                "When evaluating a tool that generates content like text or code, returning non deterministic LLM outputs, Use an LLM to evaluate the output quality. Generate a strict and objective prompt which is sent to an LLM to evaluate the output quality.",
                "CRITICAL: For JSON parsing from LLM responses, you MUST use robust error handling:",
                "- Strip whitespace and check for empty responses",
                "- Extract JSON from markdown code blocks if present", 
                "- Use try/except blocks around json.loads()",
                "- Fall back to default scores if JSON parsing fails",
                "- Log parsing errors for debugging",
                "The evaluator MUST load training_data.json from the same directory as the evaluator.",
                "The evaluator MUST call the tool function with EVERY test case from the training data.",
                "The evaluator MUST calculate metrics for each test case and return the AVERAGE across all test cases.",
                "Use the training data examples to understand expected inputs and outputs.",
                "Include proper error handling and logging.",
                "Make the evaluation logic specific to this tool's purpose, not generic.",
                "Log all LLM inputs and outputs to the logger.",
            ),
            "required_structure": (
                "The file must include:",
                "1. Standard imports: import os, json, logging, re, importlib.util, openai",
                "   CRITICAL: DO NOT USE LANGCHAIN. Use raw OpenAI API only.",
                "2. Logger initialization: logger = logging.getLogger(__name__)",
                "3. A robust JSON parsing function with this exact signature:",
                "   def parse_json_response(response_content: str, default_metrics: dict) -> dict:",
                "   - Strip whitespace and handle empty responses",
                "   - Extract JSON from ```json``` or ``` ``` markdown blocks using regex", 
                "   - Use try/except around json.loads() with detailed error logging",
                "   - Validate response is dict and clamp values to 0.0-1.0 range",
                "   - Return default_metrics on any parsing failure",
                "4. The evaluate(program) -> dict function with proper error handling",
                "5. Load training_data.json from os.path.dirname(os.path.abspath(__file__))",
                "6. Dynamic tool loading using importlib to load the program file",
                "7. Execute the tool function with EVERY input from training_data.json",
                "8. For each test case, use LLM evaluation then call parse_json_response(response.content, default_metrics)",
                "9. Return dictionary with AVERAGED metric scores between 0.0 and 1.0 across all test cases"
            ),
            "json_parsing_example": '''def parse_json_response(response_content: str, default_metrics: dict) -> dict:
    try:
        content = response_content.strip()
        if not content:
            return default_metrics
        if "```json" in content:
            content = re.search(r'```json\\s*([^`]+)```', content, re.DOTALL).group(1).strip()
        elif "```" in content:
            content = re.search(r'```\\s*([^`]+)```', content, re.DOTALL).group(1).strip()
        parsed = json.loads(content)
        if isinstance(parsed, dict):
            return {k: max(0.0, min(1.0, float(v))) for k, v in parsed.items() if k in default_metrics}
        return default_metrics
    except Exception as e:
        logger.error(f"JSON parse error: {e}")
        return default_metrics''',
            "output_format": "python",
            "output_instructions": "Return ONLY the complete Python code for the evaluator file. Include the exact parse_json_response function shown in json_parsing_example. No markdown, backticks, or explanations."
        }

        # Build the prompt with optional feedback
        feedback_section = ""
        if feedback:
            feedback_section = f"""
PREVIOUS ATTEMPT FEEDBACK:
The previous evaluator had issues. Please fix these specific problems:
{feedback}

CRITICAL: Address all the issues mentioned above in your new implementation.
"""

        # Create specialized prompt based on function type
        if function_type == "quantitative" or "analytic" in function_type:
            evaluation_prompt = self._create_quantitative_evaluator_prompt(
                tool_name, function_description, metrics, source_code, training_data, feedback_section
            )
        elif function_type == "prompt" or "template" in function_type or function_type == "classification":
            evaluation_prompt = self._create_prompt_evaluator_prompt(
                tool_name, function_description, metrics, source_code, training_data, feedback_section
            )
        else:
            evaluation_prompt = self._create_content_evaluator_prompt(
                tool_name, function_description, metrics, source_code, training_data, feedback_section
            )

        response = self.model.invoke([HumanMessage(content=evaluation_prompt)])
        print(f"LLM Response length: {len(response.content)}")
        print(f"Response starts with: {response.content[:100]}")
        print(f"Response ends with: {response.content[-100:]}")
        
        try:
            evaluator_code = response.content
            
            # Fix common import errors in generated code
            evaluator_code = self._fix_common_import_errors(evaluator_code)
            
            # Validate the generated evaluator code
            validation_result = self._validate_evaluator_with_llm(evaluator_code)
            if not validation_result['is_valid']:
                print(f"ERROR: Generated evaluator code is invalid: {validation_result['issues']}")
                if validation_result['suggested_fix']:
                    print("Attempting to apply LLM-suggested fix...")
                    evaluator_code = validation_result['suggested_fix']
                else:
                    return None
            
            # SIMPLE cleanup - only remove markdown blocks, don't do aggressive text removal
            original_code = evaluator_code
            
            # Check if code already looks complete - if so, skip cleanup entirely
            if ('def evaluate(' in evaluator_code and 
                'return scores' in evaluator_code and 
                len(evaluator_code) > 3000):
                print("Code appears complete - skipping cleanup")
            else:
                # Only remove markdown blocks if present, nothing else
                if "```python" in evaluator_code:
                    start = evaluator_code.find('```python') + len('```python')
                    end = evaluator_code.rfind('```')
                    if end != -1 and end > start:
                        evaluator_code = evaluator_code[start:end].strip()
                    else:
                        evaluator_code = evaluator_code[start:].strip()
                elif "```" in evaluator_code:
                    start = evaluator_code.find('```') + 3
                    end = evaluator_code.rfind('```')
                    if end != -1 and end > start:
                        evaluator_code = evaluator_code[start:end].strip()
                    else:
                        evaluator_code = evaluator_code[start:].strip()
            
            if len(evaluator_code) != len(original_code):
                print(f"Cleaned code length: {len(evaluator_code)}")
            
            # Validate the code has essential components
            # This should not happen due to earlier validation, but just in case
            if 'def evaluate(' not in evaluator_code:
                print("ERROR: Generated code missing 'def evaluate(' function")
                return None
            
            if 'return' not in evaluator_code:
                print("ERROR: Generated code missing return statement")
                return None
            
            with open(evaluator_file, 'w', encoding='utf-8') as f:
                f.write(evaluator_code)
            print(f"Evaluator saved to {evaluator_file} ({len(evaluator_code)} chars)")
            return evaluator_code
        except Exception as e:
            print(f"Error saving evaluator: {e}")
            print(f"Raw response: {response.content}")
            return None


        
        


    
    def _generate_single_evaluator(self, tool_dir: Path):
        """Generate an evaluator for a single tool"""
        tool_name = tool_dir.name
        tool_file = tool_dir / "evolve_target.py"
        training_data_file = tool_dir / "training_data.json"
        metadata_file = tool_dir / "metadata.json"
        evaluator_file = tool_dir / "evaluator.py"
        
        # Check if training data exists
        if not training_data_file.exists():
            print(f"⚠️  Warning: No training data found at {training_data_file}")
            print(f"   Please run: python generate_training_data.py {tool_dir.parent}")
            return
        
        # Check if evaluator already exists and is valid - STRONG CHECK
        if evaluator_file.exists():
            try:
                with open(evaluator_file, 'r', encoding='utf-8') as f:
                    existing_code = f.read()
                # Strong validation - check for key components
                if ('def evaluate(' in existing_code and 
                    'return scores' in existing_code and 
                    'EVALUATION_METRICS' in existing_code and
                    len(existing_code) > 3000):
                    print(f"✅ Evaluator already exists and appears valid - STRONGLY skipping generation")
                    print(f"   File size: {len(existing_code)} characters")
                    print(f"   Contains evaluate function: {'def evaluate(' in existing_code}")
                    return
                else:
                    print(f"⚠️ Evaluator exists but seems incomplete - will regenerate")
            except Exception as e:
                print(f"⚠️ Error reading existing evaluator: {e}")
                pass
        
        # Read tool source code
        try:
            with open(tool_file, 'r', encoding='utf-8') as f:
                tool_source = f.read()
        except Exception as e:
            print(f"Error reading tool file: {e}")
            return
        
        # Read metadata if available
        metadata = {}
        if metadata_file.exists():
            try:
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
            except Exception as e:
                print(f"Warning: Could not read metadata: {e}")
        
        # Generate evaluator using existing training data
        self._generate_and_validate_evaluator(tool_dir)

    def _analyze_evaluator(self, evaluator_code: str, tool_dir: Path) -> Dict[str, Any]:
        """Analyze generated evaluator code for issues"""
        import ast
        import re
        
        tool_name = tool_dir.name
        tool_file = tool_dir / "evolve_target.py"
        training_data_file = tool_dir / "training_data.json"
        
        issues = []
        fixes = []
        
        # Read tool source to understand the main function
        try:
            with open(tool_file, 'r') as f:
                tool_source = f.read()
        except:
            return {"issues": ["Could not read tool source"], "fixes": []}
        
        # Read training data to understand expected inputs
        try:
            with open(training_data_file, 'r') as f:
                training_data = json.load(f)
                sample_input = training_data[0] if training_data else {}
        except:
            return {"issues": ["Could not read training data"], "fixes": []}
        
        # Find the main function name from tool source
        try:
            tree = ast.parse(tool_source)
            main_functions = []
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    # Skip helper functions, find the main tool function
                    if not node.name.startswith('_') and node.name not in ['main', 'test']:
                        main_functions.append(node.name)
            
            # The main function is likely the one matching the tool name or the longest one
            main_function = None
            if main_functions:
                # Try to find function matching tool name pattern
                for func in main_functions:
                    if tool_name.replace('_', '').lower() in func.replace('_', '').lower():
                        main_function = func
                        break
                
                # Fallback to first non-helper function
                if not main_function:
                    main_function = main_functions[0]
        except:
            main_function = None
        
        print(f"  Detected main function: {main_function}")
        
        # Check 1: Wrong function being called (getattr pattern)
        if main_function:
            if f"getattr(tool_module, '{main_function}')" not in evaluator_code:
                # Find what function is being called with getattr
                getattr_matches = re.findall(r"getattr\(tool_module,\s*['\"]([^'\"]+)['\"]", evaluator_code)
                if getattr_matches:
                    wrong_function = getattr_matches[0]
                    issues.append(f"Wrong function called: '{wrong_function}' instead of '{main_function}'")
                    fixes.append(f"Replace 'getattr(tool_module, '{wrong_function}')' with 'getattr(tool_module, '{main_function}')'")
                else:
                    issues.append(f"Could not find getattr call for main function '{main_function}'")
                    fixes.append(f"Add 'tool_function = getattr(tool_module, '{main_function}')'")
        
        # Check 1b: Wrong function being called (direct call pattern)
        if main_function:
            # Check for direct tool_module.function_name() calls
            direct_call_pattern = r"tool_module\.([a-zA-Z_][a-zA-Z0-9_]*)\s*\("
            direct_matches = re.findall(direct_call_pattern, evaluator_code)
            if direct_matches:
                for called_function in direct_matches:
                    if called_function != main_function:
                        issues.append(f"Direct wrong function call: 'tool_module.{called_function}()' instead of 'tool_module.{main_function}()'")
                        fixes.append(f"Replace 'tool_module.{called_function}(' with 'tool_module.{main_function}('")
        
        # Check 2: Function parameter compatibility
        if main_function and sample_input:
            # Extract function signature from tool source
            try:
                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef) and node.name == main_function:
                        expected_params = [arg.arg for arg in node.args.args]
                        provided_params = list(sample_input.keys())
                        
                        # Check if parameters match
                        if set(expected_params) != set(provided_params):
                            issues.append(f"Parameter mismatch: function expects {expected_params}, training data provides {provided_params}")
                            
                            # If training data is correct, suggest using **input_data
                            if all(param in provided_params for param in expected_params):
                                fixes.append(f"Use 'result = tool_function(**input_data)' instead of individual parameters")
                        break
            except:
                pass
        
        # Check 3: Training data structure compatibility
        if "training_data[" in evaluator_code and isinstance(training_data, list):
            # Check if evaluator assumes wrong data structure
            if "brand_research" in evaluator_code:
                if not any("brand_research" in str(item) for item in training_data):
                    issues.append("Evaluator assumes training_data has 'brand_research' key but it doesn't")
                    fixes.append("Remove 'brand_research' key assumption, use training_data directly as list")
        
        # Check 4: EVALUATION_METRICS consistency
        if "EVALUATION_METRICS" in evaluator_code:
            # Extract metrics from evaluator
            metrics_match = re.search(r'EVALUATION_METRICS\s*=\s*\[([^\]]+)\]', evaluator_code)
            if metrics_match:
                metrics_content = metrics_match.group(1)
                evaluator_metrics = re.findall(r"'([^']+)'", metrics_content)
                
                # Check if metrics are used consistently in evaluation prompt
                for metric in evaluator_metrics:
                    if f"- {metric}" not in evaluator_code:
                        issues.append(f"Metric '{metric}' defined but not used in evaluation prompt")
                        fixes.append(f"Add '- {metric}' to evaluation prompt metrics list")
        
        # Check 5: Import issues
        required_imports = ["os", "json", "importlib.util", "re", "langchain_openai", "langchain_core.messages"]
        for imp in required_imports:
            if imp not in evaluator_code:
                issues.append(f"Missing import: {imp}")
                fixes.append(f"Add 'import {imp}' or 'from {imp} import ...'")
        
        # Check 6: Error handling for tool execution
        has_try_except = re.search(r'\btry\s*:', evaluator_code) and re.search(r'\bexcept\b', evaluator_code)
        # Check if the try block contains the tool function call
        has_tool_error_handling = False
        if has_try_except:
            # Look for tool function calls inside try blocks
            if f"tool_module.{main_function}" in evaluator_code or "tool_function(" in evaluator_code:
                # Check if there's a try block around tool calls
                try_sections = evaluator_code.split("try:")
                for section in try_sections[1:]:  # Skip first split (before any try)
                    except_pos = section.find("except")
                    if except_pos != -1:
                        try_content = section[:except_pos]
                        if f"tool_module.{main_function}" in try_content or "tool_function(" in try_content:
                            has_tool_error_handling = True
                            break
        
        if not has_tool_error_handling:
            issues.append("No error handling for tool execution")
            fixes.append("Add try-except block around tool function call")
        
        return {
            "issues": issues,
            "fixes": fixes,
            "main_function": main_function,
            "training_data_structure": type(training_data).__name__,
            "sample_input": sample_input
        }

    def _fix_evaluator_code(self, evaluator_code: str, analysis: Dict[str, Any]) -> str:
        """Apply fixes to evaluator code based on analysis"""
        fixed_code = evaluator_code
        
        # Apply each fix
        for i, fix in enumerate(analysis["fixes"]):
            print(f"    Applying fix {i+1}: {fix}")
            
            # Fix 1: Wrong function name (getattr pattern)
            if "Replace 'getattr(tool_module," in fix:
                # Extract old and new function names from fix description
                match = re.search(r"Replace 'getattr\(tool_module, '([^']+)'\)' with 'getattr\(tool_module, '([^']+)'\)'", fix)
                if match:
                    old_func, new_func = match.groups()
                    fixed_code = fixed_code.replace(f"getattr(tool_module, '{old_func}')", f"getattr(tool_module, '{new_func}')")
            
            # Fix 1b: Wrong function name (direct call pattern)
            elif "Replace 'tool_module." in fix and "(' with 'tool_module." in fix:
                # Extract old and new function calls from fix description
                match = re.search(r"Replace 'tool_module\.([^']+)\(' with 'tool_module\.([^']+)\('", fix)
                if match:
                    old_func, new_func = match.groups()
                    fixed_code = fixed_code.replace(f"tool_module.{old_func}(", f"tool_module.{new_func}(")
            
            # Fix 2: Parameter usage
            elif "Use 'result = tool_function(**input_data)'" in fix:
                # Replace direct parameter calls with **input_data
                fixed_code = re.sub(r'result = tool_function\([^)]+\)', 'result = tool_function(**input_data)', fixed_code)
            
            # Fix 3: Training data structure
            elif "Remove 'brand_research' key assumption" in fix:
                # Fix training data access pattern
                fixed_code = fixed_code.replace('training_data["brand_research"]', 'training_data')
                fixed_code = fixed_code.replace('for input_data in training_data["brand_research"]:', 'for input_data in training_data:')
            
            # Fix 4: Add missing metrics to evaluation prompt
            elif "Add '-" in fix and "to evaluation prompt" in fix:
                metric_match = re.search(r"Add '- ([^']+)' to evaluation prompt", fix)
                if metric_match:
                    metric = metric_match.group(1)
                    # Find the evaluation prompt and add the metric
                    prompt_section = re.search(r'(Rate on these EXACT metrics.*?):([^"]*)', fixed_code, re.DOTALL)
                    if prompt_section:
                        existing_metrics = prompt_section.group(2)
                        if f"- {metric}" not in existing_metrics:
                            new_metrics = existing_metrics.rstrip() + f"\n- {metric}"
                            fixed_code = fixed_code.replace(prompt_section.group(2), new_metrics)
            
            # Fix 5: Add missing imports
            elif "Add 'import" in fix:
                import_match = re.search(r"Add '(import [^']+)'", fix)
                if import_match:
                    import_stmt = import_match.group(1)
                    if import_stmt not in fixed_code:
                        # Add import after existing imports
                        import_section = fixed_code.split('\n')
                        insert_pos = 0
                        for i, line in enumerate(import_section):
                            if line.startswith('import ') or line.startswith('from '):
                                insert_pos = i + 1
                        import_section.insert(insert_pos, import_stmt)
                        fixed_code = '\n'.join(import_section)
            
            # Fix 6: Add error handling
            elif "Add try-except block" in fix:
                # Wrap tool function call in try-except if not already present
                if "try:" not in fixed_code:
                    tool_call_pattern = r'(\s+)(result = tool_function\([^)]+\))'
                    replacement = r'\1try:\n\1    \2\n\1except Exception as e:\n\1    print(f"Error calling tool function: {e}")\n\1    continue'
                    fixed_code = re.sub(tool_call_pattern, replacement, fixed_code)
        
        return fixed_code

    def _generate_feedback_for_llm(self, analysis: Dict[str, Any]) -> str:
        """Convert analysis issues into structured feedback for the LLM"""
        if not analysis["issues"]:
            return ""
        
        feedback_parts = []
        
        # Group issues by type for better feedback
        function_issues = []
        parameter_issues = []
        data_issues = []
        metrics_issues = []
        import_issues = []
        other_issues = []
        
        for issue in analysis["issues"]:
            if "Wrong function called" in issue or "Could not find getattr call" in issue:
                function_issues.append(issue)
            elif "Parameter mismatch" in issue:
                parameter_issues.append(issue)
            elif "training_data" in issue.lower() or "brand_research" in issue:
                data_issues.append(issue)
            elif "Metric" in issue and ("defined but not used" in issue):
                metrics_issues.append(issue)
            elif "Missing import" in issue:
                import_issues.append(issue)
            else:
                other_issues.append(issue)
        
        # Generate specific feedback sections
        if function_issues:
            feedback_parts.append(f"""
FUNCTION CALL ISSUES:
{chr(10).join([f"- {issue}" for issue in function_issues])}
FIX: You must call the main tool function that matches the tool name '{analysis.get('main_function', 'unknown')}', not helper functions.
Use: tool_function = getattr(tool_module, '{analysis.get('main_function', 'unknown')}')""")
        
        if parameter_issues:
            feedback_parts.append(f"""
PARAMETER ISSUES:
{chr(10).join([f"- {issue}" for issue in parameter_issues])}
FIX: Use **input_data to unpack parameters: result = tool_function(**input_data)
Training data structure: {analysis.get('training_data_structure', 'unknown')}
Sample input: {analysis.get('sample_input', {})}""")
        
        if data_issues:
            feedback_parts.append(f"""
DATA STRUCTURE ISSUES:
{chr(10).join([f"- {issue}" for issue in data_issues])}
FIX: Training data is a {analysis.get('training_data_structure', 'list')}. Iterate directly over training_data, not nested keys.""")
        
        if metrics_issues:
            feedback_parts.append(f"""
METRICS CONSISTENCY ISSUES:
{chr(10).join([f"- {issue}" for issue in metrics_issues])}
FIX: Ensure all metrics in EVALUATION_METRICS are listed in the evaluation prompt.""")
        
        if import_issues:
            feedback_parts.append(f"""
IMPORT ISSUES:
{chr(10).join([f"- {issue}" for issue in import_issues])}
FIX: Add all required imports at the top of the file.""")
        
        if other_issues:
            feedback_parts.append(f"""
OTHER ISSUES:
{chr(10).join([f"- {issue}" for issue in other_issues])}""")
        
        return "\n".join(feedback_parts)

    def _generate_and_validate_evaluator(self, tool_dir: Path, max_attempts: int = 3):
        """Generate evaluator and validate it recursively until it's correct"""
        tool_name = tool_dir.name
        print(f"  Generating and validating evaluator for: {tool_name}")
        
        feedback = None
        
        for attempt in range(1, max_attempts + 1):
            print(f"    Attempt {attempt}/{max_attempts}")
            
            # Generate evaluator with feedback from previous attempt
            evaluator_code = self._generate_evaluator_file(tool_dir, feedback)
            if not evaluator_code:
                print(f"    ❌ Failed to generate evaluator code")
                continue
            
            # Check if this is a prompt template - skip function-based validation
            metadata_file = tool_dir / "metadata.json"
            is_prompt_template = False
            if metadata_file.exists():
                try:
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)
                        if metadata.get("category") == "prompt_optimization":
                            is_prompt_template = True
                except:
                    pass
            
            if is_prompt_template:
                print(f"    ✅ Prompt template evaluator generated successfully!")
                return True
            
            # Analyze the generated evaluator (only for functions)
            analysis = self._analyze_evaluator(evaluator_code, tool_dir)
            
            if not analysis["issues"]:
                print(f"    ✅ Evaluator is correct!")
                return True
            
            print(f"    ⚠️  Found {len(analysis['issues'])} issues:")
            for i, issue in enumerate(analysis["issues"], 1):
                print(f"      {i}. {issue}")
            
            # If this is the last attempt, save as-is with warnings
            if attempt == max_attempts:
                print(f"    ⚠️  Max attempts reached. Saving evaluator with known issues.")
                return False
            
            # Generate structured feedback for the next LLM attempt
            feedback = self._generate_feedback_for_llm(analysis)
            print(f"    🔄 Generating feedback for next attempt...")
            
            # Don't apply mechanical fixes - let the LLM handle it with feedback
            print(f"    🤖 Will regenerate with LLM feedback...")
        
        return False

    def _create_quantitative_evaluator_prompt(self, tool_name: str, function_description: str, 
                                             metrics: list, source_code: str, training_data: list, 
                                             feedback_section: str) -> str:
        """Create specialized prompt for quantitative/analytical functions"""
        return f"""Create a Python evaluator for OpenEvolve algorithmic tool: {tool_name}

FUNCTION ANALYSIS:
{function_description}

FUNCTION TYPE: ALGORITHMIC/QUANTITATIVE
This function performs algorithmic calculations, data analysis, or technical computations. 
The evaluator should analyze THE FUNCTION IMPLEMENTATION ITSELF, not just outputs.

PRE-DETERMINED METRICS (use these exactly):
{metrics}
{feedback_section}

ALGORITHMIC EVALUATION APPROACH:
For algorithmic functions, evaluation should examine:
1. IMPLEMENTATION CORRECTNESS: Is the algorithm logic sound?
2. INPUT HANDLING: Are inputs validated and processed correctly?
3. CALCULATION ACCURACY: Are mathematical operations implemented properly?
4. OUTPUT COMPLETENESS: Does output contain all expected components?
5. ERROR HANDLING: Are edge cases and errors handled appropriately?

CRITICAL REQUIREMENTS:
1. Function signature: def evaluate(program) -> dict:
2. Load training data: os.path.join(os.path.dirname(os.path.abspath(__file__)), 'training_data.json')
3. Import tool: importlib.util.spec_from_file_location("tool_module", program)
4. READ AND ANALYZE THE SOURCE CODE ITSELF using inspect.getsource()
5. Call tool with inputs AND analyze the implementation logic
6. Use LLM to evaluate algorithmic correctness, implementation quality

COMPLETE FUNCTION SOURCE CODE:
```python
{source_code}
```

TRAINING DATA STRUCTURE:
{json.dumps(training_data[0], indent=2) if training_data else {{}}}

ALGORITHMIC EVALUATION WORKFLOW:
For each training case:
1. Extract and analyze the function source code using inspect.getsource()
2. Call the function with test inputs
3. Analyze both the IMPLEMENTATION and the OUTPUT
4. Create comprehensive evaluation examining:
   - Algorithm logic and correctness
   - Input parameter handling
   - Calculation steps and formulas
   - Output structure and completeness
   - Error handling and edge cases
5. Parse LLM response with robust JSON parser
6. Accumulate scores and return averages

EVALUATION PROMPT TEMPLATE:
```python
# Get the actual function source code for analysis
function_source = inspect.getsource(getattr(tool_module, '{tool_name}'))

eval_prompt = f\\\"\\\"\\\"You are evaluating an ALGORITHMIC/QUANTITATIVE function.
Analyze BOTH the implementation code AND the output for correctness.

FUNCTION TO EVALUATE: {tool_name}
INPUT PARAMETERS: {{input_data}}
FUNCTION SOURCE CODE:
{{function_source}}

GENERATED OUTPUT: {{generated_content}}

ALGORITHMIC ANALYSIS CRITERIA:
1. IMPLEMENTATION REVIEW:
   - Is the algorithm logic mathematically sound?
   - Are calculations implemented correctly?
   - Are formulas and computations accurate?
   - Is input validation adequate?

2. OUTPUT ANALYSIS:
   - Does output match expected format/structure?
   - Are calculated values reasonable for given inputs?
   - Is the output complete with all expected fields?

3. CODE QUALITY:
   - Are there logical errors in the implementation?
   - Is error handling appropriate?
   - Are edge cases considered?

Evaluate on these EXACT metrics (0.0-1.0 scale):
{chr(10).join([f"- {metric}: Assess this aspect of both code implementation and output" for metric in metrics])}

SCORING GUIDELINES FOR ALGORITHMIC FUNCTIONS:
- 0.0-0.3: Major algorithmic errors, incorrect implementation, or broken logic
- 0.4-0.6: Generally correct algorithm but minor implementation issues or incomplete handling
- 0.7-0.8: Well-implemented algorithm with correct logic and proper output
- 0.9-1.0: Perfect algorithmic implementation with comprehensive error handling

Focus on the IMPLEMENTATION QUALITY as much as the output correctness.
Return ONLY JSON with exact metric names.\\\"\\\"\\\"
```

REQUIRED IMPORTS AND STRUCTURE:
```python
import os
import json
import importlib.util
import inspect  # CRITICAL: For analyzing function source code
import re
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
import logging

logger = logging.getLogger(__name__)
EVALUATION_METRICS = {metrics}

def parse_json_response(response_content: str) -> dict:
    try:
        content = response_content.strip()
        if not content:
            return {{metric: 0.0 for metric in EVALUATION_METRICS}}
        
        # Extract JSON from markdown blocks
        if "```json" in content:
            content = re.search(r'```json\\s*([^`]+)```', content, re.DOTALL).group(1).strip()
        elif "```" in content:
            content = re.search(r'```\\s*([^`]+)```', content, re.DOTALL).group(1).strip()
        
        # Parse JSON
        parsed = json.loads(content)
        if isinstance(parsed, dict):
            # Clamp all values to 0.0-1.0 range and return metrics that exist in EVALUATION_METRICS
            result = {{}}
            for metric in EVALUATION_METRICS:
                if metric in parsed:
                    result[metric] = max(0.0, min(1.0, float(parsed[metric])))
                else:
                    result[metric] = 0.0
            return result
        
        return {{metric: 0.0 for metric in EVALUATION_METRICS}}
    except Exception as e:
        logger.error(f"JSON parsing error: {{e}}")
        logger.error(f"Raw response: {{response_content}}")
        return {{metric: 0.0 for metric in EVALUATION_METRICS}}
    
def evaluate(program) -> dict:
    # Load training data
    training_data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'training_data.json')
    with open(training_data_path, 'r') as f:
        training_data = json.load(f)
    
    # Import tool module
    spec = importlib.util.spec_from_file_location("tool_module", program)
    tool_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tool_module)
    
    # Get tool function
    tool_function = getattr(tool_module, '{tool_name}')
    
    # CRITICAL: Extract source code for algorithm analysis
    function_source = inspect.getsource(tool_function)
    
    # Initialize LLM
    model = ChatOpenAI(model="gpt-4o", temperature=0.0)
    
    # Process each test case with BOTH code analysis and output evaluation
    # [Continue with implementation that analyzes algorithm AND tests outputs]
```

IMPORTANT: This evaluator must analyze the FUNCTION IMPLEMENTATION ITSELF using inspect.getsource(), 
not just test the outputs. The LLM should review the actual algorithm code for correctness.

CRITICAL OUTPUT REQUIREMENT:
Return ONLY valid Python code. No explanations, no markdown code blocks, no text before or after.
Start with 'import os' and end with the final closing brace of the evaluate function.
DO NOT include ```python or ``` or any other markdown formatting.
DO NOT include any explanatory text before or after the code."""

    def _create_prompt_evaluator_prompt(self, tool_name: str, function_description: str, 
                                      metrics: list, source_code: str, training_data: list, 
                                      feedback_section: str) -> str:
        """Create specialized prompt for prompt/template optimization"""
        return f"""Create a Python evaluator for OpenEvolve prompt optimization tool: {tool_name}

FUNCTION ANALYSIS:
{function_description}

FUNCTION TYPE: PROMPT/TEMPLATE
This is a prompt template for LLM optimization. The evaluator must test the prompt by:
1. Formatting the prompt with training data parameters
2. Sending the formatted prompt to an LLM
3. Evaluating the LLM's response quality and prompt effectiveness

PRE-DETERMINED METRICS (use these exactly):
{metrics}

SPECIAL HANDLING FOR INTENT CLASSIFICATION:
If this is an intent classification prompt (determines user intent categories), 
focus primarily on ACCURACY - how well the prompt guides the LLM to classify correctly.
Compare the LLM output directly with expected_intent for exact match scoring.
{feedback_section}

CRITICAL REQUIREMENTS FOR PROMPT EVALUATION:
1. Function signature: def evaluate(program) -> dict:
2. Load training data: os.path.join(os.path.dirname(os.path.abspath(__file__)), 'training_data.json')  
3. EXTRACT PROMPT TEMPLATE from the evolve_target.py file
4. FORMAT prompt with training data parameters (use .format() or f-strings)
5. SEND formatted prompt to LLM to get response
6. EVALUATE the LLM response quality with a separate evaluation prompt

PROMPT EXTRACTION AND TESTING WORKFLOW:
```python
# 1. Extract the prompt template from the file
with open(program, 'r') as f:
    file_content = f.read()

# Find the prompt constant (look for UPPERCASE_VARIABLE = \"\"\"...\"\"\")
import re
prompt_match = re.search(r'([A-Z_]+)\\\\s*=\\\\s*\"\"\"(.*?)\"\"\"', file_content, re.DOTALL)
if prompt_match:
    prompt_template = prompt_match.group(2).strip()

# 2. For each training case, format and test the prompt
for test_case in training_data:
    # Format the prompt with test parameters
    try:
        formatted_prompt = prompt_template.format(**test_case)
    except KeyError:
        # Handle prompts that don't use .format() - just use as-is
        formatted_prompt = prompt_template
    
    # Send to LLM
    response = model.invoke([HumanMessage(content=formatted_prompt)])
    llm_output = response.content
    
    # Evaluate the output quality
    eval_prompt = f\"\"\"Evaluate this prompt and its output...\"\"\"
```

TRAINING DATA STRUCTURE:
Training data should contain parameters that the prompt accepts, NOT input/output pairs.
Expected format: [{{"param1": "value1", "param2": "value2", ...}}]
Example: {json.dumps(training_data[0], indent=2) if training_data else {{}}}

EVALUATION APPROACH:
For each test case:
1. Extract prompt template from evolve_target.py
2. Format prompt with test case parameters: formatted_prompt = template.format(**test_case)
3. Send formatted prompt to LLM: response = model.invoke([HumanMessage(content=formatted_prompt)])
4. Create evaluation prompt to assess both prompt quality AND LLM output
5. Get evaluation scores and parse JSON response

EVALUATION PROMPT TEMPLATE:
```python
eval_prompt = f\\\"\\\"\\\"You are evaluating a PROMPT TEMPLATE and the output it generates.

ORIGINAL PROMPT TEMPLATE:
{{prompt_template}}

TEST PARAMETERS: {{test_case}}
FORMATTED PROMPT SENT TO LLM:
{{formatted_prompt}}

LLM RESPONSE TO THE PROMPT:
{{llm_output}}

EVALUATION CRITERIA:
1. PROMPT QUALITY:
   - Is the prompt clear and well-structured?
   - Does it provide adequate context and instructions?
   - Is the language precise and unambiguous?

2. OUTPUT EFFECTIVENESS:
   - Does the LLM response align with the prompt's intent?
   - Is the output relevant and appropriate?
   - Does the prompt successfully guide the LLM behavior?

3. PROMPT ENGINEERING:
   - Does the prompt handle the input parameters well?
   - Is the prompt format optimal for LLM understanding?
   - Could the prompt be improved for better results?

Rate on these EXACT metrics (0.0-1.0 scale):
{chr(10).join([f"- {metric}: Evaluate prompt effectiveness for this aspect" for metric in metrics])}

SCORING GUIDELINES:
- 0.0-0.3: Poor prompt design, unclear instructions, ineffective output
- 0.4-0.6: Adequate prompt but room for improvement
- 0.7-0.8: Well-crafted prompt producing good results  
- 0.9-1.0: Exceptional prompt engineering with optimal output

Return ONLY JSON with exact metric names.\\\"\\\"\\\"
```

COMPLETE EVALUATOR STRUCTURE:
```python
import os
import json
import re
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
import logging

logger = logging.getLogger(__name__)
EVALUATION_METRICS = {metrics}

def parse_json_response(response_content: str) -> dict:
    try:
        content = response_content.strip()
        if not content:
            return {{metric: 0.0 for metric in EVALUATION_METRICS}}
        
        # Extract JSON from markdown blocks
        if "```json" in content:
            content = re.search(r'```json\\s*([^`]+)```', content, re.DOTALL).group(1).strip()
        elif "```" in content:
            content = re.search(r'```\\s*([^`]+)```', content, re.DOTALL).group(1).strip()
            
        parsed = json.loads(content)
        if isinstance(parsed, dict):
            result = {{}}
            for metric in EVALUATION_METRICS:
                if metric in parsed:
                    result[metric] = max(0.0, min(1.0, float(parsed[metric])))
                else:
                    result[metric] = 0.0
            return result
        
        return {{metric: 0.0 for metric in EVALUATION_METRICS}}
    except Exception as e:
        logger.error(f"JSON parsing error: {{e}}")
        return {{metric: 0.0 for metric in EVALUATION_METRICS}}

def evaluate(program) -> dict:
    # Load training data  
    training_data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'training_data.json')
    with open(training_data_path, 'r') as f:
        training_data = json.load(f)
    
    # Extract prompt template from file
    with open(program, 'r') as f:
        file_content = f.read()
    
    # Find prompt constant
    prompt_match = re.search(r'([A-Z_]+)\\\\s*=\\\\s*\"\"\"(.*?)\"\"\"', file_content, re.DOTALL)
    if not prompt_match:
        logger.error("Could not find prompt template in file")
        return {{metric: 0.0 for metric in EVALUATION_METRICS}}
    
    prompt_template = prompt_match.group(2).strip()
    
    # Initialize LLM
    model = ChatOpenAI(model="gpt-4o", temperature=0.0)
    
    # Evaluate prompt with each test case
    scores = {{metric: 0.0 for metric in EVALUATION_METRICS}}
    num_cases = len(training_data)
    
    for test_case in training_data:
        # Format prompt with parameters
        try:
            formatted_prompt = prompt_template.format(**test_case)
        except:
            formatted_prompt = prompt_template
            
        # Send to LLM
        response = model.invoke([HumanMessage(content=formatted_prompt)])
        llm_output = response.content
        
        # Create evaluation prompt
        eval_prompt = f\"\"\"You are evaluating a PROMPT TEMPLATE and the output it generates.

ORIGINAL PROMPT TEMPLATE:
{{prompt_template}}

TEST PARAMETERS: {{test_case}}
FORMATTED PROMPT SENT TO LLM:
{{formatted_prompt}}

LLM RESPONSE TO THE PROMPT:
{{llm_output}}

EVALUATION CRITERIA FOR INTENT CLASSIFICATION:

1. ACCURACY ASSESSMENT:
   - Does the LLM output match the expected_intent exactly?
   - How reliably does the prompt produce correct classifications?
   - Are the classification instructions clear enough for consistent results?

2. PROMPT QUALITY (Secondary):
   - Is the prompt clear and well-structured?
   - Does it provide adequate context and examples?
   - Are the intent categories clearly defined?

EVALUATION APPROACH:
For intent classification, calculate accuracy by comparing LLM output with expected_intent:
- If LLM output == expected_intent: Contribute 1.0 to accuracy
- If LLM output != expected_intent: Contribute 0.0 to accuracy
- Average across all test cases for final accuracy score

Rate on these EXACT metrics (0.0-1.0 scale):
- accuracy: Primary metric - exact match between LLM output and expected_intent
- clarity: Secondary metric - how clear are the prompt instructions
- effectiveness: Secondary metric - overall prompt performance  
- consistency: Secondary metric - how consistently the prompt performs

Return ONLY JSON with exact metric names.\"\"\"
        
        # Get evaluation
        eval_response = model.invoke([HumanMessage(content=eval_prompt)])
        case_scores = parse_json_response(eval_response.content)
        
        # Accumulate scores
        for metric in EVALUATION_METRICS:
            scores[metric] += case_scores[metric]
    
    # Average scores
    for metric in EVALUATION_METRICS:
        scores[metric] /= num_cases
        
    return scores
```

IMPORTANT: This evaluator FORMATS the prompt with parameters, SENDS it to an LLM, 
and EVALUATES the response quality. It does NOT execute functions.

CRITICAL OUTPUT REQUIREMENT:
Return ONLY valid Python code. No explanations, no markdown code blocks, no text before or after.
Start with 'import os' and end with the final closing brace of the evaluate function.
DO NOT include ```python or ``` or any other markdown formatting.
DO NOT include any explanatory text before or after the code."""

    def _create_content_evaluator_prompt(self, tool_name: str, function_description: str, 
                                       metrics: list, source_code: str, training_data: list, 
                                       feedback_section: str) -> str:
        """Create specialized prompt for creative/content functions"""
        return f"""Create a Python evaluator for OpenEvolve content tool: {tool_name}

FUNCTION ANALYSIS:
{function_description}

FUNCTION TYPE: CREATIVE/CONTENT
This function generates creative content, text, or subjective materials. The evaluator should focus on:
- Quality and coherence of generated content
- Relevance to input requirements  
- Engagement and effectiveness
- Authenticity and originality

PRE-DETERMINED METRICS (use these exactly):
{metrics}
{feedback_section}

CONTENT EVALUATION APPROACH:
For content functions, evaluation should assess:
1. QUALITY: Is the content well-written and coherent?
2. RELEVANCE: Does it meet the specified requirements?
3. ENGAGEMENT: Is it compelling and interesting?
4. AUTHENTICITY: Does it feel genuine and original?

CRITICAL REQUIREMENTS:
1. Function signature: def evaluate(program) -> dict:
2. Load training data: os.path.join(os.path.dirname(os.path.abspath(__file__)), 'training_data.json')
3. Import tool: importlib.util.spec_from_file_location("tool_module", program)
4. Call tool with CORRECT function name: {tool_name}
5. Use LLM to evaluate content quality subjectively

TOOL FUNCTION SIGNATURE:
{source_code[source_code.find('def '):source_code.find(':', source_code.find('def '))+1] if 'def ' in source_code else 'def unknown_function():'}

TRAINING DATA STRUCTURE:
{json.dumps(training_data[0], indent=2) if training_data else {{}}}

CONTENT EVALUATION WORKFLOW:
For each training case:
1. Call the main tool function with correct parameters
2. Handle return value (may be tuple - extract content)
3. Create SUBJECTIVE EVALUATION prompt focusing on content quality
4. Parse LLM response with robust JSON parser
5. Accumulate scores and return averages

EVALUATION PROMPT TEMPLATE:
```python
eval_prompt = f\\\"\\\"\\\"You are a STRICT content evaluator. Be critical and harsh in scoring.

Evaluate this generated content on 0.0-1.0 scale:

Generated Content: {{generated_content}}
Original Request: {{input_data}}

Rate on these EXACT metrics:
{chr(10).join([f"- {metric}" for metric in metrics])}

Scoring Guidelines for Content:
- 0.0-0.3: Poor quality, irrelevant, or unusable content
- 0.4-0.6: Mediocre content (most outputs should score here)  
- 0.7-0.8: Good quality, relevant, engaging content
- 0.9-1.0: Exceptional content (very rare)

Be harsh. Most content is mediocre. Return ONLY JSON.\\\"\\\"\\\"
```

CRITICAL OUTPUT REQUIREMENT:
Return ONLY valid Python code. No explanations, no markdown code blocks, no text before or after.
Start with 'import os' and end with the final closing brace of the evaluate function.
DO NOT include ```python or ``` or any other markdown formatting.
DO NOT include any explanatory text before or after the code."""


def main():
    """Main entry point"""
    if len(sys.argv) > 2:
        print("Usage: python generate_evaluators.py [tools_directory]")
        print("Example: python generate_evaluators.py evolution/tools")
        sys.exit(1)
    
    # Check for OpenAI API key
    if not os.getenv('OPENAI_API_KEY'):
        print("Error: OPENAI_API_KEY environment variable is required")
        print("Please set your OpenAI API key: export OPENAI_API_KEY=your_key_here")
        sys.exit(1)
    
    tools_directory = sys.argv[1] if len(sys.argv) == 2 else None
    
    generator = EvaluatorGenerator()
    generator.generate_evaluators(tools_directory)


if __name__ == "__main__":
    main()