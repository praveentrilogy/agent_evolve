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
import openai
import re
import ast

from agent_evolve.prompts.evaluator_prompts import (
    FUNCTION_ANALYSIS_PROMPT,
    VALIDATION_PROMPT,
    get_quantitative_evaluator_prompt,
    get_prompt_evaluator_prompt,
    get_content_evaluator_prompt
)


class EvaluatorGenerator:
    """Generates OpenEvolve evaluators for agent tools using LLM"""
    
    def __init__(self, model_name: str = "gpt-4o"):
        self.client = openai.OpenAI()
        self.model_name = model_name
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

    def _strip_markdown_code_blocks(self, code: str) -> str:
        """Strip markdown code blocks from the generated code"""
        
        
        # Remove markdown code blocks with language specification
        code = re.sub(r'```python\s*\n', '', code)
        code = re.sub(r'```\s*\n', '', code)
        code = re.sub(r'\n```$', '', code)
        
        # Also handle cases where there might be no newline after opening
        code = re.sub(r'```python', '', code)
        code = re.sub(r'```', '', code)
        
        return code.strip()

    def _clean_json_response(self, response_text: str) -> str:
        """Clean JSON response by removing markdown formatting and extracting JSON"""
        
        # Remove any markdown code blocks
        response_text = self._strip_markdown_code_blocks(response_text)
        
        # Try to extract JSON from the response
        # Look for JSON object pattern
        json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
        json_matches = re.findall(json_pattern, response_text, re.DOTALL)
        
        if json_matches:
            # Return the longest match (most likely to be complete JSON)
            return max(json_matches, key=len)
        
        # If no JSON pattern found, try to find JSON array pattern
        array_pattern = r'\[[^\[\]]*(?:\{[^{}]*\}[^\[\]]*)*\]'
        array_matches = re.findall(array_pattern, response_text, re.DOTALL)
        
        if array_matches:
            # Return the longest match
            return max(array_matches, key=len)
        
        # If still no JSON pattern found, return the cleaned text
        return response_text.strip()

    def _fix_common_import_errors(self, code: str) -> str:
        """Fix common import errors and enforce raw OpenAI API usage"""
        
        # First strip any markdown code blocks
        code = self._strip_markdown_code_blocks(code)
        
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
        validation_prompt = VALIDATION_PROMPT.format(code=code)

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "system", "content": validation_prompt}],
                temperature=0.0
            )


            print(f"LLM Validation Response: {response.choices[0].message.content}")

            # Clean the response to extract JSON
            cleaned_response = self._clean_json_response(response.choices[0].message.content)
            print(f"Cleaned JSON response: {cleaned_response}")
            
            try:
                result = json.loads(cleaned_response)
                return result
            except json.JSONDecodeError as json_error:
                print(f"JSON parsing error: {json_error}")
                print(f"Failed to parse: {cleaned_response}")
                # Try to extract JSON with more aggressive cleaning
                import re
                # Remove any non-JSON text before and after
                json_only = re.sub(r'^[^{]*', '', cleaned_response)
                json_only = re.sub(r'[^}]*$', '', json_only)
                try:
                    result = json.loads(json_only)
                    return result
                except:
                    pass
                
                # Fallback to basic validation
                return {
                    "is_valid": self._validate_evaluator_code(code),
                    "issues": [f"JSON parsing failed: {json_error}"],
                    "suggested_fix": None
                }
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
        analysis_prompt = FUNCTION_ANALYSIS_PROMPT.format(
            source_code=source_code,
            training_sample=training_sample
        )

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "system", "content": analysis_prompt}],
            temperature=0.0
        )
        print(response.choices[0].message.content)
        
        try:
            # Extract JSON from response
            content = response.choices[0].message.content.strip()
            # Clean the response to extract JSON
            cleaned_content = self._clean_json_response(content)
            
            analysis = json.loads(cleaned_content)
            description = analysis.get("function_description", f"Function: {tool_name}")
            function_type = analysis.get("function_type", "creative").lower()
            metrics = analysis.get("metrics", ["quality", "relevance", "usefulness", "clarity"])
            
            print(f"Function analysis: {description}")
            print(f"Function type: {function_type}")
            print(f"Selected metrics: {metrics}")
            
            return description, function_type, metrics
            
        except Exception as e:
            print(f"Error parsing function analysis: {e}")
            print(f"Raw response: {response.choices[0].message.content}")
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
            evaluation_prompt = get_quantitative_evaluator_prompt(
                tool_name, function_description, metrics, source_code, training_data, feedback_section
            )
        elif function_type == "prompt" or "template" in function_type or function_type == "classification":
            evaluation_prompt = get_prompt_evaluator_prompt(
                tool_name, function_description, metrics, source_code, training_data, feedback_section
            )
        else:
            evaluation_prompt = get_content_evaluator_prompt(
                tool_name, function_description, metrics, source_code, training_data, feedback_section
            )

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "system", "content": evaluation_prompt}],
            temperature=0.0
        )
        print(f"LLM Response: {response.choices[0].message.content}")

        
        try:
            evaluator_code = response.choices[0].message.content
            
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
            print(f"Raw response: {response.choices[0].message.content}")
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