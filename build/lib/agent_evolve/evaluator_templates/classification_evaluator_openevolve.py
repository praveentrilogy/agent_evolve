"""
Classification Evaluator Template - OpenEvolve Format
Auto-generated evaluator
"""
import json
import os
import sys
from pathlib import Path

# Import LLM function for evaluation
try:
    from agent_evolve.llm import get_llm_response, get_llm_response_json
except ImportError:
    def get_llm_response(prompt, model="gpt-5", temperature=1):
        raise NotImplementedError("Please implement get_llm_response function or install agent_evolve.llm")
    
    def get_llm_response_json(prompt, model="gpt-5", temperature=1):
        raise NotImplementedError("Please implement get_llm_response_json function or install agent_evolve.llm")

def load_training_data():
    """Load training data from filesystem"""
    current_dir = Path(__file__).parent
    training_data_file = current_dir / "training_data.json"
    
    if training_data_file.exists():
        with open(training_data_file, 'r') as f:
            return json.load(f)
    return []

def extract_target_from_program(program_path):
    """Extract the target function/prompt from the program file"""
    try:
        # Import the program module
        spec = __import__('importlib.util', fromlist=['spec_from_file_location']).spec_from_file_location(
            "program_module", program_path
        )
        program_module = __import__('importlib.util', fromlist=['module_from_spec']).module_from_spec(spec)
        spec.loader.exec_module(program_module)
        
        target_name = Path(__file__).parent.name
        
        # Debug: print what we're looking for and what we found
        print(f"Looking for target: {target_name}")
        all_attrs = [(name, type(getattr(program_module, name)).__name__) for name in dir(program_module) if not name.startswith('_')]
        print(f"Available attributes: {all_attrs}")
        
        # FIRST: Look for string constants (prioritize these over functions)
        for attr_name in dir(program_module):
            if not attr_name.startswith('_'):
                attr = getattr(program_module, attr_name)
                if isinstance(attr, str) and len(attr) > 50:
                    print(f"Found string constant: {attr_name} (length: {len(attr)})")
                    # Prefer exact match or _TEXT variants
                    if (attr_name == target_name or 
                        attr_name == f"{target_name}_TEXT" or 
                        attr_name.startswith(target_name)):
                        print(f"Using string constant: {attr_name}")
                        return attr
        
        # SECOND: Try to read the source code directly and extract the string
        try:
            with open(program_path, 'r') as f:
                source_code = f.read()
            
            # Look for string assignment patterns
            import re
            # Pattern to match: SOMETHING = """...""" or SOMETHING = "..."
            string_patterns = [
                rf'{target_name}_TEXT\s*=\s*"""(.*?)"""',
                rf'{target_name}_TEXT\s*=\s*"([^"]*)"',
                rf'{target_name}_TEXT\s*=\s*\'([^\']*)\'',
                # Also look for the original pattern in case it's there
                rf'{target_name}\s*=\s*"""(.*?)"""',
            ]
            
            for pattern in string_patterns:
                match = re.search(pattern, source_code, re.DOTALL)
                if match:
                    extracted_string = match.group(1).strip()
                    if len(extracted_string) > 50:
                        print(f"Extracted string from source: {extracted_string[:100]}...")
                        return extracted_string
            
        except Exception as e:
            print(f"Error reading source code: {e}")
        
        # THIRD: Try the function approach only if no string found
        if hasattr(program_module, target_name):
            target = getattr(program_module, target_name)
            if callable(target):
                try:
                    result = target()
                    print(f"Function {target_name}() returned: {type(result)} (callable: {callable(result)})")
                    if isinstance(result, str) and len(result) > 50:
                        print(f"Using function result: {result[:100]}...")
                        return result
                    else:
                        print(f"Function result is not a suitable string: {result}")
                except Exception as e:
                    print(f"Error calling function {target_name}(): {e}")
                    pass
        
        # THIRD: Look for any other callable that returns a string
        for attr_name in dir(program_module):
            if not attr_name.startswith('_'):
                attr = getattr(program_module, attr_name)
                if callable(attr):
                    try:
                        result = attr()
                        if isinstance(result, str) and len(result) > 50:
                            return result
                    except:
                        continue
        
        return None
        
    except Exception as e:
        print(f"Error extracting target from {program_path}: {e}")
        return None

def execute_classification_prompt(prompt_template, inputs):
    """Execute the classification prompt with given inputs using LLM"""
    try:
        # Format the prompt with inputs if it's a template and inputs is a dict
        if isinstance(inputs, dict):
            try:
                formatted_prompt = prompt_template.format(**inputs)
            except (KeyError, ValueError):
                # If formatting fails, just append inputs
                formatted_prompt = f"{prompt_template}\n\nInput: {inputs}"
        else:
            formatted_prompt = f"{prompt_template}\n\nInput: {inputs}"
        
        # Execute with LLM
        response = get_llm_response(formatted_prompt)
        return response.strip()
    except Exception as e:
        return f"Error: {e}"

def evaluate_classification_accuracy(predicted, expected):
    """Evaluate classification accuracy"""
    if not predicted or not expected:
        return 0.0
    
    # Clean up responses for comparison
    predicted = str(predicted).lower().strip()
    expected = str(expected).lower().strip()
    
    # Simple exact match comparison
    return 1.0 if predicted == expected else 0.0

def evaluate(program_path):
    """
    OpenEvolve-compatible evaluation function for classification tasks
    
    Args:
        program_path (str): Path to the program file to evaluate
        
    Returns:
        dict: Evaluation results with metrics
    """
    # Load training data from filesystem
    training_data = load_training_data()
    
    if not training_data:
        return {
            "metrics": {},
            "details": {"error": "No training data available"},
            "recommendations": ["Add training data to training_data.json file"]
        }
    
    # Extract the target prompt from the program
    target_prompt = extract_target_from_program(program_path)
    if not target_prompt:
        return {
            "metrics": {},
            "details": {"error": "Could not extract target prompt from program"},
            "recommendations": ["Ensure the program file contains the target prompt/function"]
        }
    
    # Initialize metrics tracking
    correct_predictions = 0
    total_predictions = 0
    accuracy_scores = []
    evaluation_details = []
    
    # Process each training sample
    for i, sample in enumerate(training_data[:50]):  # Limit for performance
        try:
            # Extract inputs and expected output from sample
            if isinstance(sample, dict):
                inputs = sample.get('input', sample.get('inputs', ''))
                expected_output = sample.get('output', sample.get('outputs', ''))
            else:
                inputs = str(sample)
                expected_output = ''
            
            if not expected_output:
                continue  # Skip samples without expected outputs
            
            # Execute classification prompt with LLM
            predicted_output = execute_classification_prompt(target_prompt, inputs)
            
            # Evaluate accuracy
            accuracy_score = evaluate_classification_accuracy(predicted_output, expected_output)
            accuracy_scores.append(accuracy_score)
            
            if accuracy_score >= 0.8:  # Consider >= 0.8 as correct
                correct_predictions += 1
            total_predictions += 1
            
            # Store detailed results
            evaluation_details.append({
                "sample_id": i,
                "inputs": str(inputs)[:100] + "..." if len(str(inputs)) > 100 else str(inputs),
                "predicted": predicted_output,
                "expected": expected_output,
                "accuracy_score": accuracy_score,
                "correct": accuracy_score >= 0.8
            })
            
        except Exception as e:
            evaluation_details.append({
                "sample_id": i,
                "error": f"Failed to evaluate sample: {e}",
                "inputs": str(inputs)[:100] + "..." if len(str(inputs)) > 100 else str(inputs)
            })
    
    if total_predictions == 0:
        return {
            "metrics": {},
            "details": {"error": "No valid samples with expected outputs found"},
            "recommendations": ["Ensure training data has 'output' field with expected classifications"]
        }
    
    # Calculate accuracy
    overall_accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
    
    return {
        "accuracy": overall_accuracy
    }

if __name__ == "__main__":
    # Get program path from command line argument
    if len(sys.argv) > 1:
        program_path = sys.argv[1]
    else:
        # Default to evolve_target.py in the same directory
        program_path = str(Path(__file__).parent / "evolve_target.py")
    
    results = evaluate(program_path)
    print(json.dumps(results, indent=2))