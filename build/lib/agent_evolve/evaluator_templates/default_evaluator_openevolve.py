"""
Default Evaluator Template - OpenEvolve Format
Auto-generated evaluator
"""
import json
import os
import sys
from pathlib import Path

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
        
        # FIRST: Look for string constants (prioritize these over functions)
        for attr_name in dir(program_module):
            if not attr_name.startswith('_'):
                attr = getattr(program_module, attr_name)
                if isinstance(attr, str) and len(attr) > 50:
                    # Prefer exact match or _TEXT variants
                    if (attr_name == target_name or 
                        attr_name == f"{target_name}_TEXT" or 
                        attr_name.startswith(target_name)):
                        return attr
        
        # SECOND: Try the function approach only if no string found
        if hasattr(program_module, target_name):
            target = getattr(program_module, target_name)
            if callable(target):
                try:
                    result = target()
                    if isinstance(result, str) and len(result) > 50:
                        return result
                except:
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

def basic_function_evaluation(func, inputs):
    """Basic evaluation for functions"""
    try:
        if callable(func):
            # Try to call the function with inputs
            if isinstance(inputs, dict):
                result = func(**inputs)
            elif isinstance(inputs, (list, tuple)):
                result = func(*inputs)
            else:
                result = func(inputs)
            return result, None
        else:
            return str(func), None
    except Exception as e:
        return None, str(e)

def basic_prompt_evaluation(prompt, inputs):
    """Basic evaluation for prompts"""
    try:
        if isinstance(prompt, str):
            # Try to format the prompt with inputs if possible
            if isinstance(inputs, dict):
                try:
                    formatted = prompt.format(**inputs)
                    return formatted, None
                except (KeyError, ValueError):
                    return f"{prompt}\n\nInput: {inputs}", None
            else:
                return f"{prompt}\n\nInput: {inputs}", None
        return str(prompt), None
    except Exception as e:
        return None, str(e)

def evaluate(program_path):
    """
    OpenEvolve-compatible evaluation function
    
    Args:
        program_path (str): Path to the program file to evaluate
        
    Returns:
        dict: Evaluation results with metrics
    """
    # Load training data from filesystem
    training_data = load_training_data()
    
    if not training_data:
        return {
            "metrics": {"basic_score": 0.5},
            "details": {"message": "No training data available - using basic evaluation"},
            "recommendations": ["Add training data to training_data.json file for better evaluation"]
        }
    
    # Extract the target from the program
    target = extract_target_from_program(program_path)
    if target is None:
        return {
            "metrics": {"basic_score": 0.0},
            "details": {"error": "Could not extract target from program"},
            "recommendations": ["Ensure the program file contains the target function/prompt"]
        }
    
    # Determine if target is a function or prompt
    is_function = callable(target)
    
    # Initialize metrics tracking
    successful_evaluations = 0
    total_evaluations = 0
    evaluation_details = []
    
    # Process each training sample
    for i, sample in enumerate(training_data[:20]):  # Limit for performance
        try:
            # Extract inputs from sample
            if isinstance(sample, dict):
                inputs = sample.get('input', sample.get('inputs', sample.get('input_data', '')))
                expected_output = sample.get('output', sample.get('outputs', sample.get('expected', '')))
            else:
                inputs = sample
                expected_output = ''
            
            # Evaluate based on type
            if is_function:
                result, error = basic_function_evaluation(target, inputs)
            else:
                result, error = basic_prompt_evaluation(target, inputs)
            
            total_evaluations += 1
            
            if error is None:
                successful_evaluations += 1
                evaluation_success = True
            else:
                evaluation_success = False
            
            # Store detailed results
            evaluation_details.append({
                "sample_id": i,
                "inputs": str(inputs)[:100] + "..." if len(str(inputs)) > 100 else str(inputs),
                "result": str(result)[:200] + "..." if result and len(str(result)) > 200 else str(result),
                "expected": str(expected_output)[:100] + "..." if len(str(expected_output)) > 100 else str(expected_output),
                "success": evaluation_success,
                "error": error
            })
            
        except Exception as e:
            total_evaluations += 1
            evaluation_details.append({
                "sample_id": i,
                "error": f"Failed to evaluate sample: {e}",
                "success": False
            })
    
    # Calculate basic metrics
    success_rate = successful_evaluations / total_evaluations if total_evaluations > 0 else 0
    
    metrics = {
        "success_rate": success_rate,
        "basic_score": success_rate * 0.8,  # Conservative score
        "total_samples_evaluated": total_evaluations,
        "successful_evaluations": successful_evaluations,
        "target_type": "function" if is_function else "prompt"
    }
    
    # Generate recommendations
    recommendations = []
    if success_rate < 0.8:
        recommendations.append("Success rate is low. Check if inputs match the expected function/prompt format.")
    if success_rate < 0.5:
        recommendations.append("Very low success rate. Review the training data format and target implementation.")
    if total_evaluations < 5:
        recommendations.append("Add more training samples for better evaluation coverage.")
    
    recommendations.append("This is a basic evaluator. Consider creating a specialized evaluator for better metrics.")
    
    return {
        "metrics": metrics,
        "details": {
            "evaluated_samples": total_evaluations,
            "evaluation_method": "Basic function/prompt execution",
            "target_type": "function" if is_function else "prompt",
            "evaluation_details": evaluation_details[:5]  # Limit details
        },
        "recommendations": recommendations
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