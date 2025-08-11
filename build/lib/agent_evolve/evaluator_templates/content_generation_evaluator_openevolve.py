"""
Content Generation Evaluator Template - OpenEvolve Format
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
    def get_llm_response(prompt, model="gpt-4", temperature=0.7):
        raise NotImplementedError("Please implement get_llm_response function or install agent_evolve.llm")
    
    def get_llm_response_json(prompt, model="gpt-4", temperature=0.0):
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

def execute_prompt_with_llm(prompt_template, inputs):
    """Execute the prompt template with given inputs using LLM"""
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
        return response
    except Exception as e:
        return f"Error executing prompt: {e}"

def grade_content_output(original_prompt, inputs, generated_output):
    """Grade the generated content using LLM on multiple metrics"""
    
    grading_prompt = f"""You are an expert content evaluator. Please evaluate the following generated content on these specific metrics:

**Original Prompt/Task:** {original_prompt}

**Input/Context:** {inputs}

**Generated Content:** {generated_output}

Please rate the content on a scale of 0.0 to 1.0 (where 1.0 is excellent) for each metric:

1. **Readability**: How clear, well-structured, and easy to understand is the content?
2. **Engagement**: How compelling, interesting, and likely to capture the reader's attention?  
3. **Authenticity**: How natural, genuine, and human-like does the content sound?
4. **Depth**: How thorough, detailed, and substantive is the content?

Also provide:
5. **Qualitative Feedback**: 2-3 sentences of specific feedback on strengths and areas for improvement.

Respond ONLY with a JSON object in this exact format:
{{
  "readability": <score 0.0-1.0>,
  "engagement": <score 0.0-1.0>,
  "authenticity": <score 0.0-1.0>,
  "depth": <score 0.0-1.0>,
  "qualitative_feedback": "<specific feedback>"
}} """

    try:
        result = get_llm_response_json(grading_prompt)
        
        # Validate the response has required fields
        required_fields = ["readability", "engagement", "authenticity", "depth", "qualitative_feedback"]
        if not all(field in result for field in required_fields):
            return {
                "readability": 0.5,
                "engagement": 0.5,
                "authenticity": 0.5,
                "depth": 0.5,
                "qualitative_feedback": "Error: LLM response missing required fields"
            }
        
        return result
        
    except Exception as e:
        return {
            "readability": 0.0,
            "engagement": 0.0,
            "authenticity": 0.0,
            "depth": 0.0,
            "qualitative_feedback": f"Error during grading: {e}"
        }

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
    all_scores = {
        "readability": [],
        "engagement": [],
        "authenticity": [],
        "depth": []
    }
    
    all_feedback = []
    evaluation_details = []
    
    # Process each training sample (limit to 10 samples to control costs)
    for i, sample in enumerate(training_data[:10]):
        try:
            # Extract inputs from sample
            if isinstance(sample, dict):
                inputs = sample.get('input', sample.get('inputs', ''))
                expected_output = sample.get('output', sample.get('outputs', ''))
            else:
                inputs = str(sample)
                expected_output = ''
            
            # Execute prompt with LLM
            generated_output = execute_prompt_with_llm(target_prompt, inputs)

            # Grade the generated output
            scores = grade_content_output(target_prompt, inputs, generated_output)

            # Collect scores
            for metric in ["readability", "engagement", "authenticity", "depth"]:
                if metric in scores and isinstance(scores[metric], (int, float)):
                    all_scores[metric].append(scores[metric])
            
            # Collect feedback
            feedback = scores.get("qualitative_feedback", "No feedback available")
            all_feedback.append(feedback)
            
            # Store detailed results
            evaluation_details.append({
                "sample_id": i,
                "inputs": str(inputs)[:100] + "..." if len(str(inputs)) > 100 else str(inputs),
                "generated_output": generated_output[:200] + "..." if len(generated_output) > 200 else generated_output,
                "expected_output": str(expected_output)[:100] + "..." if len(str(expected_output)) > 100 else str(expected_output),
                "scores": scores
            })
            
        except Exception as e:
            evaluation_details.append({
                "sample_id": i,
                "error": f"Failed to evaluate sample: {e}"
            })
    
    # Calculate aggregate metrics
    metrics = {}
    for metric_name, scores in all_scores.items():
        if scores:
            metrics[f"avg_{metric_name}"] = sum(scores) / len(scores)
            metrics[f"min_{metric_name}"] = min(scores)
            metrics[f"max_{metric_name}"] = max(scores)
    
    # Overall score (average of all metrics)
    metric_averages = [metrics.get(f"avg_{m}", 0) for m in ["readability", "engagement", "authenticity", "depth"]]
    metrics["overall_score"] = sum(metric_averages) / len(metric_averages) if metric_averages else 0
    
    # Add count metrics
    metrics.update({
        "total_samples_evaluated": len(evaluation_details),
        "successful_evaluations": len([d for d in evaluation_details if "error" not in d])
    })
    
    # Generate recommendations
    recommendations = []
    if metrics.get("overall_score", 0) < 0.6:
        recommendations.append("Overall content quality is below average. Consider refining the prompt.")
    if metrics.get("avg_readability", 0) < 0.7:
        recommendations.append("Content readability needs improvement. Consider clearer structure and language.")
    if metrics.get("avg_engagement", 0) < 0.7:
        recommendations.append("Content engagement is low. Add more compelling hooks and interesting elements.")
    if metrics.get("avg_authenticity", 0) < 0.7:
        recommendations.append("Content sounds artificial. Work on making it more natural and human-like.")
    if metrics.get('avg_depth', 0) < 0.7:
        recommendations.append("Content lacks depth. Add more detailed information and insights.")
    
    if len(training_data) < 5:
        recommendations.append("Add more training samples for better evaluation coverage.")
    
    return {
        "metrics": metrics,
        "details": {
            "evaluated_samples": len(evaluation_details),
            "evaluation_method": "LLM-based content grading",
            "evaluation_details": evaluation_details[:5]  # Limit details to avoid huge outputs
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