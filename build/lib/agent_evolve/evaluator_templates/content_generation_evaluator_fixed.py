"""
Content Generation Evaluator Template
Auto-generated Content Generation Evaluator for {item_name}
Type: {item_type}
"""
import json
import os
from pathlib import Path

{import_statement}

# Import LLM function for evaluation
try:
    # Try to import from the agent_evolve library
    import sys
    sys.path.append(str(Path(__file__).parent.parent.parent / "agent_evolve"))
    from agent_evolve.llm import get_llm_response, get_llm_response_json
except ImportError:
    # Fallback: user needs to implement their own LLM function
    def get_llm_response(prompt, model="gpt-4", temperature=0.7):
        raise NotImplementedError("Please implement get_llm_response function or install agent_evolve.llm")
    
    def get_llm_response_json(prompt, model="gpt-4", temperature=0.0):
        raise NotImplementedError("Please implement get_llm_response_json function or install agent_evolve.llm")

def load_training_data():
    """Load real training data from files"""
    current_dir = Path(__file__).parent
    real_data_file = current_dir / "real_training_data.json"
    
    if real_data_file.exists():
        with open(real_data_file, 'r') as f:
            return json.load(f)
    return []

def load_synthetic_data():
    """Load synthetic training data from files"""
    current_dir = Path(__file__).parent
    synthetic_data_file = current_dir / "synthesized_training_data.json"
    
    if synthetic_data_file.exists():
        with open(synthetic_data_file, 'r') as f:
            return json.load(f)
    return []

def execute_prompt_with_llm(prompt_template, inputs):
    """Execute the prompt template with given inputs using LLM"""
    try:
        # Format the prompt with inputs if it's a template
        if isinstance(inputs, dict):
            formatted_prompt = prompt_template.format(**inputs)
        else:
            formatted_prompt = str(prompt_template) + "\\n\\nInput: " + str(inputs)
        
        # Execute with LLM
        response = get_llm_response(formatted_prompt, temperature=0.7)
        return response
    except Exception as e:
        return "Error executing prompt: " + str(e)

def grade_content_output(original_prompt, inputs, generated_output):
    """Grade the generated content using LLM on multiple metrics"""
    
    grading_prompt = """You are an expert content evaluator. Please evaluate the following generated content on these specific metrics:

**Original Prompt/Task:** """ + str(original_prompt) + """

**Input/Context:** """ + str(inputs) + """

**Generated Content:** """ + str(generated_output) + """

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
}}"""

    try:
        result = get_llm_response_json(grading_prompt, temperature=0.0)
        
        # Validate the response has required fields
        required_fields = ["readability", "engagement", "authenticity", "depth", "qualitative_feedback"]
        if not all(field in result for field in required_fields):
            return {{
                "readability": 0.5,
                "engagement": 0.5,
                "authenticity": 0.5,
                "depth": 0.5,
                "qualitative_feedback": "Error: LLM response missing required fields"
            }}
        
        return result
        
    except Exception as e:
        return {{
            "readability": 0.0,
            "engagement": 0.0,
            "authenticity": 0.0,
            "depth": 0.0,
            "qualitative_feedback": "Error during grading: " + str(e)
        }}

def evaluate():
    """
    Content Generation Evaluation Function
    
    For each training sample:
    1. Execute the prompt against an LLM
    2. Grade the output using a second LLM call
    3. Calculate aggregate metrics
    
    Returns:
        dict: {{"metrics": dict, "details": dict, "recommendations": list}}
    """
    # Load training data
    real_data = load_training_data()
    synthetic_data = load_synthetic_data()
    all_data = real_data + synthetic_data
    
    if not all_data:
        return {{
            "metrics": {{}},
            "details": {{"error": "No training data available"}},
            "recommendations": ["Add training data files to this directory"]
        }}
    
    # Initialize metrics tracking
    all_scores = {{
        "readability": [],
        "engagement": [],
        "authenticity": [],
        "depth": []
    }}
    
    all_feedback = []
    evaluation_details = []
    
    # Process each training sample
    for i, sample in enumerate(all_data[:10]):  # Limit to 10 samples to control costs
        try:
            # Extract inputs from sample
            inputs = sample.get('inputs', sample.get('input', ''))
            expected_output = sample.get('outputs', sample.get('output', ''))
            
            # Execute prompt with LLM
            generated_output = execute_prompt_with_llm({item_name}, inputs)
            
            # Grade the generated output
            scores = grade_content_output({item_name}, inputs, generated_output)
            
            # Collect scores
            for metric in ["readability", "engagement", "authenticity", "depth"]:
                if metric in scores and isinstance(scores[metric], (int, float)):
                    all_scores[metric].append(scores[metric])
            
            # Collect feedback
            feedback = scores.get("qualitative_feedback", "No feedback available")
            all_feedback.append(feedback)
            
            # Store detailed results
            evaluation_details.append({{
                "sample_id": i,
                "inputs": str(inputs)[:100] + "..." if len(str(inputs)) > 100 else str(inputs),
                "generated_output": generated_output[:200] + "..." if len(generated_output) > 200 else generated_output,
                "expected_output": str(expected_output)[:100] + "..." if len(str(expected_output)) > 100 else str(expected_output),
                "scores": scores,
                "data_source": sample.get('data_source', 'unknown')
            }})
            
        except Exception as e:
            evaluation_details.append({{
                "sample_id": i,
                "error": "Failed to evaluate sample: " + str(e),
                "data_source": sample.get('data_source', 'unknown')
            }})
    
    # Calculate aggregate metrics
    metrics = {{}}
    for metric_name, scores in all_scores.items():
        if scores:
            metrics["avg_" + metric_name] = sum(scores) / len(scores)
            metrics["min_" + metric_name] = min(scores)
            metrics["max_" + metric_name] = max(scores)
    
    # Overall score (average of all metrics)
    metric_averages = [metrics.get("avg_" + m, 0) for m in ["readability", "engagement", "authenticity", "depth"]]
    metrics["overall_score"] = sum(metric_averages) / len(metric_averages) if metric_averages else 0
    
    # Add count metrics
    metrics.update({{
        "total_samples_evaluated": len(evaluation_details),
        "real_samples": len(real_data),
        "synthetic_samples": len(synthetic_data),
    }})
    
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
    if metrics.get("avg_depth", 0) < 0.7:
        recommendations.append("Content lacks depth. Add more detailed information and insights.")
    
    if len(all_data) < 5:
        recommendations.append("Add more training samples for better evaluation coverage.")
    
    return {{
        "metrics": metrics,
        "details": {{
            "evaluated_samples": len(evaluation_details),
            "sample_breakdown": {{
                "real": len(real_data),
                "synthetic": len(synthetic_data)
            }},
            "evaluation_method": "LLM-based content generation evaluation",
            "sample_evaluations": evaluation_details,
            "qualitative_feedback_summary": all_feedback[:5]  # First 5 feedback items
        }},
        "recommendations": recommendations
    }}

if __name__ == "__main__":
    results = evaluate()
    print(json.dumps(results, indent=2))