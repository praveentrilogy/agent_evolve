"""
Classification Evaluator Template
Auto-generated Classification Evaluator for ${item_name}
Type: ${item_type}
"""
import json
import os
from pathlib import Path
import sqlite3
import sys

$import_statement
import logging

logger = logging.getLogger(__name__)

# Import LLM function for evaluation
try:
    # Try to import from the agent_evolve library
    sys.path.append(str(Path(__file__).parent.parent.parent / "agent_evolve"))
    from agent_evolve.llm import get_llm_response
except ImportError:
    # Fallback: user needs to implement their own LLM function
    def get_llm_response(prompt, model="gpt-4", temperature=0.7):
        raise NotImplementedError("Please implement get_llm_response function or install agent_evolve.llm")

def load_training_data(db_path, prompt_id, data_source=None):
    """Load training data from database"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    query = "SELECT inputs, outputs, data_source FROM prompt_training_data WHERE prompt_id = ?"
    params = [prompt_id]
    
    if data_source:
        query += " AND data_source = ?"
        params.append(data_source)
        
    cursor.execute(query, params)
    
    training_samples = []
    for row in cursor.fetchall():
        inputs, outputs, source = row
        training_samples.append({
            'inputs': json.loads(inputs) if inputs else {},
            'outputs': json.loads(outputs) if outputs else {},
            'data_source': source
        })
    
    conn.close()
    return training_samples

def execute_prompt_with_llm(prompt_template, inputs):
    """Execute the prompt template with given inputs using LLM"""
    try:
        # Format the prompt with inputs if it's a template
        if isinstance(inputs, dict):
            formatted_prompt = prompt_template.format(**inputs)
        else:
            formatted_prompt = f"{prompt_template}\n\nInput: {inputs}"
        
        # Execute with LLM
        response = get_llm_response(formatted_prompt, temperature=0.0) # Temperature 0 for classification
        return response
    except Exception as e:
        return f"Error executing prompt: {e}"

def evaluate():
    """
    Classification Evaluation Function
    
    For each training sample:
    1. Execute the prompt against an LLM
    2. Compare the output to the expected output
    3. Calculate aggregate metrics
    
    Returns:
        dict: {"metrics": dict, "details": dict, "recommendations": list}
    """
    db_path = sys.argv[1] if len(sys.argv) > 1 else 'data/graph.db'
    prompt_id = "$item_id"
    
    all_data = load_training_data(db_path, prompt_id)
    
    if not all_data:
        return {
            "metrics": {},
            "details": {"error": "No training data available"},
            "recommendations": ["Add training data files to this directory"]
        }
    
    true_positives = 0
    false_positives = 0
    true_negatives = 0
    false_negatives = 0
    
    evaluation_details = []
    
    for i, sample in enumerate(all_data):
        try:
            inputs = sample.get('inputs', sample.get('input', ''))
            expected_output = sample.get('outputs', sample.get('output', ''))
            
            generated_output = execute_prompt_with_llm(${item_name}, inputs)
            
            # Simple exact match for classification
            is_correct = str(generated_output).strip().lower() == str(expected_output).strip().lower()
            
            if is_correct:
                true_positives += 1
            else:
                false_positives += 1

            evaluation_details.append({
                "sample_id": i,
                "inputs": str(inputs)[:100] + "...",
                "generated_output": generated_output,
                "expected_output": expected_output,
                "is_correct": is_correct
            })

        except Exception as e:
            evaluation_details.append({
                "sample_id": i,
                "error": f"Failed to evaluate sample: {e}",
            })

    total_samples = len(all_data)
    accuracy = (true_positives) / total_samples if total_samples > 0 else 0
    
    metrics = {
        "accuracy": accuracy,
        "total_samples": total_samples,
        "correct_predictions": true_positives,
        "incorrect_predictions": false_positives
    }
    
    recommendations = []
    if accuracy < 0.8:
        recommendations.append("Accuracy is below 80%. Consider refining the prompt or providing more/better training data.")

    return {
        "metrics": metrics,
        "details": {
            "evaluation_method": "Exact match classification",
            "evaluation_details": evaluation_details
        },
        "recommendations": recommendations
    }

if __name__ == "__main__":
    results = evaluate()
    print(json.dumps(results, indent=2))
