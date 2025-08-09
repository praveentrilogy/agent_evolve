"""
Default Evaluator Template
Auto-generated evaluator for {item_name}
Type: {item_type}
"""
import json
import os
from pathlib import Path
import logging
import sqlite3

logger = logging.getLogger(__name__)

{import_statement}

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

def evaluate():
    """
    Main evaluation function - customize this for your specific use case
    
    Returns:
        dict: {{"metrics": dict, "details": dict, "recommendations": list}}
    """
    # Get parameters from command line or environment
    db_path = sys.argv[1] if len(sys.argv) > 1 else 'data/graph.db'
    prompt_id = sys.argv[2] if len(sys.argv) > 2 else None
    
    if not prompt_id:
        return {{"error": "No prompt_id provided"}}
    
    # Load training data
    real_data = load_training_data(db_path, prompt_id, data_source='real')
    synthetic_data = load_training_data(db_path, prompt_id, data_source='synthetic')
    all_data = real_data + synthetic_data
    
    if not all_data:
        return {{
            "metrics": {{}},
            "details": {{"error": "No training data available"}},
            "recommendations": ["Add training data files to this directory"]
        }}
    
    # Example evaluation logic - customize this!
    metrics = {{
        "accuracy": 0.85,
        "relevance": 0.90, 
        "quality": 0.80,
        "total_samples": len(all_data),
        "real_samples": len(real_data),
        "synthetic_samples": len(synthetic_data)
    }}
    
    details = {{
        "evaluated_samples": len(all_data),
        "sample_breakdown": {{
            "real": len(real_data),
            "synthetic": len(synthetic_data)
        }},
        "evaluation_method": "Template - needs customization"
    }}
    
    recommendations = [
        "Customize the evaluate() function for your specific use case",
        "Add actual {item_type} execution and comparison logic",
        "Define metrics appropriate for {item_type} evaluation"
    ]
    
    # TODO: Add your custom evaluation logic here
    # Example:
    # for sample in all_data:
    #     result = {item_name}(sample.get('input', ''))
    #     # Compare result with expected output
    #     # Calculate metrics
    
    return {{
        "metrics": metrics,
        "details": details,
        "recommendations": recommendations
    }}

if __name__ == "__main__":
    results = evaluate()
    print(json.dumps(results, indent=2))