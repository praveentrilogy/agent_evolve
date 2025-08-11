"""
Generates evaluator code from prompts and training data.
"""

import json
import os
import sqlite3
import logging
from agent_evolve.llm import get_llm_response_json
from agent_evolve.prompts.evaluator_generation import GENERATE_PROMPT_EVALUATOR_PROMPT
logger = logging.getLogger(__name__)

def generate_evaluator(prompt_id: str, prompt_name: str, db_path: str):
    """Generate evaluator code from prompt usage."""
    logger.info(f"Generating evaluator for prompt '{prompt_name}' (ID: {prompt_id})")
    
    # Get prompt definition and real usage examples
    conn = sqlite3.connect(db_path)
    
    # Get prompt definition
    cursor = conn.execute('''
        SELECT content, variables, prompt_type 
        FROM prompts 
        WHERE id = ?
    ''', (prompt_id,))

    prompt_result = cursor.fetchone()
    if not prompt_result:
        logger.warning(f"Prompt {prompt_id} not found in database")
        conn.close()
        return
    
    prompt_content, variables_json, prompt_type = prompt_result
    variables = json.loads(variables_json) if variables_json else {}
    
    training_data_path = os.path.join(".agent_evolve", prompt_name, "synthesized_training_data.json")
    with open(training_data_path, 'r') as f:
        training_data = json.load(f)
    
    # Generate evaluator code
    evaluator_code = get_llm_response_json(GENERATE_PROMPT_EVALUATOR_PROMPT.format(
        prompt_content=prompt_content,
        training_data=json.dumps(training_data[:1], indent=2)
    ))

    print(evaluator_code)
    return evaluator_code

