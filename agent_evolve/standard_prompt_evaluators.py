from agent_evolve.prompts.evaluators import CONTENT_GENERATION_EVALUATOR_PROMPT
import sqlite3
import os
import json
import logging
from agent_evolve.llm import get_llm_response_json

logger = logging.getLogger(__name__)


def evaluate_content_generation(db_path: str, prompt_id: str):
    """Evaluate the content generation prompt using database training data."""
    
    logger.info(f"Starting evaluation for prompt_id: {prompt_id}, db_path: {db_path}")
    
    try:
        # Get prompt definition and training data
        logger.info("Connecting to database...")
        conn = sqlite3.connect(db_path)
        
        # Get prompt definition
        logger.info("Querying prompt details...")
        cursor = conn.execute('''
            SELECT prompt_name, content, variables, prompt_type 
            FROM prompts 
            WHERE id = ?
        ''', (prompt_id,))

        prompt_result = cursor.fetchone()
        if not prompt_result:
            logger.warning(f"Prompt {prompt_id} not found in database")
            conn.close()
            return {"error": f"Prompt {prompt_id} not found"}
        
        logger.info(f"Found prompt: {prompt_result}")
        prompt_name, prompt_content, variables_json, prompt_type = prompt_result
        variables = json.loads(variables_json) if variables_json else {}
        
        # Get training data from database
        logger.info("Querying training data...")
        cursor = conn.execute('''
            SELECT inputs, outputs, data_source
            FROM prompt_training_data
            WHERE prompt_id = ?
            ORDER BY created_at DESC
            LIMIT 10
        ''', (prompt_id,))
    
        training_samples = []
        rows = cursor.fetchall()
        logger.info(f"Found {len(rows)} training data rows")
        
        for row in rows:
            inputs, outputs, data_source = row
            training_samples.append({
                'inputs': json.loads(inputs) if inputs else {},
                'outputs': json.loads(outputs) if outputs else {},
                'data_source': data_source
            })
        
        conn.close()
        logger.info(f"Processed {len(training_samples)} training samples")
        
        if not training_samples:
            logger.warning("No training data available")
            return {"error": "No training data available for evaluation"}
        
        # Convert database format to expected format for the evaluator
        training_data = []
        for sample in training_samples[:5]:  # Use first 5 samples
            training_data.append({
                'variable_values': sample['inputs'],
                'rendered_content': sample['outputs']
            })
        
        logger.info(f"Prepared {len(training_data)} samples for LLM evaluation")
        
        # Generate evaluator code using LLM
        # Format the training data as generated content for evaluation
        generated_content = json.dumps(training_data, indent=2)
        
        evaluation_result = get_llm_response_json(CONTENT_GENERATION_EVALUATOR_PROMPT.format(
            request=prompt_content,
            generated_content=generated_content,
        ))
        
        logger.info(f"Evaluation result: {evaluation_result}")
        return evaluation_result
        
    except Exception as e:
        logger.error(f"Error during evaluation: {e}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        return {"error": str(e), "traceback": traceback.format_exc()}