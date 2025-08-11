"""
Generates test data from prompt usage.
"""

import json
import os
import sqlite3
import logging
import uuid
from datetime import datetime
from agent_evolve.llm import get_llm_response_json
from agent_evolve.prompts.training_data import TRAINING_DATA_GENERATION_PROMPT
logger = logging.getLogger(__name__)

def get_prompt_usage(prompt_id: str, db_path: str) -> list:
    """Get usage records for a specific prompt"""
    conn = sqlite3.connect(db_path)
    cursor = conn.execute('''
        SELECT 
            pu.variable_values
        FROM prompt_usages pu
        WHERE pu.prompt_id = ?
    ''', (prompt_id,))
    
    usage_records = []
    for row in cursor.fetchall():
        usage_records.append({
            'variable_values': row[0]
        })
    
    conn.close()
    return usage_records



def extract_training_data_from_prompt_usage(prompt_id: str, prompt_name: str, db_path: str):
    """Generate test data from prompt usage."""
    logger.info(f"Generating test data for prompt '{prompt_name}' (ID: {prompt_id})")
    usage_data = get_prompt_usage(prompt_id, db_path)
    if not usage_data:
        logger.warning(f"No usage data found for prompt '{prompt_name}'. Skipping test data generation.")
        return

    test_data = []
    seen_inputs = set()
    for usage in usage_data:
        try:
            inputs = json.loads(usage['variable_values'])
            inputs_tuple = tuple(sorted(inputs.items()))
            if inputs_tuple not in seen_inputs:
                test_data.append({"inputs": inputs, "outputs": None})
                seen_inputs.add(inputs_tuple)
        except (json.JSONDecodeError, TypeError) as e:
            logger.warning(f"Skipping invalid usage record for prompt '{prompt_name}': {e}")
            continue

    if not test_data:
        logger.warning(f"No valid test data could be generated for prompt '{prompt_name}'.")
        return

    # Limit to 1000 records for real training data
    test_data = test_data[:1000]

    # Save to database instead of file
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    now = datetime.utcnow().isoformat()
    
    # Insert training data into database with data_source as 'real'
    for data_item in test_data:
        cursor.execute('''
            INSERT INTO prompt_training_data (id, prompt_id, inputs, outputs, data_source, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            str(uuid.uuid4()),
            prompt_id,
            json.dumps(data_item['inputs']),
            json.dumps(data_item.get('outputs', {})),
            'real',
            now,
            now
        ))
    
    conn.commit()
    conn.close()
    logger.info(f"Successfully generated {len(test_data)} real training samples for '{prompt_name}' in database.")


def generate_training_data(prompt_id: str, prompt_name: str, db_path: str, num_samples: int = 5):
    
    # Get the training data from the prompt usage
    extract_training_data_from_prompt_usage(prompt_id, prompt_name, db_path)

    #Synthesize additional training data using LLM
    synthesize_additional_training_data(prompt_id, prompt_name, db_path, num_samples)

def synthesize_additional_training_data(prompt_id: str, prompt_name: str, db_path: str, num_samples: int = 5):
    """Synthesize additional training data using LLM."""
    logger.info(f"Synthesizing additional training data for prompt '{prompt_name}' (ID: {prompt_id})")
    
    try:
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
        
        # Get real usage examples (limit to 5 for the prompt)
        cursor = conn.execute('''
            SELECT variable_values, rendered_content 
            FROM prompt_usages 
            WHERE prompt_id = ? 
            ORDER BY timestamp DESC 
            LIMIT 5
        ''', (prompt_id,))
        
        real_examples = []
        for row in cursor.fetchall():
            variable_values, rendered_content = row
            try:
                var_values = json.loads(variable_values) if variable_values else {}
                real_examples.append({
                    'variable_values': var_values,
                    'rendered_content': rendered_content
                })
            except json.JSONDecodeError:
                continue
        
        conn.close()
        
        if not real_examples:
            logger.warning(f"No real usage examples found for prompt '{prompt_name}'")
            return
        
        print("Examples: ", real_examples)
        
        # Create prompt for LLM to generate additional training data
        llm_prompt = TRAINING_DATA_GENERATION_PROMPT.format(
            num_samples=num_samples,
            prompt_content=prompt_content,
            variables=json.dumps(variables, indent=2),
            # real_examples=json.dumps(real_examples[:1], indent=2)
        )


        # Generate additional training data with increased token limit
        generated_content = get_llm_response_json(llm_prompt, model="gpt-5", temperature=0.7)
        print("Generated content: ", generated_content)
        
        # Parse the generated JSON
        try:
            # Clean up any markdown formatting
            
            generated_examples = generated_content['training_data']
            
            if not isinstance(generated_examples, list):
                logger.error("Generated content is not a list")
                return
            
            # Validate and clean generated examples
            valid_examples = []
            for example in generated_examples:
                if isinstance(example, dict) and 'variable_values' in example and 'rendered_content' in example:
                    valid_examples.append({
                        'inputs': example['variable_values'],
                        'outputs': None  # Keep consistent with real training data format
                    })
            
            if not valid_examples:
                logger.warning("No valid examples generated by LLM")
                return
            
            # Save synthesized training data to database
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            now = datetime.utcnow().isoformat()
            
            # Insert synthesized data into database with data_source as 'synthetic'
            for example in valid_examples:
                cursor.execute('''
                    INSERT INTO prompt_training_data (id, prompt_id, inputs, outputs, data_source, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (
                    str(uuid.uuid4()),
                    prompt_id,
                    json.dumps(example['inputs']),
                    json.dumps(example.get('outputs', {})),
                    'synthetic',
                    now,
                    now
                ))
            
            conn.commit()
            conn.close()
            
            logger.info(f"Successfully generated {len(valid_examples)} synthesized training examples for '{prompt_name}' in database")
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse generated JSON: {e}")
            logger.debug(f"Generated content: {generated_content}")
        except Exception as e:
            logger.error(f"Error processing generated examples: {e}")
            
    except Exception as e:
        logger.error(f"Failed to synthesize training data for prompt '{prompt_name}': {e}")
        import traceback
        traceback.print_exc()
            
