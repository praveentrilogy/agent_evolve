"""
Sample training data from trace_events table.
"""

import sqlite3
import json
import uuid
from datetime import datetime
import logging
import argparse
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def sample_training_data(db_path, count):
    """Sample training data from trace_events table."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Get prompts marked for evolution
    cursor.execute("SELECT prompt_id FROM prompt_evolve_queue WHERE status = 'queued'")
    prompt_ids_for_evolution = [row[0] for row in cursor.fetchall()]

    if not prompt_ids_for_evolution:
        logger.info("No prompts marked for evolution. Nothing to sample.")
        return

    # Get a random sample of trace events for the prompts marked for evolution
    cursor.execute(
        f"SELECT used_prompt_id, args FROM trace_events WHERE event_type = 'function_call' AND used_prompt_id IN ({','.join(['?']*len(prompt_ids_for_evolution))}) ORDER BY RANDOM() LIMIT ?",
        (*prompt_ids_for_evolution, count)
    )
    prompt_events = cursor.fetchall()

    prompt_samples = 0
    for prompt_id, args in prompt_events:
        try:
            inputs = json.loads(args)
            inputs_str = json.dumps(inputs, sort_keys=True)
            
            # Check for existing entry
            cursor.execute(
                "SELECT COUNT(*) FROM prompt_training_data WHERE prompt_id = ? AND inputs = ?",
                (prompt_id, inputs_str)
            )
            if cursor.fetchone()[0] > 0:
                logger.info(f"Skipping duplicate prompt training data for prompt_id {prompt_id} and inputs {inputs_str[:50]}...")
                continue

            now = datetime.utcnow().isoformat()
            cursor.execute(
                "INSERT INTO prompt_training_data (id, prompt_id, inputs, outputs, data_source, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (str(uuid.uuid4()), prompt_id, inputs_str, '{}', 'sampled', now, now)
            )
            prompt_samples += 1
        except Exception as e:
            logger.error(f"Error processing prompt event: {e}")

    code_samples = 0
    for function_name, args in code_events:
        try:
            # Get the function_id from the functions table
            cursor.execute("SELECT id FROM functions WHERE full_function_name = ?", (function_name,))
            function_id = cursor.fetchone()[0]

            inputs = json.loads(args)
            inputs_str = json.dumps(inputs, sort_keys=True)

            # Check for existing entry
            cursor.execute(
                "SELECT COUNT(*) FROM code_training_data WHERE function_id = ? AND inputs = ?",
                (function_id, inputs_str)
            )
            if cursor.fetchone()[0] > 0:
                logger.info(f"Skipping duplicate code training data for function_id {function_id} and inputs {inputs_str[:50]}...")
                continue

            now = datetime.utcnow().isoformat()
            cursor.execute(
                "INSERT INTO code_training_data (id, function_id, inputs, outputs, data_source, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (str(uuid.uuid4()), function_id, inputs_str, '{}', 'sampled', now, now)
            )
            code_samples += 1
        except Exception as e:
            logger.error(f"Error processing code event for function {function_name}: {e}")

    conn.commit()
    conn.close()

    logger.info(f"Successfully sampled {prompt_samples} prompt training data samples.")
    logger.info(f"Successfully sampled {code_samples} code training data samples.")

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Sample training data from trace_events table.")
    parser.add_argument("--db_path", default="data/graph.db", help="Path to the database.")
    parser.add_argument("--count", type=int, default=10, help="Number of samples to create.")
    args = parser.parse_args()

    sample_training_data(args.db_path, args.count)

if __name__ == "__main__":
    main()
