import sqlite3
import json
import logging

logger = logging.getLogger(__name__)

def load_training_data(db_path, prompt_id):
    """
    Loads training data for a given prompt from the SQLite database.
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.execute('''
        SELECT inputs, outputs, data_source
        FROM prompt_training_data
        WHERE prompt_id = ?
        ORDER BY created_at DESC
    ''', (prompt_id,))

    training_samples = []
    for row in cursor.fetchall():
        inputs, outputs, data_source = row
        training_samples.append({
            'inputs': json.loads(inputs) if inputs else {},
            'outputs': json.loads(outputs) if outputs else {},
            'data_source': data_source
        })
    
    conn.close()
    return training_samples
