#!/usr/bin/env python3
"""
One-off script to migrate training data from JSON files to database tables.
Reads all training data files and writes them to prompt_training_data or code_training_data tables.
Sets data_source to 'real' or 'synthetic' based on the filename.
"""

import os
import json
import sqlite3
import glob
import uuid
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def migrate_training_data(db_path='data/graph.db', base_dir=None):
    """Migrate all training data files to database."""
    
    if base_dir:
        original_dir = os.getcwd()
        os.chdir(base_dir)
    
    # Connect to database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Get all prompts to map names to IDs
    cursor.execute('SELECT id, prompt_name FROM prompts')
    prompt_map = {row[1]: row[0] for row in cursor.fetchall()}
    logger.info(f"Found {len(prompt_map)} prompts in database: {list(prompt_map.keys())}")
    
    # Counter for statistics
    stats = {
        'real_prompt': 0,
        'synthetic_prompt': 0,
        'real_code': 0,
        'synthetic_code': 0,
        'errors': 0
    }
    
    # Find all training data files
    patterns = [
        '.agent_evolve/*/real_training_data.json',
        '.agent_evolve/*/synthesized_training_data.json',
        '.agent_evolve/*/test_data.json',
        'training_data/*.json',
        'training_data_*.json',
        '*/training_data/*.json'
    ]
    
    all_files = []
    for pattern in patterns:
        found_files = glob.glob(pattern, recursive=True)
        all_files.extend(found_files)
        logger.info(f"Pattern '{pattern}' found {len(found_files)} files")
    
    logger.info(f"Found {len(all_files)} training data files to migrate")
    
    for file_path in all_files:
        try:
            logger.info(f"Processing: {file_path}")
            
            # Determine data source based on filename
            if 'real_training_data' in file_path:
                data_source = 'real'
            elif 'synthesized_training_data' in file_path:
                data_source = 'synthetic'
            else:
                # Default to synthetic if not clear from filename
                data_source = 'synthetic'
            
            # Try to extract prompt name from path
            prompt_name = None
            path_parts = file_path.split('/')
            
            # Check if it's in .agent_evolve/<prompt_name>/ structure
            if '.agent_evolve' in path_parts:
                idx = path_parts.index('.agent_evolve')
                if idx + 1 < len(path_parts):
                    prompt_name = path_parts[idx + 1]
            
            # Load the JSON file
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            # Handle different JSON structures
            samples = []
            if isinstance(data, list):
                samples = data
            elif isinstance(data, dict):
                if 'training_data' in data:
                    samples = data['training_data']
                elif 'samples' in data:
                    samples = data['samples']
                else:
                    samples = [data]
            
            # Process each sample
            for sample in samples:
                try:
                    # Extract inputs and outputs
                    inputs = sample.get('inputs', sample.get('input', sample.get('variable_values', {})))
                    outputs = sample.get('outputs', sample.get('output', sample.get('expected_output', sample.get('rendered_content', {}))))
                    
                    # Skip if no inputs
                    if not inputs:
                        continue
                    
                    # Convert to JSON strings if needed
                    if not isinstance(inputs, str):
                        inputs = json.dumps(inputs)
                    if not isinstance(outputs, str):
                        outputs = json.dumps(outputs) if outputs else '{}'
                    
                    # Try to find the prompt ID
                    prompt_id = None
                    if prompt_name and prompt_name in prompt_map:
                        prompt_id = prompt_map[prompt_name]
                    
                    if prompt_id:
                        # Insert into prompt_training_data
                        now = datetime.utcnow().isoformat()
                        cursor.execute('''
                            INSERT OR IGNORE INTO prompt_training_data 
                            (id, prompt_id, inputs, outputs, data_source, created_at, updated_at)
                            VALUES (?, ?, ?, ?, ?, ?, ?)
                        ''', (
                            str(uuid.uuid4()),
                            prompt_id,
                            inputs,
                            outputs,
                            data_source,
                            now,
                            now
                        ))
                        
                        if data_source == 'real':
                            stats['real_prompt'] += 1
                        else:
                            stats['synthetic_prompt'] += 1
                    else:
                        logger.warning(f"Could not find prompt ID for: {prompt_name or 'unknown'}")
                        stats['errors'] += 1
                        
                except Exception as e:
                    logger.error(f"Error processing sample in {file_path}: {e}")
                    stats['errors'] += 1
                    
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {e}")
            stats['errors'] += 1
    
    # Commit all changes
    conn.commit()
    conn.close()
    
    if base_dir:
        os.chdir(original_dir)
    
    # Print statistics
    logger.info("\n=== Migration Complete ===")
    logger.info(f"Real prompt training data: {stats['real_prompt']}")
    logger.info(f"Synthetic prompt training data: {stats['synthetic_prompt']}")
    logger.info(f"Real code training data: {stats['real_code']}")
    logger.info(f"Synthetic code training data: {stats['synthetic_code']}")
    logger.info(f"Total errors: {stats['errors']}")
    logger.info(f"Total migrated: {sum(stats[k] for k in stats if k != 'errors')}")

if __name__ == "__main__":
    import sys
    db_path = sys.argv[1] if len(sys.argv) > 1 else 'data/graph.db'
    base_dir = sys.argv[2] if len(sys.argv) > 2 else None
    migrate_training_data(db_path, base_dir)