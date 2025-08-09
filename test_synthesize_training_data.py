#!/usr/bin/env python3
"""
Test script for synthesize_additional_training_data function
"""

import os
import sys
import sqlite3
import json
import uuid
from datetime import datetime

# Add the agent_evolve package to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'agent_evolve'))

from agent_evolve.training_data_generator import synthesize_additional_training_data

def create_test_database():
    """Create a test database with sample prompt and usage data."""
    db_path = "test_synthesize.db"
    
    # Remove existing test database
    if os.path.exists(db_path):
        os.remove(db_path)
    
    conn = sqlite3.connect(db_path)
    
    # Create prompts table
    conn.execute('''
        CREATE TABLE IF NOT EXISTS prompts (
            id TEXT PRIMARY KEY,
            prompt_name TEXT NOT NULL,
            prompt_type TEXT NOT NULL,
            definition_location TEXT NOT NULL,
            full_code TEXT NOT NULL,
            content TEXT NOT NULL,
            variables JSON,
            function_signature TEXT,
            enum_values JSON,
            created_at TEXT NOT NULL,
            last_seen TEXT NOT NULL,
            usage_count INTEGER DEFAULT 0
        )
    ''')
    
    # Create prompt_usages table
    conn.execute('''
        CREATE TABLE IF NOT EXISTS prompt_usages (
            id TEXT PRIMARY KEY,
            trace_id TEXT NOT NULL,
            prompt_id TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            variable_values JSON,
            rendered_content TEXT
        )
    ''')
    
    # Insert test prompt
    prompt_id = str(uuid.uuid4())
    now = datetime.utcnow().isoformat()
    
    test_prompt = {
        'id': prompt_id,
        'prompt_name': 'test_prompt',
        'prompt_type': 'template',
        'definition_location': 'test_file.py:test_prompt',
        'full_code': 'test_prompt = "You are a helpful assistant. Please help with: {topic}"',
        'content': 'You are a helpful assistant. Please help with: {topic}',
        'variables': json.dumps({'topic': 'string'}),
        'function_signature': None,
        'enum_values': None,
        'created_at': now,
        'last_seen': now,
        'usage_count': 3
    }
    
    conn.execute('''
        INSERT INTO prompts 
        (id, prompt_name, prompt_type, definition_location, full_code, content, 
         variables, function_signature, enum_values, created_at, last_seen, usage_count)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        test_prompt['id'], test_prompt['prompt_name'], test_prompt['prompt_type'],
        test_prompt['definition_location'], test_prompt['full_code'], test_prompt['content'],
        test_prompt['variables'], test_prompt['function_signature'], test_prompt['enum_values'],
        test_prompt['created_at'], test_prompt['last_seen'], test_prompt['usage_count']
    ))
    
    # Insert test usage examples
    usage_examples = [
        {
            'variable_values': json.dumps({'topic': 'Python programming'}),
            'rendered_content': 'You are a helpful assistant. Please help with: Python programming'
        },
        {
            'variable_values': json.dumps({'topic': 'machine learning'}),
            'rendered_content': 'You are a helpful assistant. Please help with: machine learning'
        },
        {
            'variable_values': json.dumps({'topic': 'web development'}),
            'rendered_content': 'You are a helpful assistant. Please help with: web development'
        }
    ]
    
    for i, example in enumerate(usage_examples):
        usage_id = str(uuid.uuid4())
        trace_id = str(uuid.uuid4())
        
        conn.execute('''
            INSERT INTO prompt_usages 
            (id, trace_id, prompt_id, timestamp, variable_values, rendered_content)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            usage_id, trace_id, prompt_id, now,
            example['variable_values'], example['rendered_content']
        ))
    
    conn.commit()
    conn.close()
    
    return db_path, prompt_id

def test_synthesize_function():
    """Test the synthesize_additional_training_data function."""
    print("Creating test database...")
    db_path, prompt_id = create_test_database()
    
    print(f"Test database created: {db_path}")
    print(f"Test prompt ID: {prompt_id}")
    
    print("\nTesting synthesize_additional_training_data function...")
    try:
        synthesize_additional_training_data(
            prompt_id=prompt_id,
            prompt_name="test_prompt",
            db_path=db_path,
            num_samples=5
        )
        print("✅ Function executed successfully!")
        
        # Check if output file was created
        output_dir = ".agent_evolve/test_prompt"
        output_file = os.path.join(output_dir, "synthesized_training_data.json")
        
        if os.path.exists(output_file):
            print(f"✅ Output file created: {output_file}")
            
            # Read and display the generated data
            with open(output_file, 'r') as f:
                data = json.load(f)
                print(f"✅ Generated {len(data)} training examples")
                print("Sample generated data:")
                print(json.dumps(data[:2], indent=2))
        else:
            print("❌ Output file not found")
            
    except Exception as e:
        print(f"❌ Function failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Clean up
    if os.path.exists(db_path):
        os.remove(db_path)
        print(f"Cleaned up test database: {db_path}")

if __name__ == "__main__":
    test_synthesize_function() 