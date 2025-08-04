#!/usr/bin/env python3
"""
Agent Evolve CLI - Command line interface for the tracking package
"""
import argparse
import sys
import os
import sqlite3
import json
from datetime import datetime
from pathlib import Path
from typing import Optional
from .extract_decorated_tools import DecoratedToolExtractor
from .extract_commented_evolve import extract_commented_evolve_from_file
from .generate_training_data import TrainingDataGenerator
from .config import DEFAULT_DB_PATH

def ensure_db_directory(db_path: str):
    """Ensure the database directory exists."""
    db_dir = os.path.dirname(db_path)
    if db_dir and not os.path.exists(db_dir):
        os.makedirs(db_dir, exist_ok=True)
        print(f"📁 Created directory: {db_dir}")

def analyze_tracking_data(db_path: str, thread_id: Optional[str] = None):
    """Analyze tracking data from the database."""
    ensure_db_directory(db_path)
    if not os.path.exists(db_path):
        print(f"❌ Database not found: {db_path}")
        return
    
    conn = sqlite3.connect(db_path)
    try:
        # Get messages
        query = "SELECT * FROM tracked_messages"
        params = []
        if thread_id:
            query += " WHERE thread_id = ?"
            params.append(thread_id)
        query += " ORDER BY timestamp DESC"
        
        cursor = conn.execute(query, params)
        messages = cursor.fetchall()
        
        # Get operations
        query = "SELECT * FROM tracked_operations"
        params = []
        if thread_id:
            query += " WHERE thread_id = ?"
            params.append(thread_id)
        query += " ORDER BY started_at DESC"
        
        cursor = conn.execute(query, params)
        operations = cursor.fetchall()
        
        print(f"📊 Tracking Data Analysis")
        print(f"Database: {db_path}")
        if thread_id:
            print(f"Thread ID: {thread_id}")
        print("-" * 50)
        
        print(f"📨 Messages: {len(messages)}")
        for msg in messages[:5]:  # Show first 5
            print(f"  - {msg[2]} ({msg[1][:8]}...): {str(msg[3])[:50]}...")
        
        print(f"\n⚙️  Operations: {len(operations)}")
        for op in operations[:5]:  # Show first 5
            duration = f"{op[6]:.1f}ms" if op[6] else "N/A"
            print(f"  - {op[2]} ({op[3]}): {duration}")
        
        # Show thread IDs
        cursor = conn.execute("SELECT DISTINCT thread_id FROM tracked_messages")
        thread_ids = [row[0] for row in cursor.fetchall()]
        print(f"\n🧵 Thread IDs: {len(thread_ids)}")
        for tid in thread_ids[:3]:
            print(f"  - {tid}")
            
    finally:
        conn.close()

def export_data(db_path: str, output_file: str, format: str = 'json'):
    """Export tracking data to file."""
    ensure_db_directory(db_path)
    if not os.path.exists(db_path):
        print(f"❌ Database not found: {db_path}")
        return
    
    conn = sqlite3.connect(db_path)
    try:
        # Get all data
        cursor = conn.execute("SELECT * FROM tracked_messages ORDER BY timestamp")
        messages = cursor.fetchall()
        
        cursor = conn.execute("SELECT * FROM tracked_operations ORDER BY started_at")
        operations = cursor.fetchall()
        
        data = {
            'messages': [
                {
                    'id': row[0], 'thread_id': row[1], 'role': row[2], 
                    'content': row[3], 'timestamp': row[4], 'user_id': row[5],
                    'metadata': json.loads(row[6]) if row[6] else {}, 'parent_id': row[7]
                } for row in messages
            ],
            'operations': [
                {
                    'id': row[0], 'thread_id': row[1], 'operation_name': row[2],
                    'status': row[3], 'started_at': row[4], 'ended_at': row[5],
                    'duration_ms': row[6], 'input_data': json.loads(row[7]) if row[7] else None,
                    'output_data': json.loads(row[8]) if row[8] else None, 'error': row[9],
                    'metadata': json.loads(row[10]) if row[10] else {}
                } for row in operations
            ],
            'exported_at': datetime.utcnow().isoformat()
        }
        
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"✅ Exported {len(messages)} messages and {len(operations)} operations to {output_file}")
        
    finally:
        conn.close()

def clear_data(db_path: str, confirm: bool = False):
    """Clear all tracking data."""
    ensure_db_directory(db_path)
    if not os.path.exists(db_path):
        print(f"❌ Database not found: {db_path}")
        return
    
    if not confirm:
        response = input("⚠️  This will delete ALL tracking data. Type 'yes' to confirm: ")
        if response.lower() != 'yes':
            print("❌ Operation cancelled")
            return
    
    conn = sqlite3.connect(db_path)
    try:
        conn.execute("DELETE FROM tracked_messages")
        conn.execute("DELETE FROM tracked_operations")
        conn.commit()
        print("✅ All tracking data cleared")
    finally:
        conn.close()

def extract_evolution_targets(search_path: str, output_dir: str):
    """Extract functions and prompts marked with @evolve() decorator."""
    search_path = Path(search_path).resolve()
    output_dir_path = Path(output_dir).resolve()
    
    print(f"🔍 Searching for @evolve decorated functions in: {search_path}")
    print(f"📁 Output directory: {output_dir_path}")
    
    # Create .agent_evolve directory if it doesn't exist
    output_dir_path.mkdir(parents=True, exist_ok=True)
    
    # Add to .gitignore if not already present
    gitignore_path = Path(".gitignore")
    gitignore_entry = f"{output_dir}/"
    
    if gitignore_path.exists():
        with open(gitignore_path, 'r') as f:
            content = f.read()
        if gitignore_entry not in content:
            with open(gitignore_path, 'a') as f:
                if not content.endswith('\n'):
                    f.write('\n')
                f.write(f"{gitignore_entry}\n")
            print(f"✅ Added {gitignore_entry} to .gitignore")
    else:
        with open(gitignore_path, 'w') as f:
            f.write(f"{gitignore_entry}\n")
        print(f"✅ Created .gitignore with {gitignore_entry}")
    
    # Initialize extractor
    extractor = DecoratedToolExtractor(output_dir=str(output_dir_path))
    total_extracted = 0
    
    # Find all Python files
    if search_path.is_file() and search_path.suffix == '.py':
        python_files = [search_path]
    else:
        python_files = list(search_path.rglob("*.py"))
    
    print(f"📊 Found {len(python_files)} Python files to scan")
    
    for py_file in python_files:
        try:
            relative_path = py_file.relative_to(search_path) if py_file != search_path else py_file.name
            
            # Extract @evolve decorated functions
            decorated_tools = extractor.extract_from_file(str(py_file))
            
            # Save extracted decorated tools
            for tool in decorated_tools:
                extractor.save_extracted_tool(tool)
            
            # Extract commented #@evolve() targets
            commented_tools = extract_commented_evolve_from_file(str(py_file), str(output_dir_path))
            
            file_total = len(decorated_tools) + len(commented_tools)
            total_extracted += file_total
            
            if file_total > 0:
                print(f"✅ {relative_path}: Found {file_total} targets")
                for tool in decorated_tools:
                    print(f"   • {tool.get('name', 'Unknown')}")
                for tool_name in commented_tools:
                    print(f"   • {tool_name}")
                
        except Exception as e:
            print(f"❌ {relative_path}: {e}")
    
    if total_extracted > 0:
        print(f"\n🎉 Successfully extracted {total_extracted} evolution targets to {output_dir_path}")
        print(f"📋 You can now run evolution experiments on these targets")
    else:
        print(f"\n⚠️  No evolution targets found in {search_path}")
        print(f"💡 Make sure functions are decorated with @evolve() or have #@evolve() comments")

def generate_training_data(tools_directory: str, num_samples: int, force: bool):
    """Generate training data for extracted tools."""
    # Check for OpenAI API key
    if not os.getenv('OPENAI_API_KEY'):
        print("❌ Error: OPENAI_API_KEY environment variable is required")
        print("💡 Please set your OpenAI API key: export OPENAI_API_KEY=your_key_here")
        return
    
    # Validate num_samples
    if num_samples < 1:
        print("❌ Error: Number of samples must be at least 1")
        return
    
    if num_samples > 100:
        print("⚠️  Warning: Generating more than 100 samples per tool may be expensive and slow")
        response = input("Continue? (y/N): ")
        if response.lower() != 'y':
            return
    
    # Initialize generator
    generator = TrainingDataGenerator(num_samples=num_samples)
    
    # Generate training data
    generator.generate_training_data(tools_directory, force=force)

def main():
    parser = argparse.ArgumentParser(
        description="Agent Evolve CLI - Tracking data management",
        prog="python -m agent_evolve"
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Analyze command
    analyze_parser = subparsers.add_parser('analyze', help='Analyze tracking data')
    analyze_parser.add_argument('--db', default=DEFAULT_DB_PATH, 
                               help=f'Database path (default: {DEFAULT_DB_PATH})')
    analyze_parser.add_argument('--thread-id', help='Filter by thread ID')
    
    # Export command
    export_parser = subparsers.add_parser('export', help='Export tracking data')
    export_parser.add_argument('--db', default=DEFAULT_DB_PATH,
                              help=f'Database path (default: {DEFAULT_DB_PATH})')
    export_parser.add_argument('--output', required=True, help='Output file path')
    export_parser.add_argument('--format', choices=['json'], default='json',
                              help='Export format (default: json)')
    
    # Clear command
    clear_parser = subparsers.add_parser('clear', help='Clear all tracking data')
    clear_parser.add_argument('--db', default=DEFAULT_DB_PATH,
                             help=f'Database path (default: {DEFAULT_DB_PATH})')
    clear_parser.add_argument('--yes', action='store_true',
                             help='Skip confirmation prompt')
    
    # Status command
    status_parser = subparsers.add_parser('status', help='Show tracking status')
    status_parser.add_argument('--db', default=DEFAULT_DB_PATH,
                              help=f'Database path (default: {DEFAULT_DB_PATH})')
    
    # Extract command
    extract_parser = subparsers.add_parser('extract', help='Extract functions and prompts marked with @evolve()')
    extract_parser.add_argument('--path', default='.', help='Path to search for decorated functions (default: current directory)')
    extract_parser.add_argument('--output-dir', default='.agent_evolve', help='Output directory name (default: .agent_evolve)')
    
    # Generate training data command
    train_parser = subparsers.add_parser('generate-training-data', help='Generate training data for extracted tools')
    train_parser.add_argument('tools_directory', nargs='?', default='.agent_evolve', 
                             help='Directory containing tool subdirectories (default: .agent_evolve)')
    train_parser.add_argument('--num-samples', '-n', type=int, default=10,
                             help='Number of training samples to generate per tool (default: 10)')
    train_parser.add_argument('--force', '-f', action='store_true',
                             help='Force regeneration of existing training data')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    if args.command == 'analyze':
        analyze_tracking_data(args.db, args.thread_id)
    elif args.command == 'export':
        export_data(args.db, args.output, args.format)
    elif args.command == 'clear':
        clear_data(args.db, args.yes)
    elif args.command == 'status':
        analyze_tracking_data(args.db)
    elif args.command == 'extract':
        extract_evolution_targets(args.path, args.output_dir)
    elif args.command == 'generate-training-data':
        generate_training_data(args.tools_directory, args.num_samples, args.force)

if __name__ == '__main__':
    main()