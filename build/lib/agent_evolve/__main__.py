#!/usr/bin/env python3
"""
Agent Evolve CLI - Command line interface for the tracking package
"""
import argparse
import sys
import os

# from agent_evolve.evolve_decorator import evolve
# from agent_evolve.extract_decorated_tools import extract_decorated_tools
# from agent_evolve.generate_training_data import generate_training_data
# from agent_evolve.generate_evaluators import generate_evaluators
# from agent_evolve.generate_openevolve_configs import generate_openevolve_configs
# from agent_evolve.run_openevolve import run_openevolve
# Dashboard is now a Streamlit app, no app object to import
# from agent_evolve.safe_tracer import enable_safe_tracing, analyze_traces
from agent_evolve.trace_tracer import enable_trace_tracing
from agent_evolve.evolve_daemon import main as evolve_daemon_main
from agent_evolve.evaluator_engine import generate_evaluator_cli, auto_generate_evaluator_for_extracted_tool
from agent_evolve.extract_decorated_tools import DecoratedToolExtractor
from agent_evolve.extract_commented_evolve import extract_commented_evolve_from_file
import subprocess
import sqlite3
import json
import uuid
from datetime import datetime
import asyncio
import logging

logger = logging.getLogger(__name__)

def sample_training_data_cli(db_path='data/graph.db', count=10):
    """Sample training data from trace_events table and create training data entries."""
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        print("🔍 Sampling training data from trace events...")
        
        # Get all prompts that have been used (have trace events)
        cursor.execute("""
            SELECT DISTINCT p.id, p.prompt_name, COUNT(te.id) as usage_count, 'prompt' as item_type
            FROM prompts p
            JOIN trace_events te ON te.used_prompt_id = p.id
            WHERE te.event_type = 'function_call'
            GROUP BY p.id, p.prompt_name
            ORDER BY usage_count DESC
        """)
        
        prompts_with_usage = cursor.fetchall()
        
        # Get all functions that have been used (have trace events)
        cursor.execute("""
            SELECT DISTINCT f.id, f.full_function_name, COUNT(te.id) as usage_count, 'code' as item_type
            FROM functions f
            JOIN trace_events te ON te.function_name = f.full_function_name
            WHERE te.event_type = 'function_call'
            GROUP BY f.id, f.full_function_name
            ORDER BY usage_count DESC
        """)
        
        functions_with_usage = cursor.fetchall()
        
        # Combine prompts and functions
        all_items = prompts_with_usage + functions_with_usage
        
        if not all_items:
            print("No prompts or functions with trace events found.")
            conn.close()
            return
            
        print(f"\n📊 Found {len(prompts_with_usage)} prompts and {len(functions_with_usage)} functions with trace events:")
        if prompts_with_usage:
            print("\n📝 Prompts:")
            for item_id, item_name, usage_count, _ in prompts_with_usage:
                print(f"  • {item_name}: {usage_count} trace events")
        if functions_with_usage:
            print("\n🔧 Functions:")
            for item_id, item_name, usage_count, _ in functions_with_usage:
                print(f"  • {item_name}: {usage_count} trace events")
        
        # Sample trace events for each item
        total_samples_created = 0
        samples_per_item = max(1, count // len(all_items))
        
        for item_id, item_name, _, item_type in all_items:
            if item_type == 'prompt':
                # Get random sample of trace events for this prompt
                cursor.execute("""
                    SELECT args, result 
                    FROM trace_events 
                    WHERE event_type = 'function_call' 
                    AND used_prompt_id = ? 
                    ORDER BY RANDOM() 
                    LIMIT ?
                """, (item_id, samples_per_item))
            else:  # code/function
                # Get random sample of trace events for this function
                cursor.execute("""
                    SELECT args, result 
                    FROM trace_events 
                    WHERE event_type = 'function_call' 
                    AND function_name = ? 
                    ORDER BY RANDOM() 
                    LIMIT ?
                """, (item_name, samples_per_item))
            
            events = cursor.fetchall()
            samples_created = 0
            
            for (args_json, result_json) in events:
                try:
                    inputs = json.loads(args_json) if args_json else {}
                    outputs = json.loads(result_json) if result_json else {}
                    
                    inputs_str = json.dumps(inputs, sort_keys=True)
                    outputs_str = json.dumps(outputs, sort_keys=True)
                    
                    if item_type == 'prompt':
                        # Check for existing entry in prompt_training_data
                        cursor.execute("""
                            SELECT COUNT(*) 
                            FROM prompt_training_data 
                            WHERE prompt_id = ? AND inputs = ?
                        """, (item_id, inputs_str))
                        
                        if cursor.fetchone()[0] > 0:
                            continue
                        
                        # Insert new prompt training data
                        now = datetime.utcnow().isoformat()
                        cursor.execute("""
                            INSERT INTO prompt_training_data 
                            (id, prompt_id, inputs, outputs, data_source, created_at, updated_at)
                            VALUES (?, ?, ?, ?, ?, ?, ?)
                        """, (
                            str(uuid.uuid4()),
                            item_id,
                            inputs_str,
                            outputs_str,
                            'real',
                            now,
                            now
                        ))
                    else:  # code/function
                        # Check for existing entry in code_training_data
                        cursor.execute("""
                            SELECT COUNT(*) 
                            FROM code_training_data 
                            WHERE function_id = ? AND inputs = ?
                        """, (item_id, inputs_str))
                        
                        if cursor.fetchone()[0] > 0:
                            continue
                        
                        # Insert new code training data
                        now = datetime.utcnow().isoformat()
                        cursor.execute("""
                            INSERT INTO code_training_data 
                            (id, function_id, inputs, outputs, data_source, created_at, updated_at)
                            VALUES (?, ?, ?, ?, ?, ?, ?)
                        """, (
                            str(uuid.uuid4()),
                            item_id,
                            inputs_str,
                            outputs_str,
                            'real',
                            now,
                            now
                        ))
                    
                    samples_created += 1
                    total_samples_created += 1
                    
                except Exception as e:
                    print(f"  ⚠️  Error processing event: {e}")
                    continue
            
            if samples_created > 0:
                print(f"\n✅ Created {samples_created} training samples for {item_name}")
        
        conn.commit()
        conn.close()
        
        print(f"\n🎉 Total: Created {total_samples_created} new training data samples")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

def launch_dashboard(port=8501):
    """Launch the Streamlit dashboard."""
    import pkg_resources
    
    # Get the path to the dashboard.py file in the package
    dashboard_path = pkg_resources.resource_filename('agent_evolve', 'dashboard.py')
    
    print(f"Launching Agent Evolve dashboard on port {port}...")
    print(f"Dashboard will be available at: http://localhost:{port}")
    
    # Launch streamlit
    try:
        subprocess.run([
            "streamlit", "run", dashboard_path, 
            "--server.port", str(port),
            "--server.headless", "false"
        ], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error launching dashboard: {e}")
        print("Make sure Streamlit is installed: pip install streamlit")
    except FileNotFoundError:
        print("Error: streamlit command not found")
        print("Please install Streamlit: pip install streamlit")

def extract_decorated_tools_cli(path: str = '.', output_dir: str = '.agent_evolve'):
    """Extract functions marked with @evolve decorator (both regular and commented)"""
    try:
        # First extract regular @evolve decorators
        extractor = DecoratedToolExtractor(output_dir)
        regular_tools = []
        
        if os.path.isfile(path):
            tools = extractor.extract_from_file(path)
            for tool in tools:
                extractor.save_extracted_tool(tool)
                regular_tools.append(tool['name'])
        elif os.path.isdir(path):
            extractor.extract_from_directory(path)
            regular_tools = [tool['name'] for tool in extractor.extracted_tools]
        else:
            print(f"Error: {path} is not a valid file or directory")
            return
        
        # Then extract commented #@evolve decorators
        commented_tools = []
        if os.path.isfile(path):
            # Extract from single file
            extracted = extract_commented_evolve_from_file(path, output_dir)
            commented_tools.extend(extracted)
        elif os.path.isdir(path):
            # Extract from all Python files in directory
            from pathlib import Path
            for py_file in Path(path).rglob("*.py"):
                extracted = extract_commented_evolve_from_file(str(py_file), output_dir)
                commented_tools.extend(extracted)
        
        # Report results
        total_tools = len(regular_tools) + len(commented_tools)
        if total_tools > 0:
            print(f"\n✅ Extracted {total_tools} tools:")
            if regular_tools:
                print(f"  - {len(regular_tools)} with @evolve decorators")
            if commented_tools:
                print(f"  - {len(commented_tools)} with #@evolve decorators")
            print(f"\nExtracted tools saved to: {output_dir}")
        else:
            print(f"\n⚠️  No tools with @evolve or #@evolve decorators found in {path}")
        
    except Exception as e:
        print(f"Error extracting tools: {e}")
        import traceback
        traceback.print_exc()

def generate_evaluator_openevolve_cli(target_name: str, db_path: str = 'data/graph.db', project_root: str = os.getcwd()):
    """CLI function to generate an OpenEvolve-format evaluator for extracted tools."""
    logger.info(f"Generating OpenEvolve evaluator for '{target_name}'...")
    
    # Look for the extracted tool directory
    tool_dir_path = os.path.join(project_root, ".agent_evolve", target_name)
    
    if not os.path.exists(tool_dir_path):
        logger.error(f"Error: Tool directory not found for '{target_name}' at {tool_dir_path}")
        logger.error(f"Please run 'agent-evolve extract' first to extract the tool.")
        return
    
    # Generate the evaluator using the OpenEvolve approach
    success = auto_generate_evaluator_for_extracted_tool(
        tool_dir_path=tool_dir_path,
        project_root=project_root
    )
    
    if success:
        evaluator_path = os.path.join(tool_dir_path, "evaluator.py")
        logger.info(f"✅ Successfully generated OpenEvolve evaluator at {evaluator_path}")
        print(f"✅ Evaluator generated for '{target_name}'")
        print(f"   Location: {evaluator_path}")
        print(f"   Format: OpenEvolve (supports evaluate(program_path) signature)")
    else:
        logger.error(f"❌ Failed to generate evaluator for '{target_name}'")
        print(f"❌ Failed to generate evaluator for '{target_name}'")

def run_evolve_command(target_name: str, db_path: str = 'data/graph.db', project_root: str = os.getcwd(), checkpoint: int = 0, iterations: int = None):
    """Run OpenEvolve on a specific target (prompt or function)"""
    from agent_evolve.run_openevolve import run_openevolve_for_tool
    
    logger.info(f"Starting evolution for '{target_name}'...")
    
    # Determine the evaluator directory path
    evaluator_dir = os.path.join(project_root, ".agent_evolve", target_name)
    
    if not os.path.exists(evaluator_dir):
        logger.error(f"Error: No evaluator directory found for '{target_name}' at {evaluator_dir}")
        logger.error(f"Please run 'agent-evolve generate-evaluator {target_name}' first.")
        return
    
    # Check for required files
    required_files = {
        'evaluator.py': os.path.join(evaluator_dir, 'evaluator.py'),
        'openevolve_config.yaml': os.path.join(evaluator_dir, 'openevolve_config.yaml'),
        'evolve_target.py': os.path.join(evaluator_dir, 'evolve_target.py')
    }
    
    missing_files = []
    for file_name, file_path in required_files.items():
        if not os.path.exists(file_path):
            missing_files.append(file_name)
    
    if missing_files:
        logger.error(f"Error: Missing required files in {evaluator_dir}:")
        for file_name in missing_files:
            logger.error(f"  - {file_name}")
        logger.error(f"\nPlease ensure all required files are generated.")
        return
    
    # Run OpenEvolve asynchronously
    try:
        success = asyncio.run(run_openevolve_for_tool(evaluator_dir, checkpoint, iterations))
        if success:
            logger.info(f"Evolution completed successfully for '{target_name}'!")
        else:
            logger.error(f"Evolution failed for '{target_name}'.")
    except Exception as e:
        logger.error(f"Error running evolution: {e}")
        import traceback
        traceback.print_exc()

def main():
    parser = argparse.ArgumentParser(
        description="Agent Evolve CLI for managing AI agent evolution."
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Extract command
    extract_parser = subparsers.add_parser(
        "extract", help="Extract functions marked with @evolve decorator"
    )
    extract_parser.add_argument(
        "--path", type=str, default=".", help="Path to the code to extract from"
    )
    extract_parser.add_argument(
        "--output-dir", type=str, default=".agent_evolve", help="Output directory for extracted tools"
    )
    extract_parser.set_defaults(func=extract_decorated_tools_cli)

    # Generate Training Data command
    generate_training_data_parser = subparsers.add_parser(
        "generate-training-data", help="Generate training data for extracted functions"
    )
    generate_training_data_parser.add_argument(
        "tool_name", type=str, help="Name of the tool/function to generate data for"
    )
    # generate_training_data_parser.set_defaults(func=generate_training_data)

    # Generate Evaluators command
    generate_evaluators_parser = subparsers.add_parser(
        "generate-evaluators", help="Generate evaluation functions"
    )
    generate_evaluators_parser.add_argument(
        "tool_name", type=str, help="Name of the tool/function to generate evaluators for"
    )
    # generate_evaluators_parser.set_defaults(func=generate_evaluators)

    # Generate Configs command
    generate_configs_parser = subparsers.add_parser(
        "generate-configs", help="Generate OpenEvolve configuration files"
    )
    generate_configs_parser.add_argument(
        "tool_name", type=str, help="Name of the tool/function to generate configs for"
    )
    # generate_configs_parser.set_defaults(func=generate_openevolve_configs)

    # Evolve command
    evolve_parser = subparsers.add_parser(
        "evolve", help="Run evolution optimization on a specific tool"
    )
    evolve_parser.add_argument(
        "target_name", type=str, help="Name of the prompt or function to evolve (e.g., ORCHESTRATOR_PROMPT)"
    )
    evolve_parser.add_argument(
        "--db-path", type=str, default="data/graph.db", help="Path to the SQLite database file"
    )
    evolve_parser.add_argument(
        "--project-root", type=str, default=os.getcwd(), help="Root directory of the project"
    )
    evolve_parser.add_argument(
        "--checkpoint", type=int, default=0, help="Checkpoint number to resume from (0 for fresh start)"
    )
    evolve_parser.add_argument(
        "--iterations", type=int, help="Number of iterations to run (overrides config file)"
    )
    evolve_parser.set_defaults(func=run_evolve_command)

    # Pipeline command
    pipeline_parser = subparsers.add_parser(
        "pipeline", help="Run the complete pipeline (training data → evaluator → config → evolution)"
    )
    pipeline_parser.add_argument(
        "tool_name", type=str, help="Name of the tool/function to run the pipeline for"
    )
    # pipeline_parser.set_defaults(func=lambda tool_name: (
    #     generate_training_data(tool_name),
    #     generate_evaluators(tool_name),
    #     generate_openevolve_configs(tool_name),
    #     run_openevolve(tool_name)
    # ))

    # Dashboard command
    dashboard_parser = subparsers.add_parser(
        "dashboard", help="Launch the interactive Streamlit dashboard"
    )
    dashboard_parser.add_argument(
        "--port", type=int, default=8501, help="Port to run the dashboard on"
    )
    dashboard_parser.set_defaults(func=launch_dashboard)

    # Analyze command
    analyze_parser = subparsers.add_parser(
        "analyze", help="Analyze tracking data"
    )
    analyze_parser.add_argument(
        "--db-path", type=str, default="safe_traces.db", help="Path to the SQLite database file"
    )
    # analyze_parser.set_defaults(func=analyze_traces)

    # Trace command
    trace_parser = subparsers.add_parser(
        "trace", help="Enable trace-based auto-tracing using sys.settrace()"
    )
    trace_parser.add_argument(
        "--db-path", type=str, default="data/graph.db", help="Path to the SQLite database file"
    )
    trace_parser.add_argument(
        "--project-root", type=str, default=os.getcwd(), help="Root directory of project to trace"
    )
    trace_parser.add_argument(
        "--include-modules", nargs='*', help="Module patterns to include (e.g., my_app.module1 my_app.module2)"
    )
    trace_parser.set_defaults(func=enable_trace_tracing)

    # Daemon command
    daemon_parser = subparsers.add_parser(
        "daemon", help="Run the Agent Evolve daemon to process the evolution queue"
    )
    daemon_parser.add_argument(
        "--db-path", type=str, default="data/graph.db", help="Path to the SQLite database file"
    )
    daemon_parser.add_argument(
        "--app-start-command", type=str, help="Command to start the application server (e.g., 'python app.py')"
    )
    daemon_parser.add_argument(
        "--project-root", type=str, default=os.getcwd(), help="Root directory of the project being evolved"
    )
    daemon_parser.set_defaults(func=evolve_daemon_main)

    # Generate Evaluator CLI command
    generate_evaluator_cli_parser = subparsers.add_parser(
        "generate-evaluator", help="Generate an evaluator for a specific prompt or function"
    )
    generate_evaluator_cli_parser.add_argument(
        "target_name", type=str, help="Name of the prompt or function to generate evaluator for (e.g., ORCHESTRATOR_PROMPT)"
    )
    generate_evaluator_cli_parser.add_argument(
        "--db-path", type=str, default="data/graph.db", help="Path to the SQLite database file"
    )
    generate_evaluator_cli_parser.add_argument(
        "--project-root", type=str, default=os.getcwd(), help="Root directory of the project containing the target"
    )
    generate_evaluator_cli_parser.set_defaults(func=generate_evaluator_openevolve_cli)

    # Sample Training Data command
    sample_training_data_parser = subparsers.add_parser(
        "sample-training-data", help="Sample training data from trace_events table."
    )
    sample_training_data_parser.add_argument(
        "--db-path", type=str, default="data/graph.db", help="Path to the database."
    )
    sample_training_data_parser.add_argument(
        "--count", type=int, default=10, help="Number of samples to create."
    )
    sample_training_data_parser.set_defaults(func=sample_training_data_cli)

    args = parser.parse_args()

    if hasattr(args, "func"):
        # Filter out args that are not part of the function's signature
        func_args = {k: v for k, v in vars(args).items() if k in args.func.__code__.co_varnames}
        args.func(**func_args)
    else:
        parser.print_help(sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()