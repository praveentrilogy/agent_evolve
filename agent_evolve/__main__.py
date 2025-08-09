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
import subprocess

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
        "path", type=str, help="Path to the code to extract from"
    )
    # extract_parser.set_defaults(func=extract_decorated_tools)

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
        "tool_name", type=str, help="Name of the tool/function to evolve"
    )
    # evolve_parser.set_defaults(func=run_openevolve)

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
    generate_evaluator_cli_parser.set_defaults(func=generate_evaluator_cli)

    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)

    args = parser.parse_args()

    if args.command == "daemon":
        if args.db_path:
            os.environ['AGENT_EVOLVE_DB_PATH'] = args.db_path
        if args.app_start_command:
            os.environ['APP_START_COMMAND'] = args.app_start_command
        if args.project_root:
            os.environ['AGENT_EVOLVE_PROJECT_ROOT'] = args.project_root
        evolve_daemon_main()
    elif hasattr(args, "func"):
        # Filter out args that are not part of the function's signature
        func_args = {k: v for k, v in vars(args).items() if k in args.func.__code__.co_varnames}
        args.func(**func_args)
    else:
        parser.print_help(sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()