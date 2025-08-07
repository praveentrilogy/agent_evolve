#!/usr/bin/env python3
"""
Agent Evolve CLI - Command line interface for the tracking package
"""
import argparse
import sys
import os
import sqlite3
import json
import asyncio
from datetime import datetime
from pathlib import Path
from typing import Optional
from .extract_decorated_tools import DecoratedToolExtractor
from .extract_commented_evolve import extract_commented_evolve_from_file
from .generate_training_data import TrainingDataGenerator
from .generate_evaluators import EvaluatorGenerator
from .generate_openevolve_configs import OpenEvolveConfigGenerator
from .run_openevolve import run_openevolve_for_tool, list_available_tools
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

def run_dashboard(base_dir: str = ".agent_evolve", port: int = 8501, host: str = "localhost"):
    """Launch the Streamlit dashboard for visualizing evolution results."""
    try:
        import streamlit.web.cli as stcli
        import sys
        from pathlib import Path
        
        # Get the dashboard script path
        dashboard_script = Path(__file__).parent / "dashboard.py"
        
        if not dashboard_script.exists():
            print("❌ Dashboard script not found")
            return
        
        print(f"🚀 Starting Agent Evolve Dashboard...")
        print(f"📁 Base directory: {base_dir}")
        print(f"🌐 Server: http://{host}:{port}")
        print(f"💡 Press Ctrl+C to stop the server")
        
        # Set environment variable for base directory
        import os
        os.environ['AGENT_EVOLVE_BASE_DIR'] = base_dir
        
        # Launch Streamlit
        sys.argv = ["streamlit", "run", str(dashboard_script), "--server.port", str(port), "--server.address", host]
        stcli.main()
        
    except ImportError:
        print("❌ Error: Streamlit is not installed")
        print("💡 Install it with: pip install streamlit plotly")
    except Exception as e:
        print(f"❌ Error starting dashboard: {e}")

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

def generate_training_data(tool_name: str, base_dir: str, num_samples: int, force: bool):
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
    
    if tool_name:
        # Generate for specific tool
        tool_directory = os.path.join(base_dir, tool_name)
        if not os.path.exists(tool_directory):
            print(f"❌ Error: Tool '{tool_name}' not found in {base_dir}")
            return
        
        # Check if evolve_target.py exists
        evolve_target = os.path.join(tool_directory, "evolve_target.py")
        if not os.path.exists(evolve_target):
            print(f"❌ Error: {tool_name}/evolve_target.py not found")
            print("💡 Make sure the tool was properly extracted first")
            return
        
        print(f"🎯 Generating training data for: {tool_name}")
        
        # Generate training data for single tool (pass the parent directory)
        generator.generate_training_data(base_dir, force=force, specific_tool=tool_name)
    else:
        # Generate for all tools
        print(f"🎯 Generating training data for all tools in: {base_dir}")
        generator.generate_training_data(base_dir, force=force)

def generate_evaluators(tools_directory: str, model_name: str = "gpt-4o"):
    """Generate evaluators for extracted tools."""
    # Check for OpenAI API key
    if not os.getenv('OPENAI_API_KEY'):
        print("❌ Error: OPENAI_API_KEY environment variable is required")
        print("💡 Please set your OpenAI API key: export OPENAI_API_KEY=your_key_here")
        return
    
    # Initialize generator with specified model
    generator = EvaluatorGenerator(model_name=model_name)
    
    # Generate evaluators
    generator.generate_evaluators(tools_directory)

def generate_openevolve_configs(tools_directory: str):
    """Generate OpenEvolve configuration files for extracted tools."""
    # Initialize generator
    generator = OpenEvolveConfigGenerator(tools_directory)
    
    # Generate configs
    try:
        generator.generate_configs()
    except Exception as e:
        print(f"❌ Error generating configs: {e}")
        import traceback
        traceback.print_exc()

async def run_openevolve(tool_name: str, checkpoint: int = None, list_tools: bool = False, base_dir: str = ".agent_evolve", iterations: int = None):
    """Run OpenEvolve for a specific tool by name."""
    if list_tools:
        list_available_tools(base_dir)
        return
    
    if not tool_name:
        print("❌ Error: Please provide a tool name")
        print("💡 Use --list to see available tools")
        return
    
    # Construct the full tool directory path
    tool_directory = os.path.join(base_dir, tool_name)
    
    # Check if the tool directory exists
    if not os.path.exists(tool_directory):
        print(f"❌ Error: Tool '{tool_name}' not found in {base_dir}")
        print("💡 Available tools:")
        list_available_tools(base_dir)
        return
    
    # Check for OpenAI API key
    if not os.getenv('OPENAI_API_KEY'):
        print("❌ Error: OPENAI_API_KEY environment variable is required")
        print("💡 Please set your OpenAI API key: export OPENAI_API_KEY=your_key_here")
        return
    
    print(f"🚀 Running OpenEvolve for tool: {tool_name}")
    print(f"📁 Tool path: {tool_directory}")
    
    # Run OpenEvolve for the specified tool
    if checkpoint is not None:
        success = await run_openevolve_for_tool(tool_directory, checkpoint, iterations)
    else:
        success = await run_openevolve_for_tool(tool_directory, iterations=iterations)
    
    if success:
        print(f"\n🎉 OpenEvolve optimization completed for {tool_name}!")
    else:
        print(f"\n❌ OpenEvolve optimization failed for {tool_name}!")

async def run_full_pipeline(tool_name: str, base_dir: str = ".agent_evolve", num_samples: int = 10, 
                           model_name: str = "gpt-4o", checkpoint: int = None, force_training_data: bool = False):
    """Run the complete pipeline for a tool: generate training data, evaluator, configs, then run OpenEvolve."""
    
    # Validate inputs
    if not tool_name:
        print("❌ Error: Please provide a tool name")
        return False
    
    # Check for OpenAI API key
    if not os.getenv('OPENAI_API_KEY'):
        print("❌ Error: OPENAI_API_KEY environment variable is required")
        print("💡 Please set your OpenAI API key: export OPENAI_API_KEY=your_key_here")
        return False
    
    # Construct the full tool directory path
    tool_directory = os.path.join(base_dir, tool_name)
    
    # Check if the tool directory exists
    if not os.path.exists(tool_directory):
        print(f"❌ Error: Tool '{tool_name}' not found in {base_dir}")
        print("💡 Available tools:")
        list_available_tools(base_dir)
        return False
    
    # Check if evolve_target.py exists
    evolve_target_path = os.path.join(tool_directory, "evolve_target.py")
    if not os.path.exists(evolve_target_path):
        print(f"❌ Error: {tool_name}/evolve_target.py not found")
        print("💡 Make sure the tool was properly extracted first")
        return False
    
    print(f"🚀 Running full pipeline for tool: {tool_name}")
    print(f"📁 Tool path: {tool_directory}")
    print(f"🔧 Model: {model_name}, Samples: {num_samples}")
    print("=" * 60)
    
    success = True
    
    # Step 1: Generate training data
    print(f"\n📊 Step 1/4: Generating training data...")
    try:
        generator = TrainingDataGenerator(num_samples=num_samples)
        generator.generate_training_data(base_dir, force=force_training_data)
        print("✅ Training data generation completed")
    except Exception as e:
        print(f"❌ Training data generation failed: {e}")
        success = False
    
    # Step 2: Generate evaluator (only if step 1 succeeded)
    if success:
        print(f"\n⚖️  Step 2/4: Generating evaluator...")
        try:
            evaluator_generator = EvaluatorGenerator(model_name=model_name)
            evaluator_generator.generate_evaluators(base_dir)
            print("✅ Evaluator generation completed")
        except Exception as e:
            print(f"❌ Evaluator generation failed: {e}")
            success = False
    
    # Step 3: Generate OpenEvolve config (only if step 2 succeeded)
    if success:
        print(f"\n⚙️  Step 3/4: Generating OpenEvolve config...")
        try:
            config_generator = OpenEvolveConfigGenerator(base_dir)
            config_generator.generate_configs()
            print("✅ Config generation completed")
        except Exception as e:
            print(f"❌ Config generation failed: {e}")
            success = False
    
    # Step 4: Run OpenEvolve optimization (only if all previous steps succeeded)
    if success:
        print(f"\n🧬 Step 4/5: Running OpenEvolve optimization...")
        try:
            if checkpoint is not None:
                evolve_success = await run_openevolve_for_tool(tool_directory, checkpoint)
            else:
                evolve_success = await run_openevolve_for_tool(tool_directory)
            if evolve_success:
                print("✅ OpenEvolve optimization completed")
            else:
                print("❌ OpenEvolve optimization failed")
                success = False
        except Exception as e:
            print(f"❌ OpenEvolve optimization failed: {e}")
            success = False
    
    # Step 5: Extract best version (only if step 4 succeeded)
    if success:
        print(f"\n🏆 Step 5/5: Extracting best evolved version...")
        try:
            # Extract best version for just this tool
            tool_path = Path(base_dir) / tool_name
            best_dir = tool_path / "openevolve_output" / "best"
            
            if best_dir.exists() and (best_dir / "best_program.py").exists():
                # Read the best program
                with open(best_dir / "best_program.py", 'r') as f:
                    best_code = f.read()
                
                # Read metadata if available
                metadata_file = best_dir / "best_program_info.json"
                metadata = {}
                if metadata_file.exists():
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)
                
                # Create the best_version.py file
                best_version_file = tool_path / "best_version.py"
                
                # Add header with evolution info
                header = f'''"""
Best Evolved Version of {tool_name}

Generated by OpenEvolve optimization
'''
                
                if metadata:
                    metrics = metadata.get('metrics', {})
                    if metrics:
                        header += f"""
Evolution Metrics:
"""
                        for metric, value in metrics.items():
                            header += f"- {metric}: {value:.4f}\n"
                    
                    generation = metadata.get('generation')
                    if generation is not None:
                        header += f"- Generation: {generation}\n"
                    
                    iteration = metadata.get('iteration')
                    if iteration is not None:
                        header += f"- Iteration: {iteration}\n"
                
                header += '"""\n\n'
                
                # Write the best version file
                with open(best_version_file, 'w') as f:
                    f.write(header + best_code)
                
                print("✅ Best version extracted to: best_version.py")
                if metadata.get('metrics'):
                    metrics_str = ", ".join([f"{k}={v:.3f}" for k, v in metadata['metrics'].items()])
                    print(f"   Metrics: {metrics_str}")
                
                # Generate score comparison
                print("📊 Generating score comparison...")
                try:
                    original_scores, best_scores = await _evaluate_both_versions(tool_path, tool_name)
                    
                    if original_scores and best_scores:
                        comparison_data = {
                            "tool_name": tool_name,
                            "evaluation_date": datetime.utcnow().isoformat(),
                            "original_version": {
                                "scores": original_scores,
                                "average": sum(original_scores.values()) / len(original_scores)
                            },
                            "best_version": {
                                "scores": best_scores,
                                "average": sum(best_scores.values()) / len(best_scores)
                            },
                            "improvements": {
                                metric: best_scores[metric] - original_scores.get(metric, 0)
                                for metric in best_scores.keys()
                            }
                        }
                        
                        # Save comparison
                        comparison_file = tool_path / "score_comparison.json"
                        with open(comparison_file, 'w') as f:
                            json.dump(comparison_data, f, indent=2)
                        
                        print("✅ Score comparison saved to: score_comparison.json")
                        
                        # Show improvement summary
                        avg_improvement = comparison_data["best_version"]["average"] - comparison_data["original_version"]["average"]
                        print(f"📈 Average improvement: {avg_improvement:+.3f}")
                        
                except Exception as e:
                    print(f"⚠️  Could not generate score comparison: {e}")
                    
            else:
                print("⚠️  No best version found to extract")
                
        except Exception as e:
            print(f"⚠️  Warning: Could not extract best version: {e}")
            # Don't fail the whole pipeline for this step
    
    # Final summary
    print("\n" + "=" * 60)
    if success:
        print(f"🎉 Full pipeline completed successfully for {tool_name}!")
        print(f"📈 Your optimized tool is ready in {tool_directory}/best_version.py")
        print(f"📁 Full evolution results: {tool_directory}/openevolve_output/")
    else:
        print(f"💥 Pipeline failed for {tool_name}")
        print(f"💡 Check the error messages above and try running individual steps")
    
    return success

async def _evaluate_both_versions(tool_path: Path, tool_name: str):
    """Get scores from OpenEvolve checkpoint data."""
    try:
        openevolve_output = tool_path / "openevolve_output"
        if not openevolve_output.exists():
            return None, None
        
        # Get original version scores (from initial checkpoint_0)
        original_scores = None
        checkpoint_0 = openevolve_output / "checkpoints" / "checkpoint_0" / "programs"
        if checkpoint_0.exists():
            # Find the original program (usually the parent/seed program)
            for program_file in checkpoint_0.glob("*.json"):
                with open(program_file, 'r') as f:
                    program_data = json.load(f)
                    if program_data.get('parent_id') is None or program_data.get('generation', 0) == 0:
                        # This is likely the original/seed program
                        original_scores = program_data.get('metrics', {})
                        break
        
        # Get best version scores from the best directory
        best_scores = None
        best_info_file = openevolve_output / "best" / "best_program_info.json"
        if best_info_file.exists():
            with open(best_info_file, 'r') as f:
                best_info = json.load(f)
                best_scores = best_info.get('metrics', {})
        
        return original_scores, best_scores
        
    except Exception as e:
        print(f"   Error reading scores from checkpoints: {e}")
        return None, None

async def extract_best_versions(base_dir: str = ".agent_evolve"):
    """Extract best evolved versions from OpenEvolve outputs and save as best_version.py."""
    base_path = Path(base_dir)
    
    if not base_path.exists():
        print(f"❌ Directory {base_dir} does not exist")
        return
    
    print(f"🔍 Scanning for evolved tools in: {base_path}")
    print("=" * 60)
    
    extracted_count = 0
    failed_count = 0
    
    # Find all tool directories
    for tool_dir in base_path.iterdir():
        if not tool_dir.is_dir():
            continue
        
        # Skip non-tool directories
        skip_dirs = {'db', 'data', '__pycache__', '.git', 'logs', 'output', 'checkpoints', 'temp', 'tmp'}
        if tool_dir.name in skip_dirs:
            continue
        
        # Check if this tool has OpenEvolve output
        openevolve_output = tool_dir / "openevolve_output"
        if not openevolve_output.exists():
            continue
        
        print(f"\n📁 Processing tool: {tool_dir.name}")
        
        # Look for the best program in the openevolve_output/best directory
        best_dir = openevolve_output / "best"
        if not best_dir.exists():
            print(f"  ⚠️  No best directory found")
            failed_count += 1
            continue
        
        best_program_file = best_dir / "best_program.py"
        if not best_program_file.exists():
            print(f"  ⚠️  No best_program.py found")
            failed_count += 1
            continue
        
        # Read the best program
        try:
            with open(best_program_file, 'r') as f:
                best_code = f.read()
            
            # Read metadata if available
            metadata_file = best_dir / "best_program_info.json"
            metadata = {}
            if metadata_file.exists():
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
            
            # Create the best_version.py file
            best_version_file = tool_dir / "best_version.py"
            
            # Add header with evolution info
            header = f'''"""
Best Evolved Version of {tool_dir.name}

Generated by OpenEvolve optimization
'''
            
            if metadata:
                metrics = metadata.get('metrics', {})
                if metrics:
                    header += f"""
Evolution Metrics:
"""
                    for metric, value in metrics.items():
                        header += f"- {metric}: {value:.4f}\n"
                
                generation = metadata.get('generation')
                if generation is not None:
                    header += f"- Generation: {generation}\n"
                
                iteration = metadata.get('iteration')
                if iteration is not None:
                    header += f"- Iteration: {iteration}\n"
            
            header += '"""\n\n'
            
            # Write the best version file
            with open(best_version_file, 'w') as f:
                f.write(header + best_code)
            
            print(f"  ✅ Extracted to: best_version.py")
            if metadata.get('metrics'):
                metrics_str = ", ".join([f"{k}={v:.3f}" for k, v in metadata['metrics'].items()])
                print(f"     Metrics: {metrics_str}")
            
            # Generate score comparison
            try:
                print(f"  📊 Generating score comparison...")
                original_scores, best_scores = await _evaluate_both_versions(tool_dir, tool_dir.name)
                
                if original_scores and best_scores:
                    comparison_data = {
                        "tool_name": tool_dir.name,
                        "evaluation_date": datetime.utcnow().isoformat(),
                        "original_version": {
                            "scores": original_scores,
                            "average": sum(original_scores.values()) / len(original_scores)
                        },
                        "best_version": {
                            "scores": best_scores,
                            "average": sum(best_scores.values()) / len(best_scores)
                        },
                        "improvements": {
                            metric: best_scores[metric] - original_scores.get(metric, 0)
                            for metric in best_scores.keys()
                        }
                    }
                    
                    # Save comparison
                    comparison_file = tool_dir / "score_comparison.json"
                    with open(comparison_file, 'w') as f:
                        json.dump(comparison_data, f, indent=2)
                    
                    avg_improvement = comparison_data["best_version"]["average"] - comparison_data["original_version"]["average"]
                    print(f"     Score comparison: {avg_improvement:+.3f} average improvement")
                    
            except Exception as e:
                print(f"     Warning: Could not generate score comparison: {e}")
            
            extracted_count += 1
            
        except Exception as e:
            print(f"  ❌ Error extracting best version: {e}")
            failed_count += 1
    
    # Summary
    print("\n" + "=" * 60)
    if extracted_count > 0:
        print(f"🎉 Successfully extracted {extracted_count} best versions!")
        print(f"📁 Best versions saved as 'best_version.py' in each tool directory")
    else:
        print(f"⚠️  No best versions found to extract")
    
    if failed_count > 0:
        print(f"⚠️  Failed to extract {failed_count} tools")
        print(f"💡 Make sure tools have been evolved with OpenEvolve first")
    
    return extracted_count

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
    train_parser.add_argument('tool_name', nargs='?', default=None, 
                             help='Tool name to generate training data for (e.g., get_fundamental_analysis)')
    train_parser.add_argument('--base-dir', default='.agent_evolve',
                             help='Base directory containing tool subdirectories (default: .agent_evolve)')
    train_parser.add_argument('--num-samples', '-n', type=int, default=10,
                             help='Number of training samples to generate per tool (default: 10)')
    train_parser.add_argument('--force', '-f', action='store_true',
                             help='Force regeneration of existing training data')
    
    # Generate evaluators command
    eval_parser = subparsers.add_parser('generate-evaluators', help='Generate evaluators for extracted tools')
    eval_parser.add_argument('tools_directory', nargs='?', default='.agent_evolve',
                            help='Directory containing tool subdirectories (default: .agent_evolve)')
    eval_parser.add_argument('--model', default='gpt-4o',
                            help='LLM model to use for generation (default: gpt-4o)')
    
    # Generate OpenEvolve configs command
    config_parser = subparsers.add_parser('generate-configs', help='Generate OpenEvolve configuration files for extracted tools')
    config_parser.add_argument('tools_directory', nargs='?', default='.agent_evolve',
                              help='Directory containing tool subdirectories (default: .agent_evolve)')
    
    # Run OpenEvolve command (renamed from run-openevolve to evolve)
    evolve_parser = subparsers.add_parser('evolve', help='Run OpenEvolve optimization for a specific tool')
    evolve_parser.add_argument('tool_name', nargs='?', default=None,
                              help='Name of the tool to optimize (looks in .agent_evolve directory)')
    evolve_parser.add_argument('--checkpoint', '-c', type=int, default=None,
                              help='Resume from specific checkpoint number')
    evolve_parser.add_argument('--list', '-l', action='store_true',
                              help='List available tools with their readiness status')
    evolve_parser.add_argument('--base-dir', default='.agent_evolve',
                              help='Base directory to look for tools (default: .agent_evolve)')
    evolve_parser.add_argument('--iterations', '-i', type=int, default=None,
                              help='Number of evolution iterations to run')
    
    # Full pipeline command (renamed from evolve to pipeline)
    pipeline_parser = subparsers.add_parser('pipeline', help='Run complete evolution pipeline: generate data, evaluator, configs, then optimize')
    pipeline_parser.add_argument('tool_name', help='Name of the tool to evolve')
    pipeline_parser.add_argument('--base-dir', default='.agent_evolve',
                                help='Base directory to look for tools (default: .agent_evolve)')
    pipeline_parser.add_argument('--num-samples', '-n', type=int, default=10,
                                help='Number of training samples to generate (default: 10)')
    pipeline_parser.add_argument('--model', default='gpt-4o',
                                help='LLM model to use for generation (default: gpt-4o)')
    pipeline_parser.add_argument('--checkpoint', '-c', type=int, default=None,
                                help='Resume OpenEvolve from specific checkpoint number')
    pipeline_parser.add_argument('--force-training-data', '-f', action='store_true',
                                help='Force regeneration of existing training data')
    
    # Extract best versions command
    extract_parser = subparsers.add_parser('extract-best', help='Extract best evolved versions as best_version.py files')
    extract_parser.add_argument('--base-dir', default='.agent_evolve',
                               help='Base directory to look for evolved tools (default: .agent_evolve)')
    
    # Dashboard command
    dashboard_parser = subparsers.add_parser('dashboard', help='Launch the Streamlit dashboard for visualizing evolution results')
    dashboard_parser.add_argument('--base-dir', default='.agent_evolve',
                                 help='Base directory to look for evolved tools (default: .agent_evolve)')
    dashboard_parser.add_argument('--port', type=int, default=8501,
                                 help='Port to run the dashboard on (default: 8501)')
    dashboard_parser.add_argument('--host', default='localhost',
                                 help='Host to run the dashboard on (default: localhost)')
    
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
        generate_training_data(args.tool_name, args.base_dir, args.num_samples, args.force)
    elif args.command == 'generate-evaluators':
        generate_evaluators(args.tools_directory, args.model)
    elif args.command == 'generate-configs':
        generate_openevolve_configs(args.tools_directory)
    elif args.command == 'evolve':
        asyncio.run(run_openevolve(args.tool_name, args.checkpoint, args.list, args.base_dir, args.iterations))
    elif args.command == 'pipeline':
        asyncio.run(run_full_pipeline(args.tool_name, args.base_dir, args.num_samples, 
                                     args.model, args.checkpoint, args.force_training_data))
    elif args.command == 'extract-best':
        asyncio.run(extract_best_versions(args.base_dir))
    elif args.command == 'dashboard':
        run_dashboard(args.base_dir, args.port, args.host)

if __name__ == '__main__':
    main()