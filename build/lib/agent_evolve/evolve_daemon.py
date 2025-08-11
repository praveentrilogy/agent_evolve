"""
Agent Evolve Daemon - Processes the evolution queue.
"""

import time
import sqlite3
import subprocess
import os
from datetime import datetime
from agent_evolve.generate_training_data import TrainingDataGenerator 
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def is_git_repository(path='.'):
    """Check if the current directory or its parent is a git repository."""
    # Check current directory first
    if subprocess.call(['git', '-C', path, 'rev-parse'], stderr=subprocess.STDOUT, stdout=open(os.devnull, 'w')) == 0:
        return True
    # Check parent directory if not found in current
    return subprocess.call(['git', '-C', os.path.join(path, '..'), 'rev-parse'], stderr=subprocess.STDOUT, stdout=open(os.devnull, 'w')) == 0

def create_git_branch(branch_name):
    """Create a new git branch or checkout an existing one in the parent directory's git repo."""
    git_repo_path = '..' # Assuming the git repo is in the parent directory
    try:
        # Try to checkout the branch first
        subprocess.check_call(['git', '-C', git_repo_path, 'checkout', branch_name])
        logger.info(f"Successfully switched to existing branch '{branch_name}'.")
    except subprocess.CalledProcessError:
        # If checkout fails, it means the branch doesn't exist, so create it
        try:
            subprocess.check_call(['git', '-C', git_repo_path, 'checkout', '-b', branch_name])
            logger.info(f"Successfully created and switched to new branch '{branch_name}'.")
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to create git branch: {e}")

def run_app_server(app_start_command):
    """Run the application server in the background."""
    try:
        subprocess.Popen(app_start_command, shell=True)
        logger.info("Application server started.")
    except Exception as e:
        logger.error(f"Failed to run app server: {e}")

def process_queue(db_path, app_start_command):
    """Process the evolution queue."""
    logger.info("Checking evolution queue...")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Get the next queued item from prompt evolution queue
    cursor.execute("SELECT id, prompt_id FROM prompt_evolve_queue WHERE status = 'queued' ORDER BY created_at ASC LIMIT 1")
    item = cursor.fetchone()

    if item:
        queue_id, prompt_id = item
        logger.info(f"Found queued prompt item {queue_id} for prompt {prompt_id}")

        # Update status to 'running'
        now = datetime.utcnow().isoformat()
        cursor.execute("UPDATE prompt_evolve_queue SET status = 'running', updated_at = ? WHERE id = ?", (now, queue_id))
        conn.commit()

        # Get prompt name
        cursor.execute("SELECT prompt_name FROM prompts WHERE id = ?", (prompt_id,))
        prompt_name = cursor.fetchone()[0]

        # Evolve the prompt
        project_root = os.environ.get('AGENT_EVOLVE_PROJECT_ROOT', os.getcwd())
        if is_git_repository(project_root):
            create_git_branch(f"evolve-{prompt_name}")
        
        logger.info(f"Calling generate_training_data for prompt '{prompt_name}'")
        generator = TrainingDataGenerator()
        # For daemon, we need to work with extracted tools directory
        tools_directory = os.path.join(project_root, ".agent_evolve")
        generator.generate_training_data(tools_directory, force=False, specific_tool=prompt_name)
        
        # Generate evaluator
        logger.info(f"Generating evaluator for prompt '{prompt_name}'")
        from agent_evolve.evaluator_engine import auto_generate_evaluator_on_evolution
        auto_generate_evaluator_on_evolution(db_path, prompt_id, 'prompt', project_root)

        # run_app_server(app_start_command)

        # For now, just mark as completed
        now = datetime.utcnow().isoformat()
        cursor.execute("UPDATE prompt_evolve_queue SET status = 'completed', updated_at = ? WHERE id = ?", (now, queue_id))
        conn.commit()
        logger.info(f"Finished processing queue item {queue_id}")

    else:
        logger.info("No queued items found.")

    conn.close()

def main():
    """Main function for the evolve daemon."""
    logger.info("Agent Evolve Daemon started.")
    db_path = os.environ.get('AGENT_EVOLVE_DB_PATH', 'data/graph.db')
    app_start_command = os.environ.get('APP_START_COMMAND', 'echo "No app start command provided"')
    logger.info(f"Daemon using database path: {db_path}")
    while True:
        process_queue(db_path, app_start_command)
        time.sleep(10)

if __name__ == "__main__":
    main()
