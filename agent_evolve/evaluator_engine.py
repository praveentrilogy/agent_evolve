"""Evaluator Engine - Generates and runs prompt/code evaluators as tests"""


import os
import json
import sqlite3
import subprocess
import sys
import tempfile
import importlib.util
import logging
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from current working directory
load_dotenv()

logger = logging.getLogger(__name__)

EVALUATOR_GENERATION_PROMPT = """
You are an expert test engineer who specializes in creating evaluators for AI prompts and code.

Generate a Python evaluator file that tests the given prompt/code against training data.

Here's what you have:
- **Prompt/Code Details**: {prompt_details}
- **Use Case**: {use_case}
- **Training Data Sample**: {training_sample}
- **Import Path**: {import_path}

Create a complete Python file that:
1. Imports the actual prompt/function from its location
2. Loads training data from the database
3. Runs the prompt/function against training data
4. Calculates appropriate metrics based on the use case
5. Returns results in the format: {"metrics": dict, "details": dict, "recommendations": list}

Choose metrics appropriate for the use case:
- **Content Generation**: quality, relevance, engagement, authenticity
- **Code Functions**: correctness, performance, maintainability
- **Research/Analysis**: accuracy, completeness, depth
- **Planning**: feasibility, clarity, completeness

The evaluator should work as a standalone test that can be executed.

Return only the Python code for the evaluator file:

```python
# Generated evaluator for {prompt_name}
import sqlite3
import json
import sys
import os
from pathlib import Path

import logging

logger = logging.getLogger(__name__)

# Add the project root to Python path
sys.path.append(str(Path(__file__).parent.parent.parent))

# Import the actual prompt/function
{import_statement}

def load_training_data(db_path, prompt_id):
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

def evaluate():
    # Get parameters from command line or environment
    db_path = sys.argv[1] if len(sys.argv) > 1 else 'data/graph.db'
    prompt_id = sys.argv[2] if len(sys.argv) > 2 else None
    
    if not prompt_id:
        return {"error": "No prompt_id provided"}
    
    # Load training data
    training_data = load_training_data(db_path, prompt_id)
    
    if not training_data:
        return {"error": "No training data available"}
    
    # Initialize metrics
    metrics = {}
    details = {
        "total_samples": len(training_data),
        "real_samples": len([s for s in training_data if s['data_source'] == 'real']),
        "synthetic_samples": len([s for s in training_data if s['data_source'] == 'synthetic']),
        "evaluations": []
    }
    recommendations = []
    
    {evaluation_logic}
    
    return {
        "metrics": metrics,
        "details": details,
        "recommendations": recommendations
    }

if __name__ == "__main__":
    import json
    results = evaluate()
    print(json.dumps(results, indent=2))
```
"""

def get_use_case_from_prompt(prompt_content, prompt_type):
    """Determine the use case based on prompt content and type"""
    content_lower = prompt_content.lower()
    
    if any(word in content_lower for word in ['classify', 'categorize', 'intent', 'label']):
        return "classification"
    if any(word in content_lower for word in ['write', 'generate', 'create', 'compose']):
        return "content_generation"
    elif any(word in content_lower for word in ['research', 'analyze', 'investigate', 'study']):
        return "research_analysis"
    elif any(word in content_lower for word in ['plan', 'strategy', 'approach', 'steps']):
        return "planning"
    elif prompt_type == 'function':
        return "code_function"
    else:
        return "general"

def get_import_path_from_db(db_path, prompt_id):
    """Get the import path for the prompt from database"""
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.execute('''
            SELECT definition_location, prompt_name, full_code
            FROM prompts
            WHERE id = ?
        ''', (prompt_id,))
        
        result = cursor.fetchone()
        conn.close()
        
        if result:
            definition_location, prompt_name, full_code = result
            # Parse the location to create import path
            if definition_location and '/' in definition_location:
                # Convert file path to import path
                file_path = definition_location.replace('.py', '').replace('/', '.')
                return f"from {file_path} import {prompt_name}"
            else:
                return f"# Prompt: {prompt_name}\n# Content available in full_code variable"
        
        return "# Unable to determine import path"
    except Exception as e:
        logger.error(f"Error getting import path: {e}")
        return "# Error determining import path"

def generate_evaluation_logic(use_case):
    """Generate evaluation logic based on use case"""
    
    if use_case == "content_generation":
        return """
    # Content Generation Evaluation
    quality_scores = []
    relevance_scores = []
    engagement_scores = []
    
    for i, sample in enumerate(training_data[:10]):  # Evaluate first 10 samples
        # Here you would run the actual prompt and evaluate the output
        # For now, we'll use placeholder scoring
        
        # Simulate evaluation (replace with actual prompt execution)
        quality_score = 0.8  # Replace with actual quality assessment
        relevance_score = 0.85  # Replace with actual relevance assessment
        engagement_score = 0.75  # Replace with actual engagement assessment
        
        quality_scores.append(quality_score)
        relevance_scores.append(relevance_score)
        engagement_scores.append(engagement_score)
        
        details["evaluations"].append({
            "sample_id": i,
            "inputs": sample['inputs'],
            "quality": quality_score,
            "relevance": relevance_score,
            "engagement": engagement_score,
            "data_source": sample['data_source']
        })
    
    # Calculate aggregate metrics
    metrics["quality"] = sum(quality_scores) / len(quality_scores) if quality_scores else 0
    metrics["relevance"] = sum(relevance_scores) / len(relevance_scores) if relevance_scores else 0
    metrics["engagement"] = sum(engagement_scores) / len(engagement_scores) if engagement_scores else 0
    
    # Generate recommendations
    if metrics["quality"] < 0.7:
        recommendations.append("Quality scores are below threshold. Consider improving prompt clarity.")
    if metrics["relevance"] < 0.8:
        recommendations.append("Relevance could be improved. Ensure prompt stays on topic.")
    if len(training_data) < 20:
        recommendations.append("Consider adding more training data for better evaluation coverage.")
        """
    
    elif use_case == "research_analysis":
        return """
    # Research/Analysis Evaluation
    accuracy_scores = []
    completeness_scores = []
    depth_scores = []
    
    for i, sample in enumerate(training_data[:10]):
        # Simulate research evaluation
        accuracy_score = 0.9
        completeness_score = 0.8
        depth_score = 0.75
        
        accuracy_scores.append(accuracy_score)
        completeness_scores.append(completeness_score)
        depth_scores.append(depth_score)
        
        details["evaluations"].append({
            "sample_id": i,
            "inputs": sample['inputs'],
            "accuracy": accuracy_score,
            "completeness": completeness_score,
            "depth": depth_score,
            "data_source": sample['data_source']
        })
    
    metrics["accuracy"] = sum(accuracy_scores) / len(accuracy_scores) if accuracy_scores else 0
    metrics["completeness"] = sum(completeness_scores) / len(completeness_scores) if completeness_scores else 0
    metrics["depth"] = sum(depth_scores) / len(depth_scores) if depth_scores else 0
    
    if metrics["accuracy"] < 0.85:
        recommendations.append("Accuracy needs improvement. Verify information sources.")
    if metrics["completeness"] < 0.8:
        recommendations.append("Analysis could be more complete. Consider broader coverage.")
        """
    
    else:
        return """
    # General Evaluation
    overall_scores = []
    
    for i, sample in enumerate(training_data[:10]):
        # Basic evaluation
        overall_score = 0.8  # Placeholder
        overall_scores.append(overall_score)
        
        details["evaluations"].append({
            "sample_id": i,
            "inputs": sample['inputs'],
            "score": overall_score,
            "data_source": sample['data_source']
        })
    
    metrics["overall_score"] = sum(overall_scores) / len(overall_scores) if overall_scores else 0
    
    if metrics["overall_score"] < 0.7:
        recommendations.append("Overall performance needs improvement.")
        """

def auto_generate_evaluator_on_evolution(db_path, item_id, item_type, project_root):
    """Auto-generate evaluator.py when an item is marked for evolution"""
    try:
        conn = sqlite3.connect(db_path)
        
        if item_type == 'prompt':
            cursor = conn.execute('''
                SELECT prompt_name, content, variables, prompt_type, definition_location
                FROM prompts
                WHERE id = ?
            ''', (item_id,))
            result = cursor.fetchone()
            if not result:
                return False
            item_name, content, variables_json, item_type_detail, definition_location = result
            
        elif item_type == 'code':
            cursor = conn.execute('''
                SELECT full_name, source_code, definition_location
                FROM functions
                WHERE id = ?
            ''', (item_id,))
            result = cursor.fetchone()
            if not result:
                return False
            item_name, content, definition_location = result
            variables_json = None
            item_type_detail = 'function'
            
        conn.close()
        
        # Create evaluator directory using project_root
        evaluator_dir = os.path.join(project_root, ".agent_evolve", item_name)
        os.makedirs(evaluator_dir, exist_ok=True)
        
        # Generate evaluator.py
        return create_standard_evaluator(
            evaluator_dir=evaluator_dir,
            item_id=item_id,
            item_name=item_name,
            item_type=item_type_detail,
            definition_location=definition_location,
            content=content,
            variables=json.loads(variables_json) if variables_json else {}
        )
        
    except Exception as e:
        logger.error(f"Error auto-generating evaluator: {e}")
        return False

def create_standard_evaluator(evaluator_dir, item_id, item_name, item_type, definition_location, content, variables):
    """Create a standard evaluator.py template with proper imports and functions"""
    try:
        # Determine import statement
        import_statement = get_import_statement_from_location(definition_location, item_name)
        
        # Use LLM to determine the best evaluator template
        template_name = select_evaluator_template_with_llm(content, item_type, item_name)
        
        # Load the selected template
        evaluator_template = load_evaluator_template(template_name, item_id, item_name, item_type, import_statement)
        
        # Write the evaluator file
        evaluator_path = os.path.join(evaluator_dir, "evaluator.py")
        with open(evaluator_path, 'w') as f:
            f.write(evaluator_template)
        
        logger.info(f"Created {template_name} evaluator at {evaluator_path}")
        return True
        
    except Exception as e:
        logger.error(f"Error creating standard evaluator: {e}")
        return False

def select_evaluator_template_with_llm(content, item_type, item_name):
    """Use LLM to select the most appropriate evaluator template"""
    try:
        from agent_evolve.llm import get_llm_response_json
        
        selection_prompt = f"""You are an expert in software evaluation and testing. Based on the following information, select the most appropriate evaluator template:

**Item Name**: {item_name}
**Item Type**: {item_type}
**Content/Purpose**: {content[:500]}...

Available templates:
1. **default_evaluator** - General-purpose template for most functions/prompts
2. **content_generation_evaluator** - Specialized for content creation, writing, marketing copy, articles, blogs, etc.
3. **classification_evaluator** - For classification tasks, comparing output to a single expected value.

Respond with ONLY a JSON object:
{{
  "template": "default_evaluator" or "content_generation_evaluator" or "classification_evaluator",
  "reasoning": "Brief explanation of why this template is most suitable"
}}"""
        logger.info(f"Sending LLM selection prompt for '{item_name}':\n{selection_prompt}")
        result = get_llm_response_json(selection_prompt, temperature=0.0)
        logger.info(f"LLM selection response for '{item_name}': {result}")
        
        if isinstance(result, dict) and "template" in result:
            selected_template = result["template"]
            reasoning = result.get("reasoning", "No reasoning provided")
            
            # Validate template exists
            if selected_template in ["default_evaluator", "content_generation_evaluator", "classification_evaluator"]:
                logger.info(f"LLM selected {selected_template} for {item_name}: {reasoning}")
                return selected_template
            else:
                logger.warning(f"LLM returned unknown template {selected_template}, using default")
                return "default_evaluator"
        else:
            logger.warning("Invalid LLM response for template selection, using default")
            return "default_evaluator"
            
    except Exception as e:
        logger.warning(f"Error using LLM for template selection: {e}, falling back to default")
        return "default_evaluator"

from string import Template

def load_evaluator_template(template_name, item_id, item_name, item_type, import_statement):
    """Load and format the specified evaluator template"""
    try:
        # Get the template file path
        current_dir = os.path.dirname(__file__)
        template_path = os.path.join(current_dir, "evaluator_templates", f"{template_name}.py")
        
        if not os.path.exists(template_path):
            logger.warning(f"Template {template_name} not found, using default")
            template_path = os.path.join(current_dir, "evaluator_templates", "default_evaluator.py")
        
        # Read the template
        with open(template_path, 'r') as f:
            template_content = f.read()
        
        # Use string.Template for safe substitution
        template = Template(template_content)
        formatted_template = template.substitute(
            item_id=item_id,
            item_name=item_name,
            item_type=item_type,
            import_statement=import_statement
        )
        
        return formatted_template
        
    except Exception as e:
        logger.error(f"Error loading template {template_name}: {e}")
        # Return a basic default template as fallback
        return create_fallback_template(item_name, item_type, import_statement)

def create_fallback_template(item_name, item_type, import_statement):
    """Create a basic fallback template when template files are not available"""
    return f'''"""
Fallback evaluator for {item_name}
Type: {item_type}
"""
import json
from pathlib import Path

{import_statement}

def load_training_data():
    current_dir = Path(__file__).parent
    real_data_file = current_dir / "real_training_data.json"
    if real_data_file.exists():
        with open(real_data_file, 'r') as f:
            return json.load(f)
    return []

def load_synthetic_data():
    current_dir = Path(__file__).parent
    synthetic_data_file = current_dir / "synthesized_training_data.json"
    if synthetic_data_file.exists():
        with open(synthetic_data_file, 'r') as f:
            return json.load(f)
    return []

def evaluate():
    real_data = load_training_data()
    synthetic_data = load_synthetic_data()
    all_data = real_data + synthetic_data
    
    return {{
        "metrics": {{"total_samples": len(all_data)}},
        "details": {{"message": "Basic fallback evaluator"}},
        "recommendations": ["Customize this evaluator for your needs"]
    }}

if __name__ == "__main__":
    import json
    results = evaluate()
    print(json.dumps(results, indent=2))
'''


def get_import_statement_from_location(definition_location, item_name):
    """Generate proper import statement with path handling for executable code"""
    if not definition_location or not item_name:
        return f"# TODO: Import {item_name} from {definition_location}"
    
    try:
        # Handle format: "/full/path/to/file.py:ITEM_NAME"
        if ':' in definition_location:
            file_path = definition_location.split(':')[0]
        else:
            file_path = definition_location
            
        # Extract relative path from the full path
        # Try to find a common project structure indicator and get relative path
        relative_path = file_path
        for marker in ['/src/', '/app/', '/lib/', '/backend/', '/frontend/', '/server/', '/client/']:
            if marker in file_path:
                # Include the marker directory in the path
                marker_index = file_path.find(marker)
                relative_path = file_path[marker_index + 1:]  # Skip the leading /
                break
        
        # If no marker found, use just the filename
        if relative_path == file_path:
            relative_path = os.path.basename(file_path)
            
        if relative_path.endswith('.py'):
            # Get the directory containing the file
            module_name = os.path.basename(relative_path)[:-3]  # Remove .py
            
            return f"""# Add project root to Python path for imports
import sys
from pathlib import Path

# Dynamically find project root by looking for common markers
def find_project_root():
    current = Path(__file__).parent
    while current != current.parent:
        # Look for common project markers (generic, not project-specific)
        if any((current / marker).exists() for marker in [
            'pyproject.toml', 'setup.py', 'requirements.txt', '.git', 
            'src', 'Makefile', 'package.json', 'Cargo.toml', 'go.mod'
        ]):
            return current
        current = current.parent
    return Path(__file__).parent.parent  # fallback

project_root = find_project_root()
sys.path.insert(0, str(project_root))

# Import the original function/prompt
try:
    from {relative_path[:-3].replace('/', '.').replace('\\', '.')} import {item_name}
except ImportError as e:
    # Alternative import method if module path doesn't work
    import importlib.util
    spec = importlib.util.spec_from_file_location("{module_name}", project_root / "{relative_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    {item_name} = getattr(module, "{item_name}")"""
        else:
            return f"# TODO: Import {item_name} from {definition_location}"
    except Exception as e:
        return f"# TODO: Import {item_name} from {definition_location} (Error: {e})"

def run_evaluator(db_path, prompt_id, prompt_name, evaluator_path):
    """Run the evaluator as a test and return results"""
    try:
        # Run the evaluator as a subprocess
        result = subprocess.run([
            sys.executable, evaluator_path, db_path, prompt_id
        ], capture_output=True, text=True, timeout=60)
        
        if result.returncode != 0:
            return {"error": f"Evaluator failed with return code {result.returncode}: {result.stderr}"}
        
        # Parse the JSON output
        try:
            results = json.loads(result.stdout)
            return results
        except json.JSONDecodeError:
            # If JSON parsing fails, return the raw output
            return {"error": "Invalid JSON output from evaluator", "raw_output": result.stdout}
            
    except subprocess.TimeoutExpired:
        return {"error": "Evaluator timed out after 60 seconds"}
    except Exception as e:
        logger.error(f"Error running evaluator: {e}")
        return {"error": str(e)}

def generate_evaluator_cli(target_name: str, db_path: str = 'data/graph.db', project_root: str = os.getcwd()):
    """CLI function to generate an evaluator for a specific prompt or function."""
    logger.info(f"Attempting to generate evaluator for '{target_name}'...")
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    item_id = None
    item_type = None
    
    # Try to find in prompts table
    cursor.execute("SELECT id FROM prompts WHERE prompt_name = ?", (target_name,))
    result = cursor.fetchone()
    if result:
        item_id = result[0]
        item_type = 'prompt'
    
    # If not found, try to find in functions table
    if not item_id:
        cursor.execute("SELECT id FROM functions WHERE function_name = ? OR full_function_name = ?", (target_name, target_name))
        result = cursor.fetchone()
        if result:
            item_id = result[0]
            item_type = 'code'
            
    conn.close()
    
    if not item_id:
        logger.error(f"Error: Target '{target_name}' not found in prompts or functions database.")
        return
        
    logger.info(f"Found target '{target_name}' (ID: {item_id}, Type: {item_type}). Generating evaluator...")
    success = auto_generate_evaluator_on_evolution(db_path, item_id, item_type, project_root)
    
    if success:
        logger.info(f"Successfully generated evaluator for '{target_name}'.")
    else:
        logger.error(f"Failed to generate evaluator for '{target_name}'.")