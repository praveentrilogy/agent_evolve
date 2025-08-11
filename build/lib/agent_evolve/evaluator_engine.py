"""Evaluator Engine - Generates and runs prompt/code evaluators as tests"""


import os
import json
import sqlite3
import subprocess
import sys
import tempfile
import importlib.util
import logging
import re
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from current working directory
load_dotenv()

logger = logging.getLogger(__name__)

EVALUATOR_GENERATION_PROMPT = """
You are an expert test engineer who specializes in creating evaluators for AI prompts and code.

Generate a Python evaluator file that tests the given prompt/code against training data.

Here's what you have:
- **Item Details**: {item_details} (This can be prompt or code details)
- **Use Case**: {use_case} (e.g., 'code_function', 'content_generation')
- **Training Data Sample**: {training_sample}
- **Import Statement for Target**: {import_statement} (This is how to import the function/prompt to be evaluated)

Create a complete Python file named `evaluator.py` that:
1.  **Imports the actual prompt/function** from its location using the provided `{import_statement}`.
2.  **Loads training data** from the database using `agent_evolve.agent_evolve.data_utils.load_training_data`.
3.  **Defines an `evaluate()` function** that takes no arguments and returns a dictionary.
4.  **Iterates through each training data point**. For each point:
    *   **Executes the target function/prompt** with the `inputs` from the training data.
    *   **Performs an evaluation of the output**. This evaluation should be subjective and dynamic.
    *   **If the target function's output is non-deterministic** (e.g., content generation, code generation, creative tasks), use an **LLM call** (import `get_llm_response` from `agent_evolve.llm`) to subjectively assess the quality, correctness, relevance, or other appropriate metrics of the generated output against the expected output (if available) or the use case.
    *   **Collects metrics** for that specific training data point. Choose metrics appropriate for the use case and the nature of the function's output.
5.  **Aggregates results** and returns them in the format: `{"metrics": dict, "details": dict, "recommendations": list}`. The `metrics` dictionary should contain overall aggregated metrics, and `details` should include a list of individual evaluation results for each training data point.

Choose metrics appropriate for the use case:
- **Content Generation**: quality, relevance, engagement, authenticity, creativity
- **Code Functions**: correctness, performance, maintainability, security, efficiency
- **Research/Analysis**: accuracy, completeness, depth, conciseness
- **Planning**: feasibility, clarity, completeness, robustness

The evaluator should work as a standalone test that can be executed.

Return only the Python code for the evaluator file. Ensure all necessary imports (like `agent_evolve.llm` for LLM calls) are included.

```python
# Generated evaluator for {item_name}
import sqlite3
import json
import sys
import os
from pathlib import Path
import logging

# Import data utility functions
from agent_evolve.agent_evolve.data_utils import load_training_data
# Import LLM for subjective evaluation if needed
from agent_evolve.llm import get_llm_response

logger = logging.getLogger(__name__)

# Import the actual prompt/function to be evaluated
{import_statement}

def evaluate():
    # Get parameters from command line or environment
    db_path = sys.argv[1] if len(sys.argv) > 1 else 'data/graph.db'
    item_id = sys.argv[2] if len(sys.argv) > 2 else None # This will be prompt_id or function_id
    
    if not item_id:
        return {"error": "No item_id provided"}
    
    # Load training data
    training_data = load_training_data(db_path, item_id)
    
    if not training_data:
        return {"error": "No training data available"}
    
    # Initialize metrics and details
    metrics = {}
    details = {
        "total_samples": len(training_data),
        "real_samples": len([s for s in training_data if s['data_source'] == 'real']),
        "synthetic_samples": len([s for s in training_data if s['data_source'] == 'synthetic']),
        "evaluations": []
    }
    recommendations = []
    
    # --- LLM-generated evaluation logic starts here ---
    # The LLM will fill this part based on the use case and item details.
    # It should iterate through training_data, call the imported function/prompt,
    # perform subjective evaluation (using get_llm_response for non-deterministic outputs),
    # and collect metrics for each sample.
    # The LLM should also ensure that the 'metrics' dictionary contains comparable numerical values.
    
    # IMPORTANT: For prompts, the target program is the content of 'evolve_target.py'.
    # For code functions, the target program is the imported function itself (named '{item_name}').
    # The LLM should use 'item_type' (which is passed as a template variable) to determine this.
    
    {evaluation_logic}
    
    # --- LLM-generated evaluation logic ends here ---
    
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

def auto_generate_evaluator_for_extracted_tool(tool_dir_path, project_root):
    """Auto-generate evaluator for extracted tool using file-based approach"""
    try:
        tool_dir = Path(tool_dir_path)
        item_name = tool_dir.name
        
        # Read metadata to determine item type
        metadata_file = tool_dir / "metadata.json"
        if not metadata_file.exists():
            logger.error(f"Metadata file not found: {metadata_file}")
            return False
            
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        item_type = 'prompt' if metadata.get('type') == 'prompt_template' else 'code'
        
        # Read the evolve_target.py to get content
        evolve_target = tool_dir / "evolve_target.py"
        if not evolve_target.exists():
            logger.error(f"evolve_target.py not found: {evolve_target}")
            return False
            
        with open(evolve_target, 'r') as f:
            content = f.read()
        
        definition_location = metadata.get('original_file', '')
        
        # Generate evaluator.py
        if item_type == 'prompt':
            return create_prompt_evaluator_file_based(
                evaluator_dir=str(tool_dir),
                item_name=item_name,
                item_type='template',
                definition_location=definition_location,
                content=content,
                variables={}
            )
        else:
            return create_code_evaluator_file_based(
                evaluator_dir=str(tool_dir),
                item_name=item_name,
                item_type='function',
                definition_location=definition_location,
                content=content
            )
        
    except Exception as e:
        logger.error(f"Error auto-generating evaluator: {e}")
        return False

def create_prompt_evaluator_file_based(evaluator_dir, item_name, item_type, definition_location, content, variables):
    """Create a prompt evaluator using file-based approach with OpenEvolve format"""
    try:
        # Use LLM to determine the best evaluator template
        template_name = select_evaluator_template_with_llm(content, item_type, item_name)
        
        # Load and customize the selected template for file-based operation
        evaluator_template = load_file_based_evaluator_template(template_name, item_name, item_type)
        
        # Write the evaluator file
        evaluator_path = os.path.join(evaluator_dir, "evaluator.py")
        with open(evaluator_path, 'w') as f:
            f.write(evaluator_template)
        
        logger.info(f"Created {template_name} evaluator at {evaluator_path}")
        return True
        
    except Exception as e:
        logger.error(f"Error creating file-based prompt evaluator: {e}")
        return False

def create_code_evaluator_file_based(evaluator_dir, item_name, item_type, definition_location, content):
    """Create a code evaluator using file-based approach with OpenEvolve format"""
    try:
        # For code, determine the best evaluator template
        template_name = select_evaluator_template_with_llm(content, item_type, item_name)
        
        # Load and customize the selected template for file-based operation
        evaluator_template = load_file_based_evaluator_template(template_name, item_name, item_type)
        
        # Write the evaluator file
        evaluator_path = os.path.join(evaluator_dir, "evaluator.py")
        with open(evaluator_path, 'w') as f:
            f.write(evaluator_template)
        
        logger.info(f"Created {template_name} evaluator at {evaluator_path}")
        return True
        
    except Exception as e:
        logger.error(f"Error creating file-based code evaluator: {e}")
        return False

def load_file_based_evaluator_template(template_name, item_name, item_type):
    """Load evaluator template customized for file-based operation with OpenEvolve format"""
    try:
        # Get the template file path
        current_dir = os.path.dirname(__file__)
        template_path = os.path.join(current_dir, "evaluator_templates", f"{template_name}_openevolve.py")
        
        # If OpenEvolve version doesn't exist, use regular version and adapt it
        if not os.path.exists(template_path):
            template_path = os.path.join(current_dir, "evaluator_templates", f"{template_name}.py")
        
        if not os.path.exists(template_path):
            logger.warning(f"Template {template_name} not found, using default")
            template_path = os.path.join(current_dir, "evaluator_templates", "default_evaluator.py")
        
        # Read the template
        with open(template_path, 'r') as f:
            template_content = f.read()
        
        # Use string.Template for safe substitution
        template = Template(template_content)
        formatted_template = template.safe_substitute(
            item_name=item_name,
            item_type=item_type
        )
        
        return formatted_template
        
    except Exception as e:
        logger.error(f"Error loading file-based template {template_name}: {e}")
        # Return a basic fallback template
        return create_openevolve_fallback_template(item_name, item_type)

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
                SELECT function_name, definition_location
                FROM functions
                WHERE id = ?
            ''', (item_id,))
            result = cursor.fetchone()
            if not result:
                return False
            item_name, definition_location = result
            
            # Extract function content from the file path
            content = extract_function_code(definition_location, item_name)
            if content is None:
                return False
                
            variables_json = None
            item_type_detail = 'function'
            
        conn.close()
        
        # Create evaluator directory using project_root
        evaluator_dir = os.path.join(project_root, ".agent_evolve", item_name)
        os.makedirs(evaluator_dir, exist_ok=True)
        
        # Generate evaluator.py
        if item_type == 'prompt':
            return create_prompt_evaluator(
                evaluator_dir=evaluator_dir,
                item_id=item_id,
                item_name=item_name,
                item_type=item_type_detail,
                definition_location=definition_location,
                content=content,
                variables=json.loads(variables_json) if variables_json else {}
            )
        elif item_type == 'code':
            return create_code_evaluator(
                evaluator_dir=evaluator_dir,
                item_id=item_id,
                item_name=item_name,
                item_type=item_type_detail,
                definition_location=definition_location,
                content=content
            )
        
    except Exception as e:
        logger.error(f"Error auto-generating evaluator: {e}")
        return False

def create_prompt_evaluator(evaluator_dir, item_id, item_name, item_type, definition_location, content, variables):
    """Create a prompt evaluator.py template with proper imports and functions"""
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

def create_code_evaluator(evaluator_dir, item_id, item_name, item_type, definition_location, content):
    """Create a code evaluator.py template with proper imports and functions"""
    try:
        # Determine import statement for the code function
        import_statement = get_import_statement_from_location(definition_location, item_name)

        # For code, the use_case is always 'code_function'
        use_case = "code_function"

        # Use LLM to determine the best evaluator template (or directly use a code-specific one)
        # For now, we'll assume select_evaluator_template_with_llm can handle 'code_function'
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
        logger.error(f"Error creating code evaluator: {e}")
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
        result = get_llm_response_json(selection_prompt)
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
        # The project root is assumed to be added to sys.path by the evaluator.py itself.
        # We need to calculate the module path relative to the project root.
        
        # First, determine the project root from the current working directory
        # This logic is duplicated from find_project_root in the generated template,
        # but it's necessary here to calculate the relative path.
        current_script_dir = Path(__file__).parent
        project_root = None
        temp_dir = current_script_dir
        while temp_dir != temp_dir.parent:
            if any((temp_dir / marker).exists() for marker in [
                'pyproject.toml', 'setup.py', 'requirements.txt', '.git', 
                'src', 'Makefile', 'package.json', 'Cargo.toml', 'go.mod'
            ]):
                project_root = temp_dir
                break
            temp_dir = temp_dir.parent
        if project_root is None:
            project_root = current_script_dir.parent.parent # Fallback, similar to generated code

        # Ensure definition_location is an absolute path
        abs_definition_path = Path(definition_location).resolve()

        # Calculate the relative path from the project root to the file
        relative_file_path = os.path.relpath(str(abs_definition_path), str(project_root))
        
        # Convert file path to module import path (e.g., 'src/app/module.py' -> 'src.app.module')
        module_import_path = str(Path(relative_file_path).with_suffix('')).replace(os.sep, '.')
        
        # If the module is directly in the project root, it might start with '.'
        # Remove leading '.' if present
        if module_import_path.startswith('.'):
            module_import_path = module_import_path[1:]

        # Construct the import statement
        return f"from {module_import_path} import {item_name}"

    except Exception as e:
        logger.error(f"Error generating import statement for {item_name} from {definition_location}: {e}")
        return f"# Error generating import statement for {item_name} from {definition_location}"

import re

def extract_function_code(file_path, function_name):
    """
    Extracts the source code of a specific function from a Python file.
    This is a simplified implementation using regex and might not be robust for all cases.
    """
    try:
        with open(file_path, 'r') as f:
            content = f.read()

        # Regex to find the function definition and its body
        # It looks for 'def function_name(...):' and captures the indented block following it.
        # This regex is very basic and assumes standard Python function definition and indentation.
        pattern = r"^(def\s+" + re.escape(function_name) + r"\s*\(.*\):(?:\s*#.*|\s*|\n(?:\s{4}.*|\s*#.*|\s*)*)*)"
        
        match = re.search(pattern, content, re.MULTILINE)
        
        if match:
            # Get the full matched function definition including its body
            function_code = match.group(1)
            return function_code
        else:
            logger.warning(f"Function '{function_name}' not found in file '{file_path}'.")
            return None
    except Exception as e:
        logger.error(f"Error extracting function '{function_name}' from '{file_path}': {e}")
        return None

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