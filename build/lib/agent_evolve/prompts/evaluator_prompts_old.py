"""
Prompts for evaluator generation
"""

# Common code snippets used across evaluator types
COMMON_IMPORTS = """import os
import json
import importlib.util
import inspect
import re
import openai
import logging

logger = logging.getLogger(__name__)"""

PARSE_JSON_FUNCTION = """def parse_json_response(response_content: str) -> dict:
    try:
        content = response_content.strip()
        if not content:
            return {metric: 0.0 for metric in EVALUATION_METRICS}
        
        # Extract JSON from markdown blocks
        if "```json" in content:
            content = re.search(r'```json\\s*([^`]+)```', content, re.DOTALL).group(1).strip()
        elif "```" in content:
            content = re.search(r'```\\s*([^`]+)```', content, re.DOTALL).group(1).strip()
        
        # Parse JSON
        parsed = json.loads(content)
        if isinstance(parsed, dict):
            # Clamp all values to 0.0-1.0 range and return metrics that exist in EVALUATION_METRICS
            result = {}
            for metric in EVALUATION_METRICS:
                if metric in parsed:
                    result[metric] = max(0.0, min(1.0, float(parsed[metric])))
                else:
                    result[metric] = 0.0
            return result
        
        return {metric: 0.0 for metric in EVALUATION_METRICS}
    except Exception as e:
        logger.error(f"JSON parsing error: {e}")
        logger.error(f"Raw response: {response_content}")
        return {metric: 0.0 for metric in EVALUATION_METRICS}"""

LOAD_TRAINING_DATA = """# Load training data
    training_data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'training_data.json')
    with open(training_data_path, 'r') as f:
        training_data = json.load(f)"""

IMPORT_TOOL_MODULE = """# Import tool module
    spec = importlib.util.spec_from_file_location("tool_module", program)
    tool_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tool_module)"""

INITIALIZE_OPENAI = """# Initialize OpenAI client
    client = openai.OpenAI()"""

CRITICAL_OUTPUT_REQUIREMENT = """CRITICAL OUTPUT REQUIREMENT:
Return ONLY valid Python code. No explanations, no markdown code blocks, no text before or after.
Start with 'import os' and end with the final closing brace of the evaluate function.
DO NOT include ```python or ``` or any other markdown formatting.
DO NOT include any explanatory text before or after the code."""

SCORING_GUIDELINES_TEMPLATE = """SCORING GUIDELINES:
- 0.0-0.3: {poor}
- 0.4-0.6: {average}
- 0.7-0.8: {good}
- 0.9-1.0: {excellent}"""

# Main prompts
FUNCTION_ANALYSIS_PROMPT = """Analyze this Python function and determine the most relevant evaluation metrics.

##FUNCTION CODE:
{source_code}

##TRAINING DATA SAMPLE:
{training_sample}

##TASK: 
# Provide a concise analysis and 1 to 5 metrics for evaluating this function's output quality.

##INSTRUCTIONS:
1. Analyze what this function does based on its name, code, and purpose
2. Determine if this is a QUANTITATIVE/ANALYTICAL function or a CREATIVE/CONTENT function
3. Choose 1 to 5 metrics that are most relevant for evaluating this specific tool. Choose only as many as are relevant, and no more. Less is more.
4. For quantitative functions, focus on correctness, completeness, and logical soundness. If the function is purely quantitative, then the metric can be correctness.
5. For creative functions, focus on quality, relevance, and subjective measures

##FUNCTION TYPE CLASSIFICATION:
- QUANTITATIVE/ANALYTICAL: Functions that perform calculations, data analysis, technical analysis, mathematical operations, statistical computations, financial analysis, etc.
- CREATIVE/CONTENT: Functions that generate text, creative content, marketing materials, essays, stories, etc.
- PROMPT/TEMPLATE: Pure prompt templates, system messages, instruction strings, or prompt engineering patterns without executable code.
- CLASSIFICATION: Intent classification, category classification, or classification tasks with expected outputs.

##EXAMPLE METRICS BY FUNCTION TYPE:
- Quantitative/Analytical functions: correctness, completeness, accuracy, consistency, precision, logical_soundness, data_coverage, calculation_validity
- Content/Creative functions: quality, relevance, engagement, authenticity, creativity, coherence, persuasiveness, clarity
- Prompt/Template functions: clarity, specificity, effectiveness, completeness, instruction_quality, prompt_structure, task_alignment, response_guidance
- Research tools: accuracy, depth, insight, completeness, relevance, comprehensiveness
- Classification: accuracy, precision, consistency, confidence
- Code generation: correctness, completeness, maintainability, readability

##RESPONSE FORMAT:
Return your response in this exact JSON format:
{{
    "function_description": "Brief description of what this function does",
    "function_type": "quantitative" or "creative" or "prompt" or "classification",
    "metrics": [<metrics that you chose>]
}}

Be specific and choose metrics that truly matter for this function's output quality.

Return ONLY JSON, no explanations or markdown."""


VALIDATION_PROMPT = """You are an expert at evaluating Python code. Analyze this evaluator code and check for common issues:

##CODE TO VALIDATE:
```python
{code}
```

##CHECK FOR THESE ISSUES:
1. Syntax errors (try compiling the code)
2. Import errors (wrong imports like 'Message' instead of proper OpenAI API)
3. API usage errors (using langchain instead of raw OpenAI API)
4. Missing error handling
5. Incorrect response parsing
6. Missing required functions (def evaluate)

##RESPONSE FORMAT:
Return a JSON with:
- "is_valid": boolean (true if code is correct)
- "issues": list of strings describing problems found
- "suggested_fixes": a bullet describing what fixes need to be made with examples

If the code needs fixing, provide the complete corrected code in suggested_fix.
Use raw OpenAI API: openai.OpenAI(), client.chat.completions.create(), response.choices[0].message.content

Return ONLY JSON, no explanations. Don't return any markdown or json```"""


def get_quantitative_evaluator_prompt(tool_name: str, function_description: str, 
                                     metrics: list, source_code: str, training_data: list, 
                                     feedback_section: str) -> str:
    """Create specialized prompt for quantitative/analytical functions"""
    
    scoring_guidelines = SCORING_GUIDELINES_TEMPLATE.format(
        poor="Major algorithmic errors, incorrect implementation, or broken logic",
        average="Generally correct algorithm but minor implementation issues or incomplete handling",
        good="Well-implemented algorithm with correct logic and proper output",
        excellent="Perfect algorithmic implementation with comprehensive error handling"
    )
    
    return f"""
###ROLE: 
You are an expert at generating evaluators for python functions.
Create a Python evaluator for the function named: {tool_name}

##FUNCTION ANALYSIS:
{function_description}

##FUNCTION TYPE: ALGORITHMIC/QUANTITATIVE
This function performs algorithmic calculations, data analysis, or technical computations. 
The evaluator should analyze THE FUNCTION IMPLEMENTATION ITSELF, not just outputs.

##PRE-DETERMINED METRICS (use these exactly):
{metrics}
{feedback_section}

##CRITICAL REQUIREMENTS:
1. The evaluator file should ALWAYS have a function with the signature: def evaluate(program) -> dict:
2. Load training data: os.path.join(os.path.dirname(os.path.abspath(__file__)), 'training_data.json')
3. Import tool: importlib.util.spec_from_file_location("tool_module", program)
4. READ AND ANALYZE THE SOURCE CODE ITSELF using inspect.getsource()
5. Call tool with inputs AND analyze the implementation logic
6. Use LLM to evaluate algorithmic correctness, implementation quality

##SOURCE CODE OF THE FUNCTION TO EVALUATE:
```python
{source_code}
```

##TRAINING DATA STRUCTURE:
This is a sample showing a single training case. The training data is a list of dictionaries, each containing the input parameters for the function.
{training_data[0] if training_data else {{}}}

##ALGORITHMIC EVALUATION WORKFLOW:
For each training case:
1. Extract and analyze the function source code using inspect.getsource()
2. Call the function with each test input in the training data
3. Analyze both the IMPLEMENTATION and the OUTPUT
4. Create comprehensive evaluation examining algorithm correctness
5. Parse LLM response with robust JSON parser
6. Accumulate scores and return averages

##REQUIRED STRUCTURE:
```python
{COMMON_IMPORTS}
EVALUATION_METRICS = {metrics}

{PARSE_JSON_FUNCTION}

def evaluate(program) -> dict:
    {LOAD_TRAINING_DATA}
    
    {IMPORT_TOOL_MODULE}
    
    # Get tool function
    tool_function = getattr(tool_module, '{tool_name}')
    
    # CRITICAL: Extract source code for algorithm analysis
    function_source = inspect.getsource(tool_function)
    
    {INITIALIZE_OPENAI}
    
    # Process each test case with BOTH code analysis and output evaluation
    all_scores = []
    
    for input_data in training_data:
        try:
            # Call the tool function
            result = tool_function(**input_data)
            
            # Create evaluation prompt
            eval_prompt = f\"\"\"You are evaluating an ALGORITHMIC/QUANTITATIVE function.
    Analyze BOTH the implementation code AND the output for correctness.

    FUNCTION TO EVALUATE: {tool_name}
    INPUT PARAMETERS: {{input_data}}
    FUNCTION SOURCE CODE:
    {{function_source}}

    GENERATED OUTPUT: {{result}}

    ALGORITHMIC ANALYSIS CRITERIA:
    1. IMPLEMENTATION REVIEW:
    - Is the algorithm logic mathematically sound?
    - Are calculations implemented correctly?
    - Are formulas and computations accurate?
    - Is input validation adequate?

    2. OUTPUT ANALYSIS:
    - Does output match expected format/structure?
    - Are calculated values reasonable for given inputs?
    - Is the output complete with all expected fields?

    3. CODE QUALITY:
    - Are there logical errors in the implementation?
    - Is error handling appropriate?
    - Are edge cases considered?

    Evaluate on these EXACT metrics (0.0-1.0 scale):
    {chr(10).join([f"- {metric}: Assess this aspect of both code implementation and output" for metric in metrics])}

    {scoring_guidelines}

    Focus on output correctness.
    Return ONLY JSON with exact metric names.\"\"\"
                
                # Get evaluation from LLM
                response = client.chat.completions.create(
                    model="gpt-5",
                    messages=[{{"role": "user", "content": eval_prompt}}],
                    temperature=0.0
                )
                
                # Parse the JSON response
                case_scores = parse_json_response(response.choices[0].message.content)
                all_scores.append(case_scores)
                
            except Exception as e:
                logger.error(f"Error evaluating case {{input_data}}: {{e}}")
                all_scores.append({{metric: 0.0 for metric in EVALUATION_METRICS}})
        
        # Calculate average scores
        if not all_scores:
            return {{metric: 0.0 for metric in EVALUATION_METRICS}}
        
        average_scores = {{metric: sum(scores[metric] for scores in all_scores) / len(all_scores) for metric in EVALUATION_METRICS}}
        return average_scores
    ```

    IMPORTANT: This evaluator must test the outputs of the function. It should not analyze the function implementation itself.

    {CRITICAL_OUTPUT_REQUIREMENT}"""


def get_prompt_evaluator_prompt(tool_name: str, function_description: str, 
                              metrics: list, source_code: str, training_data: list, 
                              feedback_section: str) -> str:
    """Create specialized prompt for prompt/template optimization"""
    
    scoring_guidelines = SCORING_GUIDELINES_TEMPLATE.format(
        poor="Poor prompt design, unclear instructions, ineffective output",
        average="Adequate prompt but room for improvement",
        good="Well-crafted prompt producing good results",
        excellent="Exceptional prompt engineering with optimal output"
    )
    
    return f"""Create a Python evaluator for OpenEvolve prompt optimization tool: {tool_name}

##FUNCTION ANALYSIS:
{function_description}

##FUNCTION TYPE: PROMPT/TEMPLATE
This is a prompt template for LLM optimization. The evaluator must test the prompt by:
1. Formatting the prompt with training data parameters
2. Sending the formatted prompt to an LLM
3. Evaluating the LLM's response quality and prompt effectiveness

##PRE-DETERMINED METRICS (use these exactly):
{metrics}

##SPECIAL HANDLING FOR INTENT CLASSIFICATION:
If this is an intent classification prompt (determines user intent categories), 
focus primarily on ACCURACY - how well the prompt guides the LLM to classify correctly.
Compare the LLM output directly with expected_intent for exact match scoring.
{feedback_section}

##CRITICAL REQUIREMENTS FOR PROMPT EVALUATION:
1. Function signature: def evaluate(program) -> dict:
2. Load training data: os.path.join(os.path.dirname(os.path.abspath(__file__)), 'training_data.json')  
3. EXTRACT PROMPT TEMPLATE from the evolve_target.py file
4. FORMAT prompt with training data parameters (use .format() or f-strings)
5. SEND formatted prompt to LLM to get response
6. EVALUATE the LLM response quality with a separate evaluation prompt

##TRAINING DATA STRUCTURE:
Training data should contain parameters that the prompt accepts, NOT input/output pairs.
Expected format: [{{"param1": "value1", "param2": "value2", ...}}]
Example: {training_data[0] if training_data else {{}}}

##COMPLETE EVALUATOR STRUCTURE:
```python
{COMMON_IMPORTS}
EVALUATION_METRICS = {metrics}

{PARSE_JSON_FUNCTION}

def evaluate(program) -> dict:
    {LOAD_TRAINING_DATA}
    
    # Extract prompt template from file
    with open(program, 'r') as f:
        file_content = f.read()
    
    # Find prompt constant
    prompt_match = re.search(r'([A-Z_]+)\\s*=\\s*\"\"\"(.*?)\"\"\"', file_content, re.DOTALL)
    if not prompt_match:
        logger.error("Could not find prompt template in file")
        return {{metric: 0.0 for metric in EVALUATION_METRICS}}
    
    prompt_template = prompt_match.group(2).strip()
    
    {INITIALIZE_OPENAI}
    
    # Evaluate prompt with each test case
    scores = {{metric: 0.0 for metric in EVALUATION_METRICS}}
    num_cases = len(training_data)
    
    for test_case in training_data:
        # Format prompt with parameters
        try:
            formatted_prompt = prompt_template.format(**test_case)
        except:
            formatted_prompt = prompt_template
            
        # Send to LLM
        response = client.chat.completions.create(
            model="gpt-5",
            messages=[{{"role": "user", "content": formatted_prompt}}],
            temperature=0.0
        )
        llm_output = response.choices[0].message.content
        
        # Create evaluation prompt
        eval_prompt = f\"\"\"You are evaluating a PROMPT TEMPLATE and the output it generates.

    ORIGINAL PROMPT TEMPLATE:
    {{prompt_template}}

    TEST PARAMETERS: {{test_case}}
    FORMATTED PROMPT SENT TO LLM:
    {{formatted_prompt}}

    LLM RESPONSE TO THE PROMPT:
    {{llm_output}}

    EVALUATION CRITERIA:
    1. PROMPT QUALITY:
    - Is the prompt clear and well-structured?
    - Does it provide adequate context and instructions?
    - Is the language precise and unambiguous?

    2. OUTPUT EFFECTIVENESS:
    - Does the LLM response align with the prompt's intent?
    - Is the output relevant and appropriate?
    - Does the prompt successfully guide the LLM behavior?

    3. PROMPT ENGINEERING:
    - Does the prompt handle the input parameters well?
    - Is the prompt format optimal for LLM understanding?
    - Could the prompt be improved for better results?

    Rate on these EXACT metrics (0.0-1.0 scale):
    {chr(10).join([f"- {metric}: Evaluate prompt effectiveness for this aspect" for metric in metrics])}

    {scoring_guidelines}

    Return ONLY JSON with exact metric names.\"\"\"
            
            # Get evaluation
            eval_response = client.chat.completions.create(
                model="gpt-5",
                messages=[{{"role": "user", "content": eval_prompt}}],
                temperature=0.0
            )
            case_scores = parse_json_response(eval_response.choices[0].message.content)
            
            # Accumulate scores
            for metric in EVALUATION_METRICS:
                scores[metric] += case_scores[metric]
        
        # Average scores
        for metric in EVALUATION_METRICS:
            scores[metric] /= num_cases
            
        return scores
    ```

    IMPORTANT: This evaluator FORMATS the prompt with parameters, SENDS it to an LLM, 
    and EVALUATES the response quality. It does NOT execute functions.

{CRITICAL_OUTPUT_REQUIREMENT}"""


def get_content_evaluator_prompt(tool_name: str, function_description: str, 
                               metrics: list, source_code: str, training_data: list, 
                               feedback_section: str) -> str:
    """Create specialized prompt for creative/content functions"""
    
    scoring_guidelines = SCORING_GUIDELINES_TEMPLATE.format(
        poor="Poor quality, irrelevant, or unusable content",
        average="Mediocre content (most outputs should score here)",
        good="Good quality, relevant, engaging content",
        excellent="Exceptional content (very rare)"
    )
    
    return f"""Create a Python evaluator for OpenEvolve content tool: {tool_name}

    ##FUNCTION ANALYSIS:
    {function_description}

    ##FUNCTION TYPE: CREATIVE/CONTENT
    This function generates creative content, text, or subjective materials. The evaluator should focus on:
    - Quality and coherence of generated content
    - Relevance to input requirements  
    - Engagement and effectiveness
    - Authenticity and originality

    ##PRE-DETERMINED METRICS (use these exactly):
    {metrics}
    {feedback_section}

    ##CONTENT EVALUATION APPROACH:
    For content functions, evaluation should assess:
    1. QUALITY: Is the content well-written and coherent?
    2. RELEVANCE: Does it meet the specified requirements?
    3. ENGAGEMENT: Is it compelling and interesting?
    4. AUTHENTICITY: Does it feel genuine and original?

    ##CRITICAL REQUIREMENTS:
    1. Function signature: def evaluate(program) -> dict:
    2. Load training data: os.path.join(os.path.dirname(os.path.abspath(__file__)), 'training_data.json')
    3. Import tool: importlib.util.spec_from_file_location("tool_module", program)
    4. Call tool with CORRECT function name: {tool_name}
    5. Use LLM to evaluate content quality subjectively

    ##TOOL FUNCTION SIGNATURE:
    {source_code[source_code.find('def '):source_code.find(':', source_code.find('def '))+1] if 'def ' in source_code else 'def unknown_function():'}

    ##TRAINING DATA STRUCTURE:
    {training_data[0] if training_data else {{}}}

    ##COMPLETE EVALUATOR STRUCTURE:
    ```python
    {COMMON_IMPORTS}
    EVALUATION_METRICS = {metrics}

    {PARSE_JSON_FUNCTION}

    def evaluate(program) -> dict:
        {LOAD_TRAINING_DATA}
        
        {IMPORT_TOOL_MODULE}
        
        # Get tool function
        tool_function = getattr(tool_module, '{tool_name}')
        
        {INITIALIZE_OPENAI}
        
        # Process each test case
        all_scores = []
        
        for input_data in training_data:
            try:
                # Call the tool function
                generated_content = tool_function(**input_data)
                
                # Handle tuple returns
                if isinstance(generated_content, tuple):
                    generated_content = generated_content[0]
                
                # Create evaluation prompt
                eval_prompt = f\"\"\"You are a STRICT content evaluator. Be critical and harsh in scoring.

    Evaluate this generated content on 0.0-1.0 scale:

    Generated Content: {{generated_content}}
    Original Request: {{input_data}}

    Rate on these EXACT metrics:
    {chr(10).join([f"- {metric}" for metric in metrics])}

    {scoring_guidelines}

    Be harsh. Most content is mediocre. Return ONLY JSON.\"\"\"
                
                # Get evaluation from LLM
                response = client.chat.completions.create(
                    model="gpt-5",
                    messages=[{{"role": "user", "content": eval_prompt}}],
                    temperature=0.0
                )
                
                # Parse the JSON response
                case_scores = parse_json_response(response.choices[0].message.content)
                all_scores.append(case_scores)
                
            except Exception as e:
                logger.error(f"Error evaluating case {{input_data}}: {{e}}")
                all_scores.append({{metric: 0.0 for metric in EVALUATION_METRICS}})
        
        # Calculate average scores
        if not all_scores:
            return {{metric: 0.0 for metric in EVALUATION_METRICS}}
        
        average_scores = {{metric: sum(scores[metric] for scores in all_scores) / len(all_scores) for metric in EVALUATION_METRICS}}
        return average_scores
    ```

    {CRITICAL_OUTPUT_REQUIREMENT}"""