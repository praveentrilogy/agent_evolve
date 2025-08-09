import openai
import os
import json
import logging


client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
logger = logging.getLogger(__name__)


def safe_parse_json(content: str, default_value: dict = None, context: str = "") -> dict:
    """
    Safely parse JSON content with error handling.
    
    Args:
        content: The JSON string to parse
        default_value: Default value to return if parsing fails (defaults to empty dict)
        context: Optional context string for better error messages
        
    Returns:
        Parsed JSON as dict, or default_value if parsing fails
    """
    if default_value is None:
        default_value = {}
        
    try:
        return json.loads(content)
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON{f' in {context}' if context else ''}: {e}")
        logger.error(f"Error at position {e.pos}: {e.msg}")
        
        # Log a preview of the problematic area
        if hasattr(e, 'pos') and e.pos is not None:
            start = max(0, e.pos - 50)
            end = min(len(content), e.pos + 50)
            logger.error(f"Content around error: ...{content[start:end]}...")
        else:
            # Log first 200 chars if position unknown
            logger.error(f"Content preview: {content[:200]}...")
            
        return default_value

def get_llm_response(prompt: str, model: str = "gpt-5", temperature: float = 0.0):
    """Get a response from the LLM."""
    if model == "gpt-5":
        temperature = 1
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": prompt}],
        temperature=temperature
    )
    return response.choices[0].message.content



def get_llm_response_json(prompt: str, model: str = "gpt-5", temperature: float = 0.0):
    """Get a response from the LLM in JSON format with better error handling."""
        
    if model == "gpt-5":
        temperature = 1

    logger.info(f"Making LLM call with model={model}")

    logger.info(f"Prompt: {prompt}")

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": prompt}],
        temperature=temperature,
        response_format={"type": "json_object"},
    )
    
    # Check if response was truncated
    if response.choices[0].finish_reason == 'length':
        logger.warning(f"LLM response was truncated due to token limit. Used {response.usage.completion_tokens} tokens.")
        # Use safe parsing with appropriate default for training data
        return safe_parse_json(
            response.choices[0].message.content, 
            default_value={"training_data": []},
            context="truncated LLM response"
        )
    
    # Use safe parsing for normal responses
    return safe_parse_json(
        response.choices[0].message.content,
        default_value={"training_data": []},
        context="LLM response"
    )


