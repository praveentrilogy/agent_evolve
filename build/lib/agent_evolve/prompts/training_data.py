TRAINING_DATA_GENERATION_PROMPT ="""You are an expert at generating training data for AI prompts. 

Given the following prompt template and variables, generate {num_samples} additional diverse training examples.

## Prompt Template:
{prompt_content}

## Template Variables:
{variables}


## Task: Generate {num_samples} additional training examples that follow the same structure as the prompt template and variables. 

## Requirements:
- Make the examples diverse and realistic
- Ensure variable values are appropriate for the prompt context
- Generate examples that would be useful for training an AI model
- Follow the same JSON structure as the variables

## Output Format:
{{
    training_data: [
        {{
            "variable_values": {{
                "variable_name": "value"
            }},
            "rendered_content": "rendered content"
        }}
    ]
}}

## Critical:
- Return ONLY JSON.
- Ensure you match {num_samples} exactly.
- Do not include any other text or comments.
- Do not include any other formatting
"""