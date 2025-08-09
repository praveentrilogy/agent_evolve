GENERATE_PROMPT_EVALUATOR_PROMPT = """
You are to generate Python code that will evaluate the behavior of a specific function
within a given Python file.

You will be provided:
1. The **prompt** to evaluate.
2. The **training data** to evaluate the prompt against.

Your generated code should:
- Import the prompt dynamically from the provided file path.
- Run a predefined set of **test cases** (you choose reasonable ones based on the function name and expected purpose).
- Capture the output and compare it against **expected results**.
- Print a clear, human-readable evaluation report:
  - Test case inputs
  - Expected outputs
  - Actual outputs
  - Pass/Fail status for each test
  - Overall pass rate
- Handle and report any exceptions without stopping the evaluation.
- Be self-contained and runnable as a standalone Python script.

Format the generated code so that:
- The evaluator function is named `evaluate(file_path: str, function_name: str) -> None`.
- The test cases are defined inside the evaluator for simplicity.

Now, using the above requirements, generate the complete Python evaluator function code.

## CRITICAL OUTPUT REQUIREMENT:
  - The evaluator function must be named `evaluate` and return a dictionary with the following keys:
  - `test_cases`: A list of test cases.
  - `pass_rate`: The pass rate of the evaluator.
  - `overall_pass_rate`: The overall pass rate of the evaluator.
  - `test_case_results`: A list of test case results.
  - `test_case_results_summary`: A summary of the test case results.

## Prompt:
{prompt_content}

## Sample Training Data:
{training_data}
"""
