#!/usr/bin/env python3
"""
Enhanced extractor that can find both @evolve() decorators and commented #@evolve() decorators.
"""

import ast
import re
from pathlib import Path
from typing import Dict, List, Any, Optional


def find_commented_evolve_decorators(file_path: str) -> List[Dict[str, Any]]:
    """Find constants/variables with commented @evolve() decorators"""
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    lines = content.split('\n')
    found_targets = []
    
    for i, line in enumerate(lines):
        # Look for commented evolve decorators
        if re.match(r'^\s*#\s*@evolve\(', line.strip()):
            # Found a commented decorator, look for the next non-comment line
            next_line_idx = i + 1
            
            # Skip empty lines and comments
            while next_line_idx < len(lines):
                next_line = lines[next_line_idx].strip()
                if next_line and not next_line.startswith('#'):
                    break
                next_line_idx += 1
            
            if next_line_idx < len(lines):
                next_line = lines[next_line_idx]
                
                # Check if it's a variable assignment (constant or enum value)
                assignment_match = re.match(r'^\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*=', next_line.strip())
                if assignment_match:
                    var_name = assignment_match.group(1)
                    
                    # Extract the value (could be multiline string)
                    var_value = extract_variable_value(lines, next_line_idx)
                    
                    found_targets.append({
                        'type': 'constant',
                        'name': var_name,
                        'value': var_value,
                        'line_number': next_line_idx + 1,
                        'decorator_line': i + 1,
                        'file_path': file_path
                    })
                    
    
    return found_targets


def extract_variable_value(lines: List[str], start_idx: int) -> str:
    """Extract the full value of a variable assignment (handles multiline strings)"""
    
    start_line = lines[start_idx]
    
    # Check for multiline string FIRST (triple quotes) - handle both """ and '''
    if '"""' in start_line or "'''" in start_line:
        result_lines = []
        quote_type = '"""' if '"""' in start_line else "'''"
        quote_count = start_line.count(quote_type)
        
        # Start collecting the string - include the opening quotes
        if '=' in start_line:
            # Get everything after the = sign
            assignment_part = start_line.split('=', 1)[1]
            result_lines.append(assignment_part.rstrip('\n'))
        else:
            result_lines.append(start_line.rstrip('\n'))
        
        if quote_count == 1:  # Opening quotes only - multiline string
            # Find closing quotes
            for i in range(start_idx + 1, len(lines)):
                line_content = lines[i]  # Don't strip newlines, preserve content
                result_lines.append(line_content)
                if quote_type in lines[i]:
                    break
        elif quote_count >= 2:  # Complete string on one line
            # Already added above
            pass
        
        return '\n'.join(result_lines)
    
    # Check if it's a simple assignment (single line string)
    elif '=' in start_line and (start_line.strip().endswith('"') or start_line.strip().endswith("'")):
        # Single line string
        return start_line.split('=', 1)[1].strip()
    
    # For other cases, just return the line
    return start_line


def save_commented_evolve_target(target_info: Dict[str, Any], output_dir: str = "evolution/tools"):
    """Save a commented evolve target as a tool"""
    
    output_path = Path(output_dir)
    tool_name = target_info['name']  # Preserve original case
    tool_dir = output_path / tool_name
    tool_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate the evolve_target.py file
    tool_code = f'''"""
Tool: {target_info['name']}
Extracted from: {Path(target_info['file_path']).name}
Type: Prompt/Template Constant

This is a prompt/template optimization target.
"""

# The target prompt/template for evolution (preserving original signature)
{target_info['name']} = {target_info['value']}


def {tool_name}():
    """Return the optimized prompt/template"""
    return {target_info['name']}


if __name__ == "__main__":
    # Test the prompt
    print({tool_name}())
'''
    
    # Save the tool file
    tool_file = tool_dir / "evolve_target.py" 
    with open(tool_file, 'w') as f:
        f.write(tool_code)
    
    # Save metadata
    metadata = {
        'name': target_info['name'],
        'type': 'prompt_template',
        'description': f"Prompt/template constant: {target_info['name']}",
        'category': 'prompt_optimization',
        'original_file': target_info['file_path'],
        'decorator_line': target_info['decorator_line'],
        'variable_line': target_info['line_number'],
        'extraction_method': 'commented_decorator'
    }
    
    metadata_file = tool_dir / "metadata.json"
    with open(metadata_file, 'w') as f:
        import json
        json.dump(metadata, f, indent=2)
    
    return tool_name


def extract_commented_evolve_from_file(file_path: str, output_dir: str = "evolution/tools"):
    """Extract all commented @evolve() targets from a file"""
    
    
    if not Path(file_path).exists():
        print(f"❌ File not found: {file_path}")
        return []
    
    targets = find_commented_evolve_decorators(file_path)
    extracted_tools = []
    
    for target in targets:
        tool_name = save_commented_evolve_target(target, output_dir)
        extracted_tools.append(tool_name)
    
    return extracted_tools


def test_commented_extraction():
    """Test the commented decorator extraction"""
    
    # Test with the marketing tools file
    marketing_file = "../../agent_rl/backend/src/marketing/tools.py"
    
    if Path(marketing_file).exists():
        extracted = extract_commented_evolve_from_file(marketing_file)
        return extracted
    else:
        print(f"❌ Test file not found: {marketing_file}")
        return []


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
        output_dir = sys.argv[2] if len(sys.argv) > 2 else "evolution/tools"
        extract_commented_evolve_from_file(file_path, output_dir)
    else:
        # Run test
        test_commented_extraction()