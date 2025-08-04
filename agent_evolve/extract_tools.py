#!/usr/bin/env python3
"""
Tool Extraction Script for Agent Evolution Framework

This script analyzes Python files in a target directory to find tool functions
and extracts them with all their dependencies into separate files for evaluation
and improvement.

Usage:
    python extract_tools.py <target_directory>
"""

import ast
import os
import sys
import shutil
from pathlib import Path
from typing import Set, Dict, List, Tuple, Any
from collections import defaultdict
import inspect


class DependencyAnalyzer(ast.NodeVisitor):
    """AST visitor to analyze function dependencies"""
    
    def __init__(self, source_code: str):
        self.source_code = source_code
        self.source_lines = source_code.split('\n')
        self.imports = []
        self.functions = {}
        self.classes = {}
        self.variables = {}
        self.constants = {}
        self.function_calls = defaultdict(set)
        self.current_function = None
        
    def visit_Import(self, node):
        for alias in node.names:
            self.imports.append({
                'type': 'import',
                'module': alias.name,
                'alias': alias.asname,
                'line': node.lineno
            })
        self.generic_visit(node)
    
    def visit_ImportFrom(self, node):
        for alias in node.names:
            self.imports.append({
                'type': 'from_import',
                'module': node.module,
                'name': alias.name,
                'alias': alias.asname,
                'line': node.lineno
            })
        self.generic_visit(node)
    
    def visit_FunctionDef(self, node):
        func_info = {
            'name': node.name,
            'args': [arg.arg for arg in node.args.args],
            'docstring': ast.get_docstring(node),
            'line_start': node.lineno,
            'line_end': node.end_lineno,
            'source': self._get_source_lines(node.lineno, node.end_lineno),
            'calls': set()
        }
        
        self.functions[node.name] = func_info
        
        # Track type hints in function signature
        old_function = self.current_function
        self.current_function = node.name
        
        # Check argument type annotations
        for arg in node.args.args:
            if arg.annotation:
                self._extract_type_annotation(arg.annotation)
        
        # Check return type annotation
        if node.returns:
            self._extract_type_annotation(node.returns)
        
        self.generic_visit(node)
        self.current_function = old_function
    
    def _extract_type_annotation(self, annotation):
        """Extract names from type annotations"""
        if isinstance(annotation, ast.Name):
            if self.current_function:
                self.function_calls[self.current_function].add(annotation.id)
        elif isinstance(annotation, ast.Attribute):
            if isinstance(annotation.value, ast.Name):
                if self.current_function:
                    self.function_calls[self.current_function].add(annotation.value.id)
        elif isinstance(annotation, ast.Subscript):
            # Handle things like List[str], Dict[str, int]
            if isinstance(annotation.value, ast.Name):
                if self.current_function:
                    self.function_calls[self.current_function].add(annotation.value.id)
    
    def visit_ClassDef(self, node):
        class_info = {
            'name': node.name,
            'line_start': node.lineno,
            'line_end': node.end_lineno,
            'source': self._get_source_lines(node.lineno, node.end_lineno),
            'methods': []
        }
        
        # Extract method names
        for item in node.body:
            if isinstance(item, ast.FunctionDef):
                class_info['methods'].append(item.name)
        
        self.classes[node.name] = class_info
        
        # Track base classes as dependencies
        if self.current_function is None:  # Only for module-level classes
            for base in node.bases:
                if isinstance(base, ast.Name):
                    # Track the base class name as a dependency
                    if hasattr(self, '_current_class_dependencies'):
                        self._current_class_dependencies.add(base.id)
                    else:
                        # Create a pseudo-function entry for class dependencies
                        if node.name not in self.function_calls:
                            self.function_calls[node.name] = set()
                        self.function_calls[node.name].add(base.id)
        
        self.generic_visit(node)
    
    def visit_Assign(self, node):
        # Track variable assignments at module level
        if self.current_function is None:
            for target in node.targets:
                if isinstance(target, ast.Name):
                    # For assignments, we need to get the entire assignment including multi-line values
                    end_line = node.end_lineno if node.end_lineno else node.lineno
                    var_info = {
                        'name': target.id,
                        'line': node.lineno,
                        'source': self._get_source_lines(node.lineno, end_line)
                    }
                    
                    # Check if it's a constant (uppercase)
                    if target.id.isupper():
                        self.constants[target.id] = var_info
                    else:
                        self.variables[target.id] = var_info
        
        self.generic_visit(node)
    
    def visit_Call(self, node):
        if self.current_function:
            # Track function calls
            if isinstance(node.func, ast.Name):
                self.function_calls[self.current_function].add(node.func.id)
            elif isinstance(node.func, ast.Attribute):
                # Handle method calls like obj.method()
                if isinstance(node.func.value, ast.Name):
                    call_name = f"{node.func.value.id}.{node.func.attr}"
                    self.function_calls[self.current_function].add(call_name)
        
        self.generic_visit(node)
    
    def visit_Name(self, node):
        # Track variable and name usage in functions
        if self.current_function and isinstance(node.ctx, ast.Load):
            # Track usage of module-level variables and constants
            if node.id in self.variables or node.id in self.constants:
                self.function_calls[self.current_function].add(node.id)
            # Track usage of any name that could be an imported module/function
            else:
                self.function_calls[self.current_function].add(node.id)
        
        self.generic_visit(node)
    
    def _get_source_lines(self, start_line: int, end_line: int) -> str:
        """Extract source code lines"""
        if end_line is None:
            end_line = start_line
        
        lines = self.source_lines[start_line-1:end_line]
        return '\n'.join(lines)


class ToolExtractor:
    """Main tool extraction class"""
    
    def __init__(self, target_dir: str, output_dir: str = None):
        self.target_dir = Path(target_dir)
        self.output_dir = Path(output_dir) if output_dir else Path("evolution/tools")
        self.analyzers = {}  # filename -> DependencyAnalyzer
    
    def extract_tools(self):
        """Main extraction method"""
        print(f"Analyzing tools in: {self.target_dir}")
        print(f"Output directory: {self.output_dir}")
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Find and analyze Python files
        python_files = list(self.target_dir.glob("**/*.py"))
        print(f"Found {len(python_files)} Python files")
        
        # Analyze each file
        for py_file in python_files:
            print(f"Analyzing: {py_file}")
            self._analyze_file(py_file)
        
        # Extract tools from each file
        for filename, analyzer in self.analyzers.items():
            self._extract_tools_from_analyzer(filename, analyzer)
    
    def _analyze_file(self, filepath: Path):
        """Analyze a single Python file"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                source_code = f.read()
            
            tree = ast.parse(source_code)
            analyzer = DependencyAnalyzer(source_code)
            analyzer.visit(tree)
            
            self.analyzers[filepath.name] = analyzer
            
        except Exception as e:
            print(f"Error analyzing {filepath}: {e}")
    
    def _extract_tools_from_analyzer(self, filename: str, analyzer: DependencyAnalyzer):
        """Extract tools from a single analyzer"""
        functions = analyzer.functions
        
        if not functions:
            print(f"No functions found in {filename}")
            return
        
        # Filter for actual tools (exclude nodes, decorators, wrappers, etc.)
        tools = self._filter_tools(functions)
        
        if not tools:
            print(f"No tools found in {filename}")
            return
        
        print(f"Found {len(tools)} tools in {filename}")
        
        for func_name, func_info in tools.items():
            print(f"Extracting tool: {func_name}")
            self._create_tool_file(filename, func_name, func_info, analyzer)
    
    def _filter_tools(self, functions: Dict) -> Dict:
        """Filter functions to only include actual tools"""
        tools = {}
        
        for func_name, func_info in functions.items():
            # Skip private functions
            if func_name.startswith('_'):
                continue
            
            # Skip node functions (end with _node)
            if func_name.endswith('_node'):
                continue
            
            # Skip routing functions
            if 'route' in func_name.lower():
                continue
            
            # Skip decorator-related functions
            if func_name in ['decorator', 'wrapper']:
                continue
            
            # Skip internal framework functions
            if func_name in ['shutdown', 'get_rl_tracker', 'track_node', 'make_serializable']:
                continue
            
            # Skip tracking functions (these are framework utilities, not tools)
            if func_name.startswith('track_'):
                continue
            
            # Skip update functions (these are usually internal)
            if func_name.startswith('update_'):
                continue
            
            # Include the rest as tools
            tools[func_name] = func_info
        
        return tools
    
    def _create_tool_file(self, source_filename: str, func_name: str, func_info: Dict, analyzer: DependencyAnalyzer):
        """Create a standalone tool file"""
        tool_dir = self.output_dir / func_name
        tool_dir.mkdir(exist_ok=True)
        
        tool_file = tool_dir / "tool.py"
        
        # Collect dependencies
        dependencies = self._collect_dependencies(func_name, analyzer)
        
        # Generate tool file content
        content = self._generate_tool_content(
            source_filename, func_name, func_info, dependencies, analyzer
        )
        
        # Write the tool file
        with open(tool_file, 'w', encoding='utf-8') as f:
            f.write(content)
        
        # Create metadata file
        self._create_metadata_file(tool_dir, func_name, func_info, source_filename)
        
        print(f"Created: {tool_file}")
    
    def _collect_dependencies(self, func_name: str, analyzer: DependencyAnalyzer, visited: Set[str] = None) -> Dict[str, Any]:
        """Collect all dependencies for a function"""
        if visited is None:
            visited = set()
        
        # Prevent infinite recursion
        if func_name in visited:
            return {
                'imports': [],
                'functions': {},
                'classes': {},
                'variables': {},
                'constants': {}
            }
        
        visited.add(func_name)
        
        dependencies = {
            'imports': [],
            'functions': {},
            'classes': {},
            'variables': {},
            'constants': {}
        }
        
        # Get direct dependencies
        called_items = analyzer.function_calls.get(func_name, set())
        
        # Collect all items referenced by this function and its dependencies
        all_referenced_items = set(called_items)
        
        # Also check for classes that might be referenced in the function body
        if func_name in analyzer.functions:
            func_source = analyzer.functions[func_name]['source']
            source_names = self._extract_names_from_source(func_source)
            all_referenced_items.update(source_names)
        
        # Add referenced functions
        for item in called_items:
            if item in analyzer.functions and item not in visited:
                dependencies['functions'][item] = analyzer.functions[item]
                # Recursively add dependencies of called functions
                sub_deps = self._collect_dependencies(item, analyzer, visited.copy())
                for sub_func, sub_info in sub_deps['functions'].items():
                    if sub_func not in dependencies['functions']:
                        dependencies['functions'][sub_func] = sub_info
                # Add items referenced by sub-dependencies
                sub_called_items = analyzer.function_calls.get(item, set())
                all_referenced_items.update(sub_called_items)
        
        # Add referenced variables and constants
        for item in called_items:
            if item in analyzer.variables:
                dependencies['variables'][item] = analyzer.variables[item]
                # If variable uses imported modules, mark them as referenced
                var_source = analyzer.variables[item]['source']
                all_referenced_items.update(self._extract_names_from_source(var_source))
            elif item in analyzer.constants:
                dependencies['constants'][item] = analyzer.constants[item]
                # If constant uses imported modules, mark them as referenced
                const_source = analyzer.constants[item]['source']
                all_referenced_items.update(self._extract_names_from_source(const_source))
        
        # Add referenced classes and their dependencies
        for item in called_items:
            if item in analyzer.classes:
                dependencies['classes'][item] = analyzer.classes[item]
                # Add dependencies from class definitions (like base classes)
                if item in analyzer.function_calls:
                    class_deps = analyzer.function_calls[item]
                    all_referenced_items.update(class_deps)
        
        # Only add imports that are actually used
        dependencies['imports'] = self._filter_used_imports(all_referenced_items, analyzer.imports)
        
        return dependencies
    
    def _extract_names_from_source(self, source_code: str) -> Set[str]:
        """Extract variable/function names from source code"""
        names = set()
        try:
            tree = ast.parse(source_code)
            for node in ast.walk(tree):
                if isinstance(node, ast.Name):
                    names.add(node.id)
                elif isinstance(node, ast.Attribute):
                    if isinstance(node.value, ast.Name):
                        names.add(node.value.id)
        except:
            # If parsing fails, do simple text analysis
            import re
            # Find potential identifiers
            identifiers = re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', source_code)
            names.update(identifiers)
        
        return names
    
    def _filter_used_imports(self, referenced_items: Set[str], all_imports: List[Dict]) -> List[Dict]:
        """Filter imports to only include those that are actually used"""
        used_imports = []
        
        for imp in all_imports:
            import_used = False
            
            if imp['type'] == 'import':
                # Check if the module or its alias is referenced
                module_name = imp['alias'] if imp['alias'] else imp['module'].split('.')[0]
                if module_name in referenced_items:
                    import_used = True
                # Also check for method calls like module.method
                for item in referenced_items:
                    if '.' in item and item.startswith(module_name + '.'):
                        import_used = True
                        break
            else:  # from_import
                # Check if the imported name or its alias is referenced
                imported_name = imp['alias'] if imp['alias'] else imp['name']
                if imported_name in referenced_items:
                    import_used = True
            
            if import_used:
                used_imports.append(imp)
        
        # Only add essential imports if they're actually referenced in the function
        # This ensures we don't bloat simple functions with unnecessary imports
        
        return used_imports
    
    def _generate_tool_content(self, source_filename: str, func_name: str, func_info: Dict, 
                             dependencies: Dict, analyzer: DependencyAnalyzer) -> str:
        """Generate the content for the tool file"""
        content = []
        
        # Header comment
        content.append(f'"""')
        content.append(f'Tool: {func_name}')
        content.append(f'Extracted from: {source_filename}')
        content.append(f'')
        if func_info.get('docstring'):
            content.append(func_info['docstring'])
        content.append(f'"""')
        content.append('')
        
        # Imports
        for imp in dependencies['imports']:
            if imp['type'] == 'import':
                if imp['alias']:
                    content.append(f"import {imp['module']} as {imp['alias']}")
                else:
                    content.append(f"import {imp['module']}")
            else:  # from_import
                if imp['alias']:
                    content.append(f"from {imp['module']} import {imp['name']} as {imp['alias']}")
                else:
                    content.append(f"from {imp['module']} import {imp['name']}")
        
        if dependencies['imports']:
            content.append('')
        
        # Constants
        for const_name, const_info in dependencies['constants'].items():
            content.append(const_info['source'])
        
        if dependencies['constants']:
            content.append('')
        
        # Variables
        for var_name, var_info in dependencies['variables'].items():
            content.append(var_info['source'])
        
        if dependencies['variables']:
            content.append('')
        
        # Classes
        for class_name, class_info in dependencies['classes'].items():
            content.append(class_info['source'])
            content.append('')
        
        # Helper functions
        for dep_func_name, dep_func_info in dependencies['functions'].items():
            if dep_func_name != func_name:  # Don't include the main function yet
                content.append(dep_func_info['source'])
                content.append('')
        
        # Main function
        content.append(func_info['source'])
        content.append('')
        
        # Add a main block for testing
        content.append('if __name__ == "__main__":')
        content.append('    # Test the tool here')
        content.append(f'    # result = {func_name}(...)')
        content.append('    pass')
        
        return '\n'.join(content)
    
    def _create_metadata_file(self, tool_dir: Path, func_name: str, func_info: Dict, source_filename: str):
        """Create metadata file for the tool"""
        metadata = {
            'name': func_name,
            'source_file': source_filename,
            'docstring': func_info.get('docstring', ''),
            'args': func_info.get('args', []),
            'line_range': f"{func_info.get('line_start', 0)}-{func_info.get('line_end', 0)}"
        }
        
        metadata_file = tool_dir / "metadata.json"
        import json
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)


def main():
    """Main entry point"""
    if len(sys.argv) != 2:
        print("Usage: python extract_tools.py <target_directory>")
        print("Example: python extract_tools.py backend/src/marketing")
        sys.exit(1)
    
    target_dir = sys.argv[1]
    
    if not os.path.exists(target_dir):
        print(f"Error: Directory '{target_dir}' does not exist")
        sys.exit(1)
    
    extractor = ToolExtractor(target_dir)
    extractor.extract_tools()
    
    print("Tool extraction completed!")


if __name__ == "__main__":
    main()