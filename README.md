# Agent Evolve

A comprehensive toolkit for evolving and tracking AI agents through evolutionary programming and automated optimization.

## Features

- 🧬 **Evolutionary Optimization**: Automatically improve your AI agents using OpenEvolve
- 📊 **Interactive Dashboard**: Real-time tracking and visualization of evolution progress
- 🎯 **Smart Extraction**: Automatically find and extract functions marked for evolution
- 🔧 **Auto-Generation**: Generate evaluators, training data, and configurations automatically
- 📈 **Metrics Tracking**: Comprehensive metrics collection and analysis
- 🚀 **CLI Interface**: Easy-to-use command-line tools for all operations

## Quick Start

### Installation

```bash
pip install agent-evolve
```

### Basic Usage

1. **Mark functions for evolution** by adding the `@evolve` decorator:

```python
from agent_evolve import evolve

@evolve()
def my_function(x: int) -> str:
    return f"Hello {x}"
```

2. **Extract evolution targets**:

```bash
agent-evolve extract /path/to/your/code
```

3. **Run the evolution pipeline**:

```bash
agent-evolve pipeline my_function
```

4. **View results in the dashboard**:

```bash
agent-evolve dashboard
```

## CLI Commands

- `extract` - Extract functions marked with @evolve decorator
- `generate-training-data` - Generate training data for extracted functions
- `generate-evaluators` - Generate evaluation functions
- `generate-configs` - Generate OpenEvolve configuration files
- `evolve` - Run evolution optimization on a specific tool
- `pipeline` - Run the complete pipeline (training data → evaluator → config → evolution)
- `dashboard` - Launch the interactive Streamlit dashboard
- `analyze` - Analyze tracking data

## Dashboard Features

- **Evolution Pipeline**: Step-by-step progress tracking
- **Code Comparison**: Side-by-side diffs of original vs evolved code
- **Metrics & Timeline**: Performance improvements over time
- **Interactive Tools**: Generate evaluators, run evolution, and more

## Requirements

- Python 3.8+
- OpenAI API key (for evaluator generation)
- OpenEvolve package

## License

MIT License - see LICENSE file for details.

## Contributing

Contributions welcome! Please read our contributing guidelines and submit pull requests.