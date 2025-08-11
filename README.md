# Agent Evolve 🧬

**Evolve your AI agents with zero manual effort using automated optimization and real-time tracking.**

Agent Evolve is a comprehensive toolkit that automatically improves AI agents, functions, and prompts through evolutionary programming. Simply add a decorator to your code, and watch as your functions evolve to perform better over time.

## 🚀 Quick Start

### Installation

Clone the repository and install in development mode:

```bash
git clone <repository-url>
cd agent_evolution/agent_evolve
pip install -e .
```

### Add to Your Python Project

1. Navigate to any Python project directory
2. Import and use the `@evolve` decorator:

```python
from agent_evolve import evolve

@evolve()
def my_function(x: int, y: str) -> str:
    """This function will be automatically optimized"""
    return f"Processing {x} with {y}"
```

3. Run your application normally - Agent Evolve will track performance automatically

### Basic Workflow

```bash
# 1. Extract functions marked for evolution
agent-evolve extract

# 2. Generate evaluator for a specific function
agent-evolve generate-evaluator my_function

# 3. Run evolution optimization
agent-evolve evolve my_function

# 4. View results in the dashboard
agent-evolve dashboard
```

## 📊 Interactive Dashboard

Launch the powerful Streamlit dashboard to monitor and manage your AI evolution:

```bash
agent-evolve dashboard
```

The dashboard provides:

### Training Data Management
![Training Data](screenshots/Training%20Data.png)

- **Automated data collection** from your application usage
- **Smart training sample generation** based on real usage patterns
- **Interactive data editing** and validation
- **Export capabilities** for external analysis

### Evolution Progress Tracking
![Performance Timeline](screenshots/Evolved%20code%20performance%20timeline.png)

- **Real-time metrics visualization** showing improvement over time
- **Performance comparison** between original and evolved versions
- **Checkpoint management** for resuming evolution runs
- **Success rate tracking** across multiple experiments

### Code Comparison & Analysis
![Code Diff](screenshots/Code%20evolution%20diff.png)

- **Side-by-side code comparison** between original and evolved versions
- **Detailed performance metrics** for each evolution checkpoint
- **Interactive code explorer** with syntax highlighting
- **Download evolved code** for integration into your project

## 🛠️ CLI Commands

Agent Evolve provides a comprehensive command-line interface:

### Core Commands

```bash
# Extract functions marked with @evolve decorator
agent-evolve extract [--path ./src] [--output-dir .agent_evolve]

# Generate training data from real usage
agent-evolve sample-training-data [--count 100]

# Generate custom evaluator for a function
agent-evolve generate-evaluator function_name

# Run evolution optimization
agent-evolve evolve function_name [--iterations 50] [--checkpoint 0]

# Launch interactive dashboard
agent-evolve dashboard [--port 8501]
```

### Advanced Commands

```bash
# Run the complete evolution pipeline
agent-evolve pipeline function_name

# Enable automatic tracing for data collection
agent-evolve trace [--include-modules my_app.*]

# Run evolution daemon for continuous optimization
agent-evolve daemon [--app-start-command "python app.py"]

# Analyze collected performance data
agent-evolve analyze [--db-path .agent_evolve/data/graph.db]
```

## 🎯 Key Features

### 🔄 Zero-Configuration Evolution
- **Automatic function detection** via `@evolve` decorator
- **Smart evaluator generation** using AI-powered analysis
- **Real-time performance tracking** without code changes
- **Seamless integration** with existing Python projects

### 📈 Advanced Metrics & Analytics
- **Multi-dimensional performance tracking** (accuracy, speed, quality)
- **Statistical significance testing** for improvements
- **A/B testing capabilities** between versions
- **Custom metric definitions** for domain-specific optimization

### 🤖 AI-Powered Optimization
- **OpenEvolve integration** for state-of-the-art evolutionary algorithms
- **Intelligent code generation** using large language models
- **Adaptive mutation strategies** based on performance feedback
- **Multi-objective optimization** balancing multiple criteria

### 🔧 Production-Ready Tools
- **Safe rollback mechanisms** to previous versions
- **Performance regression detection** 
- **Integration testing** for evolved functions
- **Deployment pipeline integration** with CI/CD systems

## 💡 Use Cases

### Function Optimization
```python
@evolve()
def calculate_pricing(customer_data, product_info):
    """Optimize pricing algorithm based on real customer behavior"""
    # Your initial implementation
    return base_price * factor
```

### Prompt Engineering
```python
@evolve()
CUSTOMER_SUPPORT_PROMPT = """
Help the customer with their inquiry about {product}.
Be helpful and professional.
"""
```

### Algorithm Enhancement
```python
@evolve()
def recommend_products(user_profile, inventory):
    """Evolve recommendation algorithms using real user interactions"""
    # Initial recommendation logic
    return recommendations
```

## 📋 Requirements

- **Python**: 3.8+ 
- **OpenAI API Key**: For AI-powered evaluator generation
- **Dependencies**: Automatically installed via pip
  - `streamlit` - Interactive dashboard
  - `plotly` - Data visualization
  - `openevolve` - Evolution engine
  - `pandas` - Data manipulation

## 🏗️ Architecture

Agent Evolve consists of several key components:

- **Decorator System**: Marks functions for evolution tracking
- **Extraction Engine**: Discovers and analyzes evolution targets
- **Evaluator Generator**: Creates custom evaluation functions using AI
- **Evolution Engine**: Runs optimization using OpenEvolve
- **Dashboard**: Streamlit-based interface for monitoring and control
- **Tracking System**: SQLite-based performance data collection

## 📝 Configuration

Agent Evolve works out-of-the-box but can be customized:

### Environment Variables
```bash
export OPENAI_API_KEY="your-api-key-here"
export AGENT_EVOLVE_DB_PATH=".agent_evolve/data/graph.db"
export AGENT_EVOLVE_OUTPUT_DIR=".agent_evolve"
```

### Project Structure
After running extraction, your project will have:
```
your_project/
├── .agent_evolve/
│   ├── data/
│   │   └── graph.db          # Performance tracking database
│   └── function_name/
│       ├── evaluator.py      # Custom evaluator
│       ├── evolve_target.py  # Evolution target code
│       ├── openevolve_config.yaml
│       └── checkpoints/      # Evolution progress saves
└── your_code.py             # Your original code with @evolve
```

## 🤝 Contributing

We welcome contributions! Agent Evolve is designed to be extensible:

1. **Custom Evaluators**: Add domain-specific evaluation logic
2. **New Evolution Strategies**: Integrate additional optimization algorithms  
3. **Dashboard Enhancements**: Improve visualization and user experience
4. **Integration Plugins**: Connect with popular ML frameworks

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

## 🆘 Support

- **Documentation**: Coming soon at [docs.agent-evolve.com]
- **Issues**: Report bugs and request features on GitHub
- **Community**: Join our Discord for discussions and support

---

**Start evolving your AI agents today!** 🚀

Simply add `@evolve()` to any function and let Agent Evolve handle the rest.