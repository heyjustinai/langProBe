# One-Shot Optimization Setup Guide

Complete setup instructions for the LangProBe One-Shot Optimization system.

## Table of Contents
- [Fresh Laptop Setup](#fresh-laptop-setup)
- [Quick Start](#quick-start)
- [Installation Details](#installation-details)
- [Configuration](#configuration)
- [Verification](#verification)
- [Usage Examples](#usage-examples)
- [Troubleshooting](#troubleshooting)

---

## Fresh Laptop Setup

### Prerequisites
- **Operating System**: macOS, Linux, or Windows
- **Python**: 3.10 or higher
- **Git**: For cloning the repository

### 1. Install Python Environment Manager

**Option A: Conda (Recommended)**
```bash
# Download and install Miniconda
curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh
bash Miniconda3-latest-MacOSX-x86_64.sh

# Restart terminal or source your shell config
source ~/.bashrc  # or ~/.zshrc on macOS
```

**Option B: Python venv**
```bash
# Ensure Python 3.10+ is installed
python3 --version  # Should show 3.10+
```

### 2. Clone and Navigate to Project

```bash
# Clone the repository (replace with actual repo URL)
git clone <repository-url> langProBe
cd langProBe

# Verify project structure
ls -la  # Should see langProBe/, experimental/, requirements.txt
```

### 3. Create Virtual Environment

**Using Conda:**
```bash
conda create -n langprobe python=3.10 -y
conda activate langprobe
```

**Using venv:**
```bash
python3 -m venv langprobe-env
source langprobe-env/bin/activate  # Windows: langprobe-env\Scripts\activate
```

### 4. Install Dependencies

```bash
# Install base dependencies
pip install -r requirements.txt

# Install additional one-shot optimization dependencies
pip install openai python-dotenv pyyaml
```

**Key Dependencies Installed:**
- `dspy>=2.6` - Core DSPy framework
- `openai` - OpenAI API client for meta-prompt optimization
- `python-dotenv` - Environment variable management
- `pyyaml` - YAML configuration support
- `torch`, `numpy`, `requests` - Supporting libraries

### 5. Set Up API Key

**Get OpenRouter API Key:**
1. Visit [OpenRouter.ai](https://openrouter.ai/)
2. Create account and generate API key (starts with `sk-or-v1-`)

**Configure API Key:**

**Option A: Environment File (Recommended)**
```bash
# Create .env file in project root
echo "OPENROUTER_API_KEY=sk-or-v1-your-actual-key-here" > .env
```

**Option B: Shell Environment**
```bash
# For current session
export OPENROUTER_API_KEY="sk-or-v1-your-actual-key-here"

# For permanent setup
echo 'export OPENROUTER_API_KEY="sk-or-v1-your-actual-key-here"' >> ~/.zshrc
source ~/.zshrc
```

---

## Quick Start

### Verify Installation
```bash
# Test 1: Check CLI works
python -m experimental.one_shot_optimization.cli --help

# Test 2: List available strategies
python -m experimental.one_shot_optimization.cli list-strategies

# Test 3: Quick optimization test
python -m experimental.one_shot_optimization.cli quick-test --benchmark HeartDisease
```

### Run Your First Optimization
```bash
# Simple pipeline run
python -m experimental.one_shot_optimization.cli pipeline

# Or with custom configuration
python -m experimental.one_shot_optimization.cli pipeline --config configs/pipeline_configs/default.yaml
```

---

## Installation Details

### Automated Setup Script
The project includes an automated setup script:

```bash
# Run automated setup
bash setup.sh
conda activate langprobe
```

### Manual Dependency Installation
If you need to install dependencies manually:

```bash
# Core LangProBe dependencies
pip install dspy>=2.6 requests shortuuid seaborn langchain
pip install black sentence-transformers torch numpy
pip install huggingface-hub langchain_community

# One-shot optimization specific
pip install openai python-dotenv pyyaml pathlib
```

### Directory Structure Verification
Your project should look like this:

```
langProBe/
├── experimental/
│   └── one_shot_optimization/
│       ├── cli.py              # Command-line interface
│       ├── optimizer.py        # Meta-prompt optimization
│       ├── orchestrator.py     # Pipeline orchestration
│       ├── extractor.py        # Prompt extraction
│       ├── config.py          # Configuration management
│       └── ...
├── langProBe/                 # Core benchmarking framework
├── configs/
│   ├── pipeline_configs/      # Pipeline configurations
│   └── custom_prompts/        # Custom prompt configs
├── requirements.txt
├── setup.sh
├── .env                       # Your API keys (create this)
└── README.md
```

---

## Configuration

### Environment Variables
Required environment variables:

```bash
# Required: OpenRouter API key for meta-prompt optimization
OPENROUTER_API_KEY=sk-or-v1-your-key-here

# Optional: Custom logging configuration
LANGPROBE_LOG_LEVEL=INFO
```

### Pipeline Configuration
Create or modify files in `configs/pipeline_configs/`:

```yaml
# Example: configs/pipeline_configs/my_config.yaml
benchmarks:
  - HeartDisease      # Medical diagnosis
  - hover             # Fact verification  
  - judgebench        # Judge evaluation

optimization:
  strategies:
    - openai_v1_research_paper_generated_with_few_data
    - gemini2.5_research_paper_generated_with_few_data
    - self_written
  
  # AI optimization settings
  model: meta-llama/llama-3.3-70b-instruct
  base_url: https://openrouter.ai/api/v1
  temperature: 0.3
  sample_size: 25
  max_concurrent: 12

# Evaluation settings
evaluation_config:
  dataset_mode: test
  lm: openrouter/meta-llama/llama-3.3-70b-instruct
  num_threads: 24

output_dir: meta-optimize-prompt
```

### Custom Prompts Configuration
For direct prompt control, create files in `configs/custom_prompts/`:

```yaml
# Example: configs/custom_prompts/medical_expert.yaml
name: "Medical Expert Comparison"
description: "Compare different medical expertise approaches"

custom_prompts:
  expert_cardiologist:
    name: "Expert Cardiologist"
    system_prompt: |
      You are a board-certified cardiologist with 20 years of experience.
      Analyze patient data systematically:
      1. Review demographics and risk factors
      2. Evaluate symptoms and clinical signs  
      3. Consider diagnostic test results
      4. Apply evidence-based medicine principles
      Provide clear diagnosis with confidence level.

  general_practitioner:
    name: "General Practitioner"
    system_prompt: |
      You are a general practitioner with broad medical knowledge.
      Review the patient information and make an assessment.
      Focus on key indicators and provide clear reasoning.

benchmarks:
  - HeartDisease

evaluation:
  sample_size: 15
  max_concurrent: 3
```

---

## Verification

### Test Commands

**1. Basic CLI Test**
```bash
python -m experimental.one_shot_optimization.cli --help
```
*Expected: Help message with available commands*

**2. Strategy List Test**
```bash
python -m experimental.one_shot_optimization.cli list-strategies
```
*Expected: List of 7 optimization strategies*

**3. API Connection Test**
```bash
python -m experimental.one_shot_optimization.cli quick-test --benchmark HeartDisease
```
*Expected: Successful optimization with before/after prompts*

**4. Full Pipeline Test**
```bash
python -m experimental.one_shot_optimization.cli pipeline --config configs/pipeline_configs/default.yaml
```
*Expected: Complete pipeline execution with results*

### Expected Output Structure
After running the pipeline, you should see:

```
meta-optimize-prompt/
├── generated/
│   ├── latest/                    # Symlink to most recent run
│   └── v2025_01_XX_XXXX/         # Timestamped run
│       ├── HeartDisease/          # Per-benchmark results
│       ├── evaluation_results/    # Performance comparisons
│       └── optimization_metadata.json
```

---

## Usage Examples

### Two Main Approaches

**1. Strategy-Based Optimization (Advanced)**
Uses pre-built meta-prompt strategies:

```bash
# Full pipeline with multiple strategies
python -m experimental.one_shot_optimization.cli pipeline --config configs/pipeline_configs/default.yaml

# Custom strategy selection
python -m experimental.one_shot_optimization.cli optimize \
  --benchmarks HeartDisease hover \
  --strategies openai_v1_research_paper_generated_with_few_data self_written
```

**2. Custom Prompt Optimization (Simplified)**
Direct control over system prompts:

```bash
# Create example configuration
python -m experimental.one_shot_optimization.cli create-example-config

# Run with custom prompts
python -m experimental.one_shot_optimization.cli run --config configs/custom_prompts/example.yaml

# Quick test without evaluation
python -m experimental.one_shot_optimization.cli run --config configs/custom_prompts/example.yaml --quick-test
```

### Common Workflows

**Research & Experimentation:**
```bash
# 1. Extract existing prompts
python -m experimental.one_shot_optimization.cli extract --benchmarks HeartDisease

# 2. Test different strategies
python -m experimental.one_shot_optimization.cli optimize \
  --benchmarks HeartDisease \
  --strategies openai_v1_research_paper_generated_with_few_data gemini2.5_research_paper_generated_with_few_data

# 3. Compare results
python -m experimental.one_shot_optimization.cli list-versions
```

**Production Optimization:**
```bash
# 1. Create custom config
cp configs/pipeline_configs/default.yaml configs/pipeline_configs/production.yaml
# Edit production.yaml with your settings

# 2. Run full pipeline
python -m experimental.one_shot_optimization.cli pipeline --config configs/pipeline_configs/production.yaml

# 3. View results
ls meta-optimize-prompt/generated/latest/evaluation_results/
```

---

## Troubleshooting

### Common Issues

**1. "Module not found" error**
```bash
# Solution: Ensure you're in the correct directory
pwd  # Should end with /langProBe
ls   # Should show experimental/ and langProBe/ directories
```

**2. "API key not found" error**
```bash
# Solution: Check environment variable
echo $OPENROUTER_API_KEY  # Should show your key
cat .env                  # Should contain OPENROUTER_API_KEY=...

# Re-set if missing
export OPENROUTER_API_KEY="sk-or-v1-your-key-here"
```

**3. "Configuration file not found"**
```bash
# Wrong path
python -m experimental.one_shot_optimization.cli pipeline --config configs/default.yaml

# Correct path  
python -m experimental.one_shot_optimization.cli pipeline --config configs/pipeline_configs/default.yaml
```

**4. Import errors**
```bash
# Solution: Reinstall dependencies
pip install -r requirements.txt
pip install openai python-dotenv pyyaml
```

**5. Rate limiting errors**
```bash
# Solution: Reduce concurrency in config
# Edit your config file:
optimization:
  max_concurrent: 4  # Reduce from 12 to 4
```

### Debug Commands

```bash
# Test installation
python -c "import experimental.one_shot_optimization; print('✅ Import successful')"

# Test API connectivity
python -c "import os; print('✅ API Key found' if os.getenv('OPENROUTER_API_KEY') else '❌ No API key')"

# List available configurations
ls configs/pipeline_configs/
ls configs/custom_prompts/

# Check generated results
ls meta-optimize-prompt/generated/latest/
```

### Performance Tips

1. **Start Small**: Use `quick-test` before full pipelines
2. **Monitor API Usage**: Watch OpenRouter dashboard for rate limits
3. **Optimize Concurrency**: Adjust `max_concurrent` based on API limits
4. **Use Appropriate Sample Sizes**: Start with 10-25 samples for testing

---

## Next Steps

1. **Verify Setup**: Run the verification tests above
2. **Try Quick Test**: `python -m experimental.one_shot_optimization.cli quick-test --benchmark HeartDisease`
3. **Explore Results**: Check `meta-optimize-prompt/generated/latest/`
4. **Run Full Pipeline**: Use `configs/pipeline_configs/default.yaml`
5. **Create Custom Configs**: Adapt configurations for your use case

For detailed usage information, see the main README.md file.

---

## Support

- **Documentation**: See `README.md` for detailed usage
- **Logging Guide**: See `LOGGING_GUIDE.md` for debugging
- **Configuration Examples**: Check `configs/` directory
- **Issues**: Check existing documentation or create GitHub issue
