# Experimental One-Shot Optimization System

Transform your prompts from basic to expert-level using AI-powered meta-prompt optimization.

## Overview

The Prompt Optimization System automatically enhances your evaluation prompts using meta-prompt strategies. It takes basic prompts and transforms them into expert-level instructions that improve model performance.

### Before vs After Example

**Original Prompt (95 chars):**
```
Given patient information, predict the presence of heart disease. I can critically assess the provided trainee opinions.
```

**AI-Optimized Prompt (819 chars):**
```
As a cardiovascular disease specialist, I will predict the presence of heart disease based on provided patient information, utilizing my expertise in cardiology to critically evaluate the case. To ensure accuracy, I will follow a structured approach: (1) review patient demographics and medical history, (2) analyze symptoms and clinical presentation, (3) evaluate diagnostic test results, and (4) consider relevant risk factors. I will then assess the trainee opinions, identifying areas of agreement and potential discrepancies, and provide a comprehensive evaluation of the patient's condition, including a clear prediction of heart disease presence and supporting rationale, with a focus on thoroughness and precision.
```

**Result:** 8.6x improvement in prompt sophistication, expert role definition, systematic methodology.

---

## Quick Start

### Prerequisites
```bash
# Ensure you're in the langProBe directory
cd /path/to/langProBe

# Set up your OpenRouter API key
export OPENROUTER_API_KEY="your-api-key-here"
```

### Run Your First Optimization
```bash
# Quick test with HeartDisease benchmark
python -m experimental.one_shot_optimization.cli quick-test --benchmark HeartDisease
```

### Run Full Pipeline
```bash
# Optimize multiple benchmarks with multiple strategies
python -m experimental.one_shot_optimization.cli pipeline --config configs/pipeline_configs/default.yaml
```

### Check Results
```bash
# List available prompt versions
python -m experimental.one_shot_optimization.cli list-versions

# View performance rankings in the terminal output or:
ls meta-optimize-prompt/generated/latest/evaluation_results/
```

---

## Installation & Setup

**📋 Complete Setup Instructions**: See [SETUP.md](SETUP.md) for detailed installation guide including fresh laptop setup.

### Quick Setup
```bash
# 1. Install dependencies (if not already done for langProBe)
pip install openai pyyaml pathlib

# 2. Set up OpenRouter API access
export OPENROUTER_API_KEY="sk-or-v1-your-key-here"

# 3. Verify installation
python -m experimental.one_shot_optimization.cli list-strategies
```

### Configuration Files
The system uses YAML configuration files located in `configs/pipeline_configs/`:

```yaml
# Example: configs/pipeline_configs/default.yaml
benchmarks:
  - HeartDisease
  - hover
  - judgebench

optimization:
  strategies:
    - openai_v1_research_paper_generated_with_few_data
    - gemini2.5_research_paper_generated_with_few_data
    - self_written
  sample_size: 25
  temperature: 0.3
  max_concurrent: 24

output_dir: meta-optimize-prompt
```

---

## Core Commands

### List Available Strategies
```bash
python -m experimental.one_shot_optimization.cli list-strategies
```
Shows all 7 available meta-prompt strategies:
- `openai_v1_research_paper_generated_with_few_data`
- `openai_v2_research_paper_generated_with_few_data`
- `gemini2.5_research_paper_generated_with_few_data`
- `metaai_research_paper_generated_with_few_data`
- `llama_405b`
- `self_written`
- `self_written_2`

### Extract Prompts Only
```bash
# Extract default prompts from all benchmarks
python -m experimental.one_shot_optimization.cli extract

# Extract from specific benchmarks
python -m experimental.one_shot_optimization.cli extract --benchmarks HeartDisease hover
```

### Optimize Existing Prompts
```bash
# Optimize with specific strategies
python -m experimental.one_shot_optimization.cli optimize \
  --benchmarks HeartDisease \
  --strategies openai_v1_research_paper_generated_with_few_data self_written \
  --sample-size 10
```

### Full Pipeline (Recommended)
```bash
# Using configuration file (recommended)
python -m experimental.one_shot_optimization.cli pipeline --config configs/pipeline_configs/default.yaml

# Using command line options
python -m experimental.one_shot_optimization.cli pipeline \
  --benchmarks HeartDisease hover judgebench \
  --strategies openai_v1_research_paper_generated_with_few_data \
  --output-dir my-prompts
```

### Quick Testing
```bash
# Test with single benchmark (fast)
python -m experimental.one_shot_optimization.cli quick-test --benchmark HeartDisease

# Available test configurations
python -m experimental.one_shot_optimization.cli pipeline --config configs/pipeline_configs/quick_test.yaml
```

---

## Practical Workflows

### Workflow 1: New User Exploration
```bash
# 1. See what's available
python -m experimental.one_shot_optimization.cli list-strategies

# 2. Quick test to understand the system
python -m experimental.one_shot_optimization.cli quick-test --benchmark HeartDisease

# 3. Check what was generated
ls meta-optimize-prompt/generated/latest/HeartDisease/

# 4. Run full optimization
python -m experimental.one_shot_optimization.cli pipeline --config configs/pipeline_configs/default.yaml
```

### Workflow 2: Research & Experimentation
```bash
# 1. Extract prompts for analysis
python -m experimental.one_shot_optimization.cli extract --benchmarks HeartDisease hover judgebench

# 2. Test different strategies
python -m experimental.one_shot_optimization.cli optimize \
  --benchmarks HeartDisease \
  --strategies openai_v1_research_paper_generated_with_few_data gemini2.5_research_paper_generated_with_few_data

# 3. Compare results
python -m experimental.one_shot_optimization.cli list-versions
```

### Workflow 3: Production Optimization
```bash
# 1. Create custom configuration
cp configs/pipeline_configs/default.yaml configs/pipeline_configs/my_config.yaml
# Edit my_config.yaml with your benchmarks and strategies

# 2. Run full pipeline
python -m experimental.one_shot_optimization.cli pipeline --config configs/pipeline_configs/my_config.yaml

# 3. Analyze results
cat meta-optimize-prompt/generated/latest/evaluation_results/*/consolidated_results.csv
```

---

## Understanding Results

### Pipeline Stages
1. **Extraction**: Discovers and extracts default prompts from benchmarks
2. **Optimization**: Uses AI to enhance prompts with meta-prompt strategies  
3. **Saving**: Stores optimized prompts in organized directory structure
4. **Evaluation**: Tests all prompt variations and ranks performance

### Output Structure
```
meta-optimize-prompt/
├── generated/
│   ├── latest/                    # Symlink to most recent run
│   └── v2025_06_30_2103/         # Timestamped run directory
│       ├── HeartDisease/          # Per-benchmark prompts
│       │   ├── HeartDisease_extracted_default_prompt.json
│       │   ├── HeartDisease_meta_openai_v1_*.json
│       │   └── ...
│       ├── evaluation_results/    # Performance results
│       │   └── HeartDisease/
│       │       └── consolidated_results.csv
│       ├── optimization_metadata.json
│       └── evaluation_config.json
```

### Performance Rankings
The system automatically ranks prompt variations by performance:
```
Performance Ranking:
  1. meta_self_written_2                      |  32.0 |  2 samples
  2. extracted_default                        |  31.0 |  2 samples  
  3. meta_openai_v1_research_paper_*          |  30.0 |  2 samples
```

---

## Configuration Guide

### Pipeline Configuration
Create or modify files in `configs/pipeline_configs/`:

```yaml
# configs/pipeline_configs/my_optimization.yaml
benchmarks:
  - HeartDisease      # Medical diagnosis
  - hover             # Fact verification
  - judgebench        # Judge evaluation
  - gsm8k             # Math reasoning

optimization:
  strategies:
    - openai_v1_research_paper_generated_with_few_data   # Best overall
    - gemini2.5_research_paper_generated_with_few_data   # Strong analytical
    - self_written                                        # Balanced approach
  
  # AI optimization settings
  model: meta-llama/llama-3.3-70b-instruct
  base_url: https://openrouter.ai/api/v1
  temperature: 0.3      # Lower = more consistent
  sample_size: 25       # Evaluation samples per prompt
  max_concurrent: 12    # Parallel optimization processes

# Evaluation settings
evaluation_config:
  dataset_mode: test    # or 'full' for complete evaluation
  lm: openrouter/meta-llama/llama-3.3-70b-instruct
  num_threads: 24

# Output
output_dir: meta-optimize-prompt
```

### Strategy Selection Guide

| Strategy | Best For | Characteristics |
|----------|----------|-----------------|
| `openai_v1_research_paper_*` | General purpose | Systematic, expert role definition |
| `openai_v2_research_paper_*` | Analytical tasks | Evidence-based reasoning |
| `gemini2.5_research_paper_*` | Complex analysis | Multi-layered approach |
| `metaai_research_paper_*` | Reasoning tasks | Advanced logical consistency |
| `llama_405b` | Sophisticated tasks | Advanced methodology |
| `self_written` | Balanced approach | Clear, actionable instructions |
| `self_written_2` | Quality focus | Systematic, thorough |

---

## Troubleshooting

### Common Issues

**Configuration file not found:**
```bash
# Wrong:
python -m experimental.one_shot_optimization.cli pipeline --config configs/default.yaml

# Correct:
python -m experimental.one_shot_optimization.cli pipeline --config configs/pipeline_configs/default.yaml
```

**No prompts extracted:**
- Some benchmarks don't have extractable prompts
- This is normal - the system will create minimal configs for optimization

**API rate limiting errors:**
- Reduce `max_concurrent` in config (try 4-8)
- Increase delay between requests
- Check OpenRouter API limits

**Import errors:**
```bash
# Make sure you're running from langProBe root directory
cd /path/to/langProBe
python -m experimental.one_shot_optimization.cli --help
```

### Debug Commands
```bash
# Test CLI installation
python -m experimental.one_shot_optimization.cli --help

# Test API connectivity
python -m experimental.one_shot_optimization.cli list-strategies

# Minimal test
python -m experimental.one_shot_optimization.cli quick-test --benchmark HeartDisease

# Check available configurations
ls configs/pipeline_configs/
```

---

## Tips for Success

1. **Start Small**: Use `quick-test` before running full pipelines
2. **Monitor API Usage**: Watch OpenRouter usage to avoid rate limits
3. **Compare Strategies**: Different strategies work better for different tasks
4. **Iterate**: Use results to inform next optimization round
5. **Save Configs**: Create reusable configurations for your common workflows

---

## Getting Started

The Prompt Optimization System transforms basic prompts into expert-level instructions automatically. 

**🚀 First Time Setup**: See [SETUP.md](SETUP.md) for complete installation instructions.

**Quick Start:**
1. Run `python -m experimental.one_shot_optimization.cli quick-test --benchmark HeartDisease`
2. Explore the generated prompts in `meta-optimize-prompt/generated/latest/`
3. Run full pipeline with `configs/pipeline_configs/default.yaml`
4. Analyze results and iterate

**Two Usage Approaches:**
- **Strategy-Based**: Use pre-built meta-prompt strategies (see commands below)
- **Custom Prompts**: Direct system prompt control (see [SETUP.md](SETUP.md) for details)
