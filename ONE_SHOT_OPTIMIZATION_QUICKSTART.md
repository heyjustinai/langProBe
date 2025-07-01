# One-Shot Optimization Quick Start

Automatically enhance langProBe evaluation prompts using experimental meta-prompt optimization.

## Quick Start

```bash
# 1. Set up API key
export OPENROUTER_API_KEY="your-key-here"

# 2. Quick test
python -m experimental.one_shot_optimization.cli quick-test --benchmark HeartDisease

# 3. Full optimization pipeline
python -m experimental.one_shot_optimization.cli pipeline --config configs/pipeline_configs/default.yaml
```

## Results

**Before:** Basic 95-character prompt
```
Given patient information, predict the presence of heart disease.
```

**After:** Expert 819-character optimized prompt
```
As a cardiovascular disease specialist, I will predict the presence of heart disease based on provided patient information, utilizing my expertise in cardiology to critically evaluate the case. To ensure accuracy, I will follow a structured approach: (1) review patient demographics...
```

**Improvement:** 8.6x length increase with systematic methodology and expert role definition

## Core Commands

```bash
# List all 7 optimization strategies
python -m experimental.one_shot_optimization.cli list-strategies

# Extract prompts from benchmarks
python -m experimental.one_shot_optimization.cli extract --benchmarks HeartDisease hover

# Run full pipeline with specific benchmarks
python -m experimental.one_shot_optimization.cli pipeline \
  --benchmarks HeartDisease hover judgebench \
  --strategies openai_v1_research_paper_generated_with_few_data

# Check generated results
python -m experimental.one_shot_optimization.cli list-versions
ls meta-optimize-prompt/generated/latest/
```

## Configuration

```yaml
# configs/pipeline_configs/default.yaml
benchmarks: [HeartDisease, hover, judgebench]
optimization:
  strategies:
    - openai_v1_research_paper_generated_with_few_data
    - gemini2.5_research_paper_generated_with_few_data
    - self_written
  sample_size: 25
  max_concurrent: 12
```

## Troubleshooting

**Configuration not found:**
```bash
# Wrong: configs/default.yaml
# Correct: configs/pipeline_configs/default.yaml
```

**Import errors:**
```bash
# Make sure you're in langProBe root directory
cd /path/to/langProBe
python -m experimental.one_shot_optimization.cli --help
```

## Documentation

- **Complete Guide**: `experimental/one_shot_optimization/README.md`
- **Pipeline Configs**: `configs/pipeline_configs/`

## Next Steps

1. **Quick Test**: `python -m experimental.one_shot_optimization.cli quick-test --benchmark HeartDisease`
2. **Explore Results**: `ls meta-optimize-prompt/generated/latest/HeartDisease/`
3. **Full Pipeline**: `python -m experimental.one_shot_optimization.cli pipeline --config configs/pipeline_configs/default.yaml`
4. **Scale Up**: Add more benchmarks and strategies to your configs

**Transform your prompts today!** 🚀
