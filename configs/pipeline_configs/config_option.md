# langProBe Configuration Options

This document provides a comprehensive overview of the configuration options available in the langProBe framework.

## Dataset Modes

The `dataset_mode` parameter controls the size of datasets used for training, validation, and testing:

| Mode | Description | Size |
|------|-------------|------|
| `full` | Uses the complete dataset | No limit |
| `lite` | Uses a medium-sized subset | 500 samples |
| `tiny` | Uses a small subset | 200 samples |
| `test` | Uses a minimal subset for quick testing | 50 samples |

When using the `test` mode:
- The main dataset is trimmed to 60 samples
- The test set is trimmed to 50 samples
- This mode is primarily for debugging purposes

## Pipeline Configuration Options

Pipeline configurations are stored in YAML files in the `configs/pipeline_configs/` directory.

### Example Configuration

```yaml
benchmarks:
- HeartDisease
- hover
- judgebench

evaluation_config:
  dataset_mode: test        # Dataset size: full, lite, tiny, test
  lm: openrouter/meta-llama/llama-3.3-70b-instruct  # Language model to use
  num_threads: 24           # Number of parallel threads for evaluation

optimization:
  base_url: https://openrouter.ai/api/v1  # API base URL
  max_concurrent: 24        # Maximum concurrent optimization processes
  model: meta-llama/llama-3.3-70b-instruct  # Model for optimization
  sample_size: 25           # Number of samples per prompt evaluation
  strategies:               # Meta-prompt strategies to use
  - openai_v2_research_paper_generated_with_few_data
  - openai_v1_research_paper_generated_with_few_data
  - gemini2.5_research_paper_generated_with_few_data
  - metaai_research_paper_generated_with_few_data
  - self_written
  - self_written_2
  - llama_405b
  temperature: 0.3          # Temperature for optimization (lower = more consistent)

output_dir: meta-optimize-prompt  # Directory to store output files
```

## Available Benchmarks

### Agent Benchmarks
- AppWorld

### Non-Agent Benchmarks
- hover - Fact verification
- Iris - Classification
- IReRa - Information retrieval
- hotpotQA - Question answering
- MATH - Mathematical reasoning
- gsm8k - Grade school math
- RAGQAArenaTech - Retrieval-augmented QA
- MMLU - Massive multitask language understanding
- swebenchAnnotation - Software engineering
- scone - Scientific reasoning
- hotpotQA_conditional - Conditional QA
- HeartDisease - Medical diagnosis
- judgebench - Legal reasoning
- humaneval - Code generation
- MedQA - Medical question answering
- PubMedQA - Medical literature QA
- MedMCQA - Medical multiple choice QA

## Command Line Arguments

When running evaluations, the following command line arguments are available:

```bash
python -m langProBe.evaluation [options]
```

| Argument | Description | Default |
|----------|-------------|---------|
| `--dataset_mode` | Dataset mode (full, lite, tiny, test) | Default from BenchmarkMeta |
| `--lm` | Language model to use | openai/gpt-4o-mini |
| `--lm_api_base` | Language model API base URL | None |
| `--lm_api_key` | Language model API key | None |
| `--benchmark_set` | Benchmark set (full, nonagent, agent) | None |
| `--benchmark` | Specific benchmark to evaluate | None |
| `--file_path` | Path to save evaluation results | timestamp-based |
| `--num_threads` | Number of threads for evaluation | 16 |
| `--use_devset` | Use dev set instead of test set | False |
| `--missing_mode` | Only evaluate missing experiments | False |
| `--program_class` | Program class to evaluate (single, archon, all) | all |

## One-Shot Optimization CLI

For meta-prompt optimization, use the one-shot optimization CLI:

```bash
python -m experimental.one_shot_optimization.cli [command] [options]
```

### Commands

- `pipeline`: Run full optimization pipeline
- `optimize`: Run optimization only
- `extract`: Extract prompts from benchmarks
- `quick-test`: Run quick test with single benchmark
- `list-strategies`: List available meta-prompt strategies

### Pipeline Command Options

```bash
python -m experimental.one_shot_optimization.cli pipeline [options]
```

| Option | Description |
|--------|-------------|
| `--config` | Configuration file to use |
| `--benchmarks` | Specific benchmarks to process |
| `--strategies` | Meta-prompt strategies to use |
| `--output-dir` | Output directory for prompts |
| `--skip-extraction` | Skip prompt extraction |
| `--skip-optimization` | Skip meta-prompt optimization |
| `--skip-saving` | Skip saving prompts |
| `--skip-evaluation` | Skip running evaluation |

## Meta-Prompt Strategies

The framework provides several meta-prompt strategies for optimization:

- `openai_v1_research_paper_generated_with_few_data`
- `openai_v2_research_paper_generated_with_few_data`
- `gemini2.5_research_paper_generated_with_few_data`
- `metaai_research_paper_generated_with_few_data`
- `self_written`
- `self_written_2`
- `llama_405b`

Each strategy uses different approaches to optimize prompts for specific tasks. 