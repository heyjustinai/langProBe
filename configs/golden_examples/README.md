# Golden Examples for Prompt Optimization

This directory contains pre-extracted golden examples from key benchmarks to enhance prompt optimization through meta-prompting.

## Overview

Golden examples are high-quality input-output pairs from benchmark training sets that provide context to the meta-prompt optimization process. They help the optimization system understand:

- The structure and format of expected inputs
- The desired output format and style
- Examples of correct reasoning patterns
- Task-specific nuances and requirements

## Directory Structure

```
configs/golden_examples/
├── README.md                           # This file
├── HeartDisease_golden_examples.json   # Heart disease prediction examples
├── hover_golden_examples.json          # Fact verification examples
├── judgebench_golden_examples.json     # Response quality evaluation examples
└── extract_golden_examples.py          # Script to extract new examples
```

## JSON Format

Each golden examples file follows this structure:

```json
[
  {
    "inputs": {
      "field1": "value1",
      "field2": "value2",
      ...
    },
    "outputs": {
      "answer": "expected_output"
    }
  },
  ...
]
```

## Supported Benchmarks

### HeartDisease
- **Task**: Medical diagnosis prediction
- **Input**: Patient medical data (age, sex, chest pain type, etc.)
- **Output**: Heart disease presence (yes/no)
- **Examples**: 5 examples with diverse patient profiles

### hover
- **Task**: Fact verification using Wikipedia evidence
- **Input**: Claim and supporting evidence text
- **Output**: Verification result (SUPPORTS/REFUTES/NOT_ENOUGH_INFO)
- **Examples**: 5 examples covering different verification scenarios

### judgebench
- **Task**: Response quality evaluation and comparison
- **Input**: Question and two response options (A and B)
- **Output**: Quality comparison (A>B or B>A)
- **Examples**: 5 examples with varied response quality differences

## How It Works

1. **Loading**: During optimization, the system automatically loads golden examples for the benchmark being optimized
2. **Template Integration**: Examples are formatted and integrated into meta-prompt templates
3. **Context Enhancement**: Examples provide context to help the LLM understand the task requirements
4. **Few-Shot Learning**: Examples can be included as few-shot examples in the generated optimized prompts

## Usage

The golden examples system is automatically activated when:
- The benchmark name matches one of the supported benchmarks
- A corresponding `{benchmark}_golden_examples.json` file exists
- The optimization process is running

No manual intervention is required - the system handles loading and formatting automatically.

## Extracting New Examples

To extract golden examples for additional benchmarks:

1. Run the extraction script:
```bash
cd experimental/one_shot_optimization
python extract_golden_examples.py
```

2. Or use the programmatic interface:
```python
from experimental.one_shot_optimization.optimizer import GoldenExampleExtractor

extractor = GoldenExampleExtractor()
examples = extractor.extract_and_save_golden_examples(num_examples=5)
```

## Adding New Benchmarks

To add support for a new benchmark:

1. Add the benchmark name to `key_benchmarks` in `GoldenExampleExtractor`
2. Ensure the benchmark follows the standard DSPy Example format
3. Run the extraction script to generate the JSON file
4. Verify the extracted examples are high-quality and representative

## Quality Guidelines

Golden examples should be:
- **Representative**: Cover diverse scenarios within the benchmark
- **High-Quality**: Demonstrate correct reasoning and formatting
- **Concise**: Clear and focused, not overly verbose
- **Diverse**: Include edge cases and different difficulty levels
- **Validated**: Manually reviewed for accuracy

## Performance Impact

- **Zero Runtime Overhead**: Examples are pre-extracted and cached
- **Fast Loading**: JSON files load instantly compared to dataset initialization
- **Minimal Storage**: Each file is typically < 10KB
- **Graceful Degradation**: System works normally if files are missing

## Maintenance

- Update examples when benchmark datasets change
- Review examples periodically for quality and relevance
- Add new benchmarks as the system expands
- Monitor optimization performance improvements with golden examples enabled 