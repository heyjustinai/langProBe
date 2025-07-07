"""
Prompt extraction from benchmark definitions.
"""

import importlib
import inspect
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import ast
import re
import yaml
import json

import sys
from pathlib import Path

# Add langProBe to path for benchmark imports
sys.path.append(str(Path(__file__).parent.parent.parent / "langProBe"))

try:
    from register_benchmark import register_all_benchmarks
except ImportError:
    # Fallback if register_benchmark is not available
    def register_all_benchmarks():
        pass
from .config import BenchmarkConfig, PromptConfig
from .benchmark_adapter import AdapterRegistry


class BenchmarkPromptExtractor:
    """Extracts default prompts from benchmark definitions."""
    
    def __init__(self, langprobe_dir: str = "langProBe", 
                 config_path: str = "configs/benchmark_configs/prompt_extraction_rules.yaml"):
        self.langprobe_dir = Path(langprobe_dir)
        self.discovered_benchmarks = {}
        self.config_path = Path(config_path)
        self.extraction_rules = self._load_extraction_rules()
        
    def _load_extraction_rules(self) -> Dict[str, Any]:
        """Load extraction rules from configuration file."""
        if self.config_path.exists():
            with open(self.config_path, 'r') as f:
                return yaml.safe_load(f) or {}
        return {}
    
    def discover_benchmarks(self) -> List[str]:
        """Discover all available benchmarks by examining the directory structure."""
        benchmark_dirs = []
        
        for item in self.langprobe_dir.iterdir():
            if (item.is_dir() and 
                not item.name.startswith('_') and 
                not item.name in ['__pycache__', 'one_shot_optimization']):
                
                # Check if it has the expected benchmark files
                program_file = item / f"{item.name}_program.py"
                if program_file.exists():
                    benchmark_dirs.append(item.name)
                    
        print(f"Discovered {len(benchmark_dirs)} benchmarks: {benchmark_dirs}")
        return benchmark_dirs
    
    def extract_prompt_from_program_file(self, benchmark_dir: Path) -> Optional[Dict[str, Any]]:
        """Extract prompt definitions from a benchmark program file."""
        benchmark_name = benchmark_dir.name
        
        # 1. Check config-based extraction first (if available)
        if self.extraction_rules:
            config_result = self._extract_via_config(benchmark_name, benchmark_dir)
            if config_result:
                return config_result
        
        # 2. Check if we have an adapter
        if AdapterRegistry.has_adapter(benchmark_name):
            adapter = AdapterRegistry.get_adapter(benchmark_name, benchmark_dir)
            if adapter:
                return self._extract_via_adapter(benchmark_name, adapter)
        
        # 3. Quick hard-coded extraction for hover and IReRa (backward compatibility)
        if benchmark_name == 'hover':
            return self._extract_hover_prompts()
        elif benchmark_name == 'IReRa':
            return self._extract_irera_prompts()
        
        # 4. Try enhanced extraction for unhandled patterns
        enhanced_result = self._try_enhanced_extraction(benchmark_dir)
        if enhanced_result:
            return enhanced_result
        
        # 5. Fall back to standard extraction
        return self._standard_extraction(benchmark_dir)
    
    def _extract_via_config(self, benchmark_name: str, benchmark_dir: Path) -> Optional[Dict[str, Any]]:
        """Extract prompts using configuration rules."""
        rules = self.extraction_rules.get('extraction_rules', {}).get(benchmark_name)
        if not rules:
            return None
            
        prompt_info = {
            'benchmark_name': benchmark_name,
            'signatures': [],
            'instructions': [],
            'task_descriptions': [rules.get('description', '')]
        }
        
        # Handle different extraction types
        if rules['type'] == 'string_signatures':
            # Extract string-based signatures
            for sig in rules.get('signatures', []):
                sig_info = {
                    'class_name': sig['name'],
                    'docstring': sig['instruction'],
                    'instructions': sig['instruction'],
                    'input_fields': sig['input_fields'],
                    'output_fields': sig['output_fields']
                }
                # Add signature_key if present
                if 'signature_key' in sig:
                    sig_info['signature_key'] = sig['signature_key']
                prompt_info['signatures'].append(sig_info)
                
            if rules.get('default_prompt'):
                prompt_info['instructions'].append(rules['default_prompt'])
                
        elif rules['type'] == 'utils_signatures':
            # Extract from utils file based on dataset
            dataset = rules.get('default_dataset', '')
            if dataset in rules.get('datasets', {}):
                dataset_info = rules['datasets'][dataset]
                prompt_info['signatures'].append({
                    'class_name': f'{benchmark_name}_{dataset}',
                    'docstring': dataset_info['instruction'],
                    'instructions': dataset_info['instruction'],
                    'input_fields': dataset_info['input_fields'],
                    'output_fields': dataset_info['output_fields']
                })
                prompt_info['instructions'].append(dataset_info['instruction'])
        
        return prompt_info if prompt_info['signatures'] or prompt_info['instructions'] else None
    
    def _extract_via_adapter(self, benchmark_name: str, adapter) -> Dict[str, Any]:
        """Extract prompts using benchmark adapter."""
        signatures = adapter.get_signatures()
        
        prompt_info = {
            'benchmark_name': benchmark_name,
            'signatures': [],
            'instructions': [adapter.get_default_prompt()],
            'task_descriptions': [adapter.get_task_description()]
        }
        
        # Convert adapter signatures to standard format
        for sig in signatures:
            prompt_info['signatures'].append({
                'class_name': sig.get('name', 'UnknownSignature'),
                'docstring': sig.get('instruction', ''),
                'instructions': sig.get('instruction', ''),
                'input_fields': sig.get('input_fields', []),
                'output_fields': sig.get('output_fields', [])
            })
        
        return prompt_info
    
    def _try_enhanced_extraction(self, benchmark_dir: Path) -> Optional[Dict[str, Any]]:
        """Try enhanced extraction methods for non-standard patterns."""
        program_file = benchmark_dir / f"{benchmark_dir.name}_program.py"
        utils_file = benchmark_dir / f"{benchmark_dir.name}_utils.py"
        
        prompt_info = {
            'benchmark_name': benchmark_dir.name,
            'signatures': [],
            'instructions': [],
            'task_descriptions': []
        }
        
        # Try to extract string-based signatures from program file
        if program_file.exists():
            with open(program_file, 'r') as f:
                content = f.read()
            
            # Look for string signatures in dspy.Predict/ChainOfThought
            pattern = r'dspy\.(Predict|ChainOfThought)\s*\(\s*["\']([^"\']+)["\']'
            matches = re.findall(pattern, content)
            
            for method, sig_string in matches:
                if '->' in sig_string:
                    inputs, output = sig_string.split('->')
                    input_fields = [i.strip() for i in inputs.split(',')]
                    
                    prompt_info['signatures'].append({
                        'class_name': f'StringSignature_{len(prompt_info["signatures"])}',
                        'docstring': f'Process {sig_string}',
                        'instructions': f'Process {sig_string}',
                        'input_fields': input_fields,
                        'output_fields': [output.strip()]
                    })
        
        # Try to extract from utils file
        if utils_file.exists() and not prompt_info['signatures']:
            with open(utils_file, 'r') as f:
                content = f.read()
            
            # Look for Signature classes
            tree = ast.parse(content)
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    for base in node.bases:
                        if (isinstance(base, ast.Attribute) and 
                            base.attr == 'Signature'):
                            
                            sig_info = self._extract_signature_info(node)
                            if sig_info:
                                prompt_info['signatures'].append(sig_info)
        
        return prompt_info if prompt_info['signatures'] else None
    
    def _standard_extraction(self, benchmark_dir: Path) -> Optional[Dict[str, Any]]:
        """Standard extraction for benchmarks following normal patterns."""
        program_file = benchmark_dir / f"{benchmark_dir.name}_program.py"
        
        if not program_file.exists():
            return None
            
        try:
            with open(program_file, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Parse the AST to find class definitions and signatures
            tree = ast.parse(content)
            
            prompt_info = {
                'benchmark_name': benchmark_dir.name,
                'signatures': [],
                'instructions': [],
                'task_descriptions': []
            }
            
            # Look for dspy.Signature definitions
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    # Check if it's a signature class
                    for base in node.bases:
                        if (isinstance(base, ast.Attribute) and 
                            isinstance(base.value, ast.Name) and
                            base.value.id == 'dspy' and 
                            base.attr == 'Signature'):
                            
                            signature_info = self._extract_signature_info(node)
                            if signature_info:
                                prompt_info['signatures'].append(signature_info)
            
            # Look for instructions in string literals and comments
            instructions = self._extract_instructions_from_content(content)
            prompt_info['instructions'].extend(instructions)
            
            # Try to infer task description from docstrings and comments
            task_descriptions = self._extract_task_descriptions(content, benchmark_dir.name)
            prompt_info['task_descriptions'].extend(task_descriptions)
            
            return prompt_info if prompt_info['signatures'] or prompt_info['instructions'] else None
            
        except Exception as e:
            print(f"Error parsing {program_file}: {e}")
            return None
    
    def _extract_hover_prompts(self) -> Dict[str, Any]:
        """Extract prompts for hover benchmark with string-based signatures."""
        return {
            'benchmark_name': 'hover',
            'signatures': [
                {
                    'class_name': 'SummarizePassages',
                    'docstring': 'Summarize passages that are relevant to the given claim.',
                    'instructions': 'Summarize passages that are relevant to the given claim.',
                    'input_fields': ['claim', 'passages'],
                    'output_fields': ['summary'],
                    'signature_key': 'summarize1'
                },
                {
                    'class_name': 'SummarizeWithContext',
                    'docstring': 'Summarize passages with additional context about the claim.',
                    'instructions': 'Summarize passages with additional context about the claim.',
                    'input_fields': ['claim', 'context', 'passages'],
                    'output_fields': ['summary'],
                    'signature_key': 'summarize2'
                },
                {
                    'class_name': 'GenerateQueryHop2',
                    'docstring': 'Generate a search query based on the claim and initial summary.',
                    'instructions': 'Generate a search query based on the claim and initial summary.',
                    'input_fields': ['claim', 'summary_1'],
                    'output_fields': ['query'],
                    'signature_key': 'query_hop2'
                },
                {
                    'class_name': 'GenerateQueryHop3',
                    'docstring': 'Generate a refined search query based on claim and multiple summaries.',
                    'instructions': 'Generate a refined search query based on claim and multiple summaries.',
                    'input_fields': ['claim', 'summary_1', 'summary_2'],
                    'output_fields': ['query'],
                    'signature_key': 'query_hop3'
                }
            ],
            'instructions': [
                'Summarize passages that are relevant to the given claim.',
                'Generate search queries based on claims and summaries for multi-hop fact verification.'
            ],
            'task_descriptions': ['Multi-hop fact verification using Wikipedia evidence']
        }
    
    def _extract_irera_prompts(self) -> Dict[str, Any]:
        """Extract prompts for IReRa benchmark with utils-based signatures."""
        # Default to ESCO dataset prompts
        return {
            'benchmark_name': 'IReRa',
            'signatures': [
                {
                    'class_name': 'InferSignatureESCO',
                    'docstring': 'Given a snippet from a job vacancy, identify all the ESCO job skills mentioned. Always return skills.',
                    'instructions': 'Given a snippet from a job vacancy, identify all the ESCO job skills mentioned. Always return skills.',
                    'input_fields': ['text'],
                    'output_fields': ['output']
                }
            ],
            'instructions': [
                'Given a snippet from a job vacancy, identify all the ESCO job skills mentioned. Always return skills.',
                'Given a snippet from a medical article, identify the adverse drug reactions affecting the patient. Always return reactions.'
            ],
            'task_descriptions': ['Information extraction from job vacancies and medical texts']
        }
    
    def _extract_signature_info(self, class_node: ast.ClassDef) -> Optional[Dict[str, str]]:
        """Extract information from a dspy.Signature class definition."""
        signature_info = {
            'class_name': class_node.name,
            'docstring': None,
            'instructions': None,
            'input_fields': [],
            'output_fields': []
        }
        
        # Get docstring
        if (class_node.body and 
            isinstance(class_node.body[0], ast.Expr) and
            isinstance(class_node.body[0].value, ast.Constant)):
            signature_info['docstring'] = class_node.body[0].value.value
        
        # Look for assignments in the class body
        for node in class_node.body:
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        if target.id == '__doc__' and isinstance(node.value, ast.Constant):
                            signature_info['instructions'] = node.value.value
                        # Look for field definitions (simplified)
                        elif 'input' in target.id.lower():
                            signature_info['input_fields'].append(target.id)
                        elif 'output' in target.id.lower():
                            signature_info['output_fields'].append(target.id)
        
        return signature_info
    
    def _extract_instructions_from_content(self, content: str) -> List[str]:
        """Extract potential instruction strings from file content."""
        instructions = []
        
        # Look for multi-line strings that might be instructions
        string_pattern = r'"""([^"]*(?:"[^"]*)*?)"""'
        multiline_strings = re.findall(string_pattern, content, re.DOTALL)
        
        for string in multiline_strings:
            string = string.strip()
            # Filter for instruction-like content (contains certain keywords)
            if any(keyword in string.lower() for keyword in 
                   ['analyze', 'predict', 'classify', 'determine', 'evaluate', 'answer']):
                if len(string) > 50:  # Reasonable length filter
                    instructions.append(string)
        
        return instructions
    
    def _extract_task_descriptions(self, content: str, benchmark_name: str) -> List[str]:
        """Extract task descriptions from comments and docstrings."""
        descriptions = []
        
        # Look for module-level docstring
        try:
            tree = ast.parse(content)
            if (tree.body and 
                isinstance(tree.body[0], ast.Expr) and
                isinstance(tree.body[0].value, ast.Constant)):
                descriptions.append(tree.body[0].value.value.strip())
        except:
            pass
        
        # Infer from benchmark name
        name_description = self._benchmark_name_to_description(benchmark_name)
        if name_description:
            descriptions.append(name_description)
            
        return descriptions
    
    def _benchmark_name_to_description(self, name: str) -> str:
        """Convert benchmark name to a descriptive task description."""
        name_mappings = {
            'HeartDisease': 'Heart disease prediction using medical data',
            'hover': 'Fact verification using Wikipedia evidence',
            'judgebench': 'Response quality evaluation and comparison',
            'MMLU': 'Multiple choice questions across academic subjects',
            'humaneval': 'Python code generation from natural language descriptions',
            'gsm8k': 'Mathematical reasoning and problem solving',
            'MATH': 'Advanced mathematical problem solving',
            'AlfWorld': 'Interactive household task completion',
            'AppWorld': 'Application-based task execution',
            'MedQA': 'Medical question answering',
            'MedMCQA': 'Medical multiple choice question answering',
            'PubMedQA': 'Biomedical literature question answering',
            'scone': 'Sequential context understanding and reasoning',
            'hotpotQA': 'Multi-hop question answering with reasoning',
            'IReRa': 'Information retrieval and reasoning tasks',
        }
        
        return name_mappings.get(name, f"{name} benchmark tasks")
    
    def extract_all_benchmark_prompts(self) -> Dict[str, BenchmarkConfig]:
        """Extract prompts from all discovered benchmarks."""
        benchmarks = self.discover_benchmarks()
        extracted_configs = {}
        
        for benchmark_name in benchmarks:
            print(f"Extracting prompts from {benchmark_name}...")
            
            benchmark_dir = self.langprobe_dir / benchmark_name
            prompt_info = self.extract_prompt_from_program_file(benchmark_dir)
            
            if prompt_info:
                config = self._create_benchmark_config(prompt_info)
                if config:
                    extracted_configs[benchmark_name] = config
                    print(f"  ✅ Extracted {len(config.prompt_variations)} prompt variations")
                else:
                    print(f"  Could not create valid config")
            else:
                print(f"  ❌ No prompts found")
                
        return extracted_configs
    
    def _create_benchmark_config(self, prompt_info: Dict[str, Any]) -> Optional[BenchmarkConfig]:
        """Create a BenchmarkConfig from extracted prompt information."""
        benchmark_name = prompt_info['benchmark_name']
        
        # Check if this is a multi-signature benchmark (hover)
        signatures = prompt_info.get('signatures', [])
        is_multi_signature = (benchmark_name == 'hover' and 
                            len([s for s in signatures if 'signature_key' in s]) > 1)
        
        if is_multi_signature:
            # Handle multi-signature benchmark
            print(f"  🔄 Detected multi-signature benchmark: {benchmark_name}")
            
            # Create a config that represents all signatures
            config = BenchmarkConfig(
                name=benchmark_name,
                base_prompt="Multi-signature benchmark - see individual signatures",
                signature="Multiple signatures",
                task_description=self._select_best_task_description(prompt_info)
            )
            
            # Store all signature information
            config.multi_signatures = signatures
            
            # Create a single variation that contains all signatures
            all_prompts = {
                sig['signature_key']: sig.get('instructions', sig.get('docstring', ''))
                for sig in signatures if 'signature_key' in sig
            }
            
            default_prompt = PromptConfig(
                name="extracted_default",
                description=f"Default prompts extracted from {benchmark_name} benchmark",
                instructions=json.dumps(all_prompts),  # Store as JSON
                task_description=config.task_description,
                signature=config.signature,
                multi_signature=True,
                signature_prompts=all_prompts
            )
            
            config.add_variation(default_prompt)
            return config
        
        else:
            # Original single-signature logic
            # Choose the best instruction from available options
            best_instruction = self._select_best_instruction(prompt_info)
            if not best_instruction:
                # Create a default instruction based on benchmark name
                best_instruction = f"You are an expert assistant. {self._benchmark_name_to_description(benchmark_name)}. Provide accurate and helpful responses."
            
            # Choose the best signature
            signature = self._select_best_signature(prompt_info)
            
            # Choose the best task description
            task_description = self._select_best_task_description(prompt_info)
            
            config = BenchmarkConfig(
                name=benchmark_name,
                base_prompt=best_instruction,
                signature=signature,
                task_description=task_description
            )
            
            # Create a default prompt variation
            default_prompt = PromptConfig(
                name="extracted_default",
                description=f"Default prompt extracted from {benchmark_name} benchmark",
                instructions=best_instruction,
                task_description=task_description,
                signature=signature,
                meta_strategy="extracted"
            )
            
            config.add_variation(default_prompt)
            return config
    
    def _select_best_instruction(self, prompt_info: Dict[str, Any]) -> Optional[str]:
        """Select the best instruction from available options."""
        instructions = prompt_info.get('instructions', [])
        
        if not instructions:
            # Try to extract from signatures
            for sig in prompt_info.get('signatures', []):
                if sig.get('instructions'):
                    instructions.append(sig['instructions'])
                elif sig.get('docstring'):
                    instructions.append(sig['docstring'])
        
        if instructions:
            # Choose the longest instruction (likely most detailed)
            return max(instructions, key=len)
        
        return None
    
    def _select_best_signature(self, prompt_info: Dict[str, Any]) -> str:
        """Select the best signature from available options."""
        signatures = prompt_info.get('signatures', [])
        
        if signatures:
            # Try to construct a meaningful signature
            sig = signatures[0]  # Take first signature
            inputs = sig.get('input_fields', [])
            outputs = sig.get('output_fields', [])
            
            if inputs and outputs:
                input_str = ', '.join(inputs[:3])  # Limit to first 3
                output_str = ', '.join(outputs[:3])
                return f"{input_str} -> {output_str}"
        
        return "Input -> Output"
    
    def _select_best_task_description(self, prompt_info: Dict[str, Any]) -> str:
        """Select the best task description from available options."""
        descriptions = prompt_info.get('task_descriptions', [])
        
        if descriptions:
            # Choose the most descriptive one
            best_desc = max(descriptions, key=len)
            # Clean up the description
            return best_desc.split('\n')[0][:200].strip()
        
        return self._benchmark_name_to_description(prompt_info['benchmark_name'])
    
    def save_extracted_prompts(self, configs: Dict[str, BenchmarkConfig], output_dir: str = "extracted_prompts"):
        """Save extracted prompts to JSON files."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        print(f"\nSaving extracted prompts to {output_path}")
        
        for benchmark_name, config in configs.items():
            benchmark_dir = output_path / benchmark_name
            benchmark_dir.mkdir(exist_ok=True)
            
            for variation in config.prompt_variations:
                filename = f"{benchmark_name}_{variation.name}_prompt.json"
                filepath = benchmark_dir / filename
                variation.to_json_file(filepath)
                print(f"  Saved: {filepath}")
        
        print(f"✅ Saved {len(configs)} benchmark configurations")
        return output_path 