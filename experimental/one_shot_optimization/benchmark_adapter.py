"""
Benchmark adapter for handling non-standard signature patterns.
"""

from typing import Dict, List, Optional, Any
from abc import ABC, abstractmethod
from pathlib import Path
import re
import ast


class BenchmarkAdapter(ABC):
    """Base class for benchmark-specific adapters."""
    
    @abstractmethod
    def get_signatures(self) -> List[Dict[str, Any]]:
        """Extract signature information for the benchmark."""
        pass
    
    @abstractmethod
    def get_default_prompt(self) -> str:
        """Get the default prompt for the benchmark."""
        pass
    
    @abstractmethod
    def get_task_description(self) -> str:
        """Get a description of the benchmark task."""
        pass


class HoverAdapter(BenchmarkAdapter):
    """Adapter for hover benchmark with string-based signatures."""
    
    def __init__(self, benchmark_dir: Path):
        self.benchmark_dir = benchmark_dir
        self.signatures = self._extract_string_signatures()
    
    def _extract_string_signatures(self) -> List[Dict[str, Any]]:
        """Extract string-based signatures from hover program file."""
        program_file = self.benchmark_dir / "hover_program.py"
        signatures = []
        
        if program_file.exists():
            with open(program_file, 'r') as f:
                content = f.read()
            
            # Find all dspy.Predict/ChainOfThought calls with string signatures
            pattern = r'dspy\.(Predict|ChainOfThought)\s*\(\s*["\']([^"\']+)["\']'
            matches = re.findall(pattern, content)
            
            signature_mappings = {
                "claim,passages->summary": {
                    "name": "SummarizePassages",
                    "instruction": "Summarize passages that are relevant to the given claim.",
                    "signature_key": "summarize1"
                },
                "claim,context,passages->summary": {
                    "name": "SummarizeWithContext",
                    "instruction": "Summarize passages with additional context about the claim.",
                    "signature_key": "summarize2"
                },
                "claim,summary_1->query": {
                    "name": "GenerateQueryHop2",
                    "instruction": "Generate a search query based on the claim and initial summary.",
                    "signature_key": "query_hop2"
                },
                "claim,summary_1,summary_2->query": {
                    "name": "GenerateQueryHop3",
                    "instruction": "Generate a refined search query based on claim and multiple summaries.",
                    "signature_key": "query_hop3"
                }
            }
            
            # Create a unique set to track processed signatures
            processed_signatures = set()
            
            for method, sig_string in matches:
                if sig_string in signature_mappings and sig_string not in processed_signatures:
                    processed_signatures.add(sig_string)
                    mapping = signature_mappings[sig_string]
                    inputs, output = sig_string.split('->')
                    input_fields = [i.strip() for i in inputs.split(',')]
                    
                    signatures.append({
                        'name': mapping['name'],
                        'signature': sig_string,
                        'method': method,
                        'instruction': mapping['instruction'],
                        'input_fields': input_fields,
                        'output_fields': [output.strip()],
                        'signature_key': mapping['signature_key']
                    })
        
        return signatures
    
    def get_signatures(self) -> List[Dict[str, Any]]:
        """Return extracted signatures."""
        return self.signatures
    
    def get_default_prompt(self) -> str:
        """Get the default prompt for hover."""
        return "Summarize passages that are relevant to the given claim."
    
    def get_task_description(self) -> str:
        """Get hover task description."""
        return "Multi-hop fact verification using Wikipedia evidence"


class IReRaAdapter(BenchmarkAdapter):
    """Adapter for IReRa benchmark with utils-based signatures."""
    
    def __init__(self, benchmark_dir: Path, dataset: str = "esco_tech"):
        self.benchmark_dir = benchmark_dir
        self.dataset = dataset
        self.signatures = self._extract_utils_signatures()
    
    def _extract_utils_signatures(self) -> List[Dict[str, Any]]:
        """Extract signatures from IReRa utils file."""
        utils_file = self.benchmark_dir / "irera_utils.py"
        signatures = []
        
        if utils_file.exists():
            with open(utils_file, 'r') as f:
                content = f.read()
            
            # Parse AST to find signature classes
            tree = ast.parse(content)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    # Check if it's a Signature class
                    for base in node.bases:
                        if (isinstance(base, ast.Attribute) and 
                            base.attr == 'Signature'):
                            
                            # Try to get docstring - check both regular docstring and __doc__ assignment
                            docstring = ast.get_docstring(node)
                            
                            # If no regular docstring, look for __doc__ assignment
                            if not docstring:
                                for item in node.body:
                                    if isinstance(item, ast.Assign):
                                        for target in item.targets:
                                            if (isinstance(target, ast.Name) and 
                                                target.id == '__doc__'):
                                                # Extract the string value
                                                if isinstance(item.value, ast.JoinedStr):
                                                    # Handle f-strings
                                                    parts = []
                                                    for value in item.value.values:
                                                        if isinstance(value, ast.Constant):
                                                            parts.append(value.value)
                                                    docstring = ''.join(parts)
                                                elif isinstance(item.value, ast.Constant):
                                                    docstring = item.value.value
                                                break
                            
                            if docstring:
                                # Extract field information
                                fields = self._extract_fields_from_class(node)
                                
                                signatures.append({
                                    'name': node.name,
                                    'instruction': docstring.strip(),
                                    'input_fields': fields['input'],
                                    'output_fields': fields['output']
                                })
        
        return signatures
    
    def _extract_fields_from_class(self, node: ast.ClassDef) -> Dict[str, List[str]]:
        """Extract input and output fields from a signature class."""
        fields = {'input': [], 'output': []}
        
        for item in node.body:
            if isinstance(item, ast.Assign):
                for target in item.targets:
                    if isinstance(target, ast.Name):
                        field_name = target.id
                        if field_name not in ['__doc__', '__module__', '__qualname__']:
                            # Check the assignment value for field type
                            if isinstance(item.value, ast.Call):
                                # Check if it's dspy.InputField or dspy.OutputField
                                if hasattr(item.value.func, 'attr'):
                                    field_type = item.value.func.attr
                                    if 'Input' in field_type:
                                        fields['input'].append(field_name)
                                    elif 'Output' in field_type:
                                        fields['output'].append(field_name)
                            # Default to input if can't determine
                            elif field_name == 'text':
                                fields['input'].append(field_name)
                            elif field_name in ['output', 'result']:
                                fields['output'].append(field_name)
        
        # If no fields found, use defaults based on signature name
        if not fields['input'] and not fields['output']:
            fields['input'] = ['text']
            fields['output'] = ['output']
        
        return fields
    
    def get_signatures(self) -> List[Dict[str, Any]]:
        """Return extracted signatures filtered by dataset."""
        # Filter signatures based on dataset
        if 'esco' in self.dataset:
            return [s for s in self.signatures if 'ESCO' in s['name']]
        elif 'biodex' in self.dataset:
            return [s for s in self.signatures if 'BioDEX' in s['name']]
        return self.signatures
    
    def get_default_prompt(self) -> str:
        """Get the default prompt for IReRa based on dataset."""
        if 'esco' in self.dataset:
            return "Given a snippet from a job vacancy, identify all the ESCO job skills mentioned. Always return skills."
        elif 'biodex' in self.dataset:
            return "Given a snippet from a medical article, identify the adverse drug reactions affecting the patient. Always return reactions."
        return "Extract relevant information from the given text."
    
    def get_task_description(self) -> str:
        """Get IReRa task description."""
        if 'esco' in self.dataset:
            return "Extract job skills from job vacancy texts"
        elif 'biodex' in self.dataset:
            return "Extract adverse drug reactions from medical articles"
        return "Information extraction and reasoning tasks"


class AdapterRegistry:
    """Registry for benchmark adapters."""
    
    _adapters = {
        'hover': HoverAdapter,
        'IReRa': IReRaAdapter
    }
    
    @classmethod
    def get_adapter(cls, benchmark_name: str, benchmark_dir: Path, **kwargs) -> Optional[BenchmarkAdapter]:
        """Get an adapter instance for a benchmark."""
        if benchmark_name in cls._adapters:
            adapter_class = cls._adapters[benchmark_name]
            return adapter_class(benchmark_dir, **kwargs)
        return None
    
    @classmethod
    def has_adapter(cls, benchmark_name: str) -> bool:
        """Check if a benchmark has an adapter."""
        return benchmark_name in cls._adapters 