"""
config.py - Configuration management for Uyghur Linguistic Translation Pipeline
"""

import os
from dataclasses import dataclass, field
import torch

# Base project directory (assumed to be root of src)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@dataclass
class PipelineConfig:
    # Component Paths
    mudt_model_dir: str = os.path.join(BASE_DIR, "MUDT", "exp", "mudt_xlmr_base", "model")
    apertium_dir: str = os.path.join(BASE_DIR, "apertium-uig")
    disambiguater_model_dir: str = os.path.join(BASE_DIR, "disambiguater", "final_model_qwen2b", "best")
    dictionary_db_path: str = os.path.join(BASE_DIR, "dictionary", "uig_eng_dict_complete.sqlite")

    # Hardware & Device Settings
    device: str = field(
        default_factory=lambda: "cuda:0" if torch.cuda.is_available() else "cpu"
    )
    torch_dtype: torch.dtype = torch.bfloat16

    # Model Generation Parameters (for Qwen Disambiguator)
    disambig_max_new_tokens: int = 8
    disambig_batch_size: int = 16

    # Downstream LLM Translation Settings
    llm_model: str = os.environ.get("TRANSLATION_LLM_MODEL", "openai/gpt-4o")
    llm_api_base: str = os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    llm_api_key: str = field(
        default_factory=lambda: (
            os.environ.get("OPENROUTER_API_KEY")
            or os.environ.get("OPENAI_API_KEY")
            or ""
        )
    )
    llm_temperature: float = 0.2
    llm_max_tokens: int = 512

    # Formatting options for in-context prompts
    # Options: 'markdown_table' | 'structured_list' | 'concise' | 'json'
    prompt_format: str = "markdown_table"
    include_clause_structure: bool = True
    include_dictionary_glosses: bool = True
    include_morphology: bool = True
    include_dependency: bool = True

    def validate(self):
        """Validate paths and configuration parameters."""
        errors = []
        if not os.path.exists(self.mudt_model_dir):
            errors.append(f"MUDT model directory not found: {self.mudt_model_dir}")
        if not os.path.exists(self.apertium_dir):
            errors.append(f"Apertium directory not found: {self.apertium_dir}")
        if not os.path.exists(self.disambiguater_model_dir):
            errors.append(f"Disambiguater model directory not found: {self.disambiguater_model_dir}")
        if not os.path.exists(self.dictionary_db_path):
            errors.append(f"Dictionary SQLite DB not found: {self.dictionary_db_path}")

        if errors:
            raise FileNotFoundError("Configuration path validation failed:\n" + "\n".join(errors))
