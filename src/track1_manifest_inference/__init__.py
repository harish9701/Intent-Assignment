"""
Primary Track 1: Intent Manifest Inference Package.
Contains Candidate Models:
- FrequencyBaselineModel (Baseline heuristic)
- StatisticalPatternModel (Model 1: Statistical Pattern Miner)
- LLMHybridManifestModel (Model 2: Hybrid Semantic Extractor - Recommended)
- OllamaLLMManifestModel (Model 3: Local Open-Weights LLM)
"""
from src.track1_manifest_inference.baseline_frequency import FrequencyBaselineModel
from src.track1_manifest_inference.statistical_ml import StatisticalPatternModel
from src.track1_manifest_inference.llm_hybrid import LLMHybridManifestModel
from src.track1_manifest_inference.ollama_llm import OllamaLLMManifestModel
from src.track1_manifest_inference.authbench_sufficiency_tightness import AuthBenchInspiredManifestModel

__all__ = [
    "FrequencyBaselineModel",
    "StatisticalPatternModel",
    "LLMHybridManifestModel",
    "OllamaLLMManifestModel",
    "AuthBenchInspiredManifestModel"
]
