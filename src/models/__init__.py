"""
Manifest Inference Models Package.
Contains Candidate Model implementations:
- FrequencyBaselineModel (Baseline heuristic)
- StatisticalPatternModel (Model 1: Statistical Pattern Miner)
- LLMHybridManifestModel (Model 2: Hybrid Semantic Extractor - Recommended)
"""
from src.models.baseline_frequency import FrequencyBaselineModel
from src.models.statistical_ml import StatisticalPatternModel
from src.models.llm_hybrid import LLMHybridManifestModel

__all__ = [
    "FrequencyBaselineModel",
    "StatisticalPatternModel",
    "LLMHybridManifestModel"
]
