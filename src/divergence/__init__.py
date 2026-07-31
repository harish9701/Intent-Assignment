"""
Intent Divergence Package.
Contains multi-view triplet analyzer for detecting prompt injection & goal drift vs benign paraphrase.
"""
from src.divergence.intent_divergence import IntentDivergenceEngine

__all__ = ["IntentDivergenceEngine"]
