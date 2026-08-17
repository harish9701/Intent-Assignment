# Comprehensive Repository Folder Structure & Architecture Guide

This document provides a complete, standardized guide to the **repository layout, folder structure, source code modules, and documentation maps** for the **Intent Manifest Inference & Divergence Research Engine**.

---

## 📁 Repository Overview Map

```
Intent-Assignment/
├── data/                                      <-- EVALUATION DATASETS & METRICS EXPORTS
│   ├── eval_dataset/
│   │   ├── eval_traces_gold.json              <-- Track 1: 60 Labeled Expert Gold Activity Traces
│   │   └── seeded_divergence_set.json         <-- Track 2: 35 Seeded Injection & Benign Paraphrase Cases
│   └── benchmark_results.json                 <-- Model Evaluation Output JSON Export
│
├── src/                                       <-- PYTHON SOURCE CODE MODULES
│   ├── track1_manifest_inference/             <-- 1️⃣ PRIMARY TRACK: MANIFEST INFERENCE
│   │   ├── __init__.py                        <-- Track 1 Package Init & Exports
│   │   ├── baseline_frequency.py             <-- Pattern 1: Frequency-Threshold Baseline (N >= 1)
│   │   ├── statistical_ml.py                 <-- Pattern 2: Classical ML (TF-IDF + Naive Bayes + Buckets)
│   │   ├── llm_hybrid.py                     <-- Pattern 3: Hybrid Semantic Schema-Bounded Extractor (Winner)
│   │
│   ├── track2_intent_divergence/              <-- 2️⃣ SECONDARY TRACK: INTENT DIVERGENCE
│   │   ├── __init__.py                        <-- Track 2 Package Init & Exports
│   │   └── intent_divergence.py              <-- Triplet {Requested, Declared, Observed} Engine
│   │
│   ├── models/                                <-- Backward-Compatibility Alias Package
│   │   └── __init__.py                        <-- Re-exports Track 1 Candidate Models
│   │
│   ├── divergence/                            <-- Backward-Compatibility Alias Package
│   │   └── __init__.py                        <-- Re-exports Track 2 Divergence Engine
│   │
│   └── evaluation/                            <-- 🧪 REPRODUCIBLE EVALUATION HARNESS
│       ├── __init__.py                        <-- Evaluation Package Init
│       ├── harness.py                        <-- Field-by-Field Precision, Recall, ECE & MAE Calculator
│       └── run_benchmark.py                  <-- Held-Out Test Set Benchmark Runner Script
│
└── docs/                                      <-- STANDARDIZED LEARNING & RESEARCH DOCS
    ├── 01_PRIMARY_TRACK_MANIFEST_INFERENCE/   <-- Track 1 Dedicated Learning Guides
    │   ├── 01_PRIMARY_TRACK_OVERVIEW.md       <-- Track 1 Pipeline, Untrusted Prompt Rule & Train/Test Split
    │   ├── PATTERN_1_BASELINE_FREQUENCY.md    <-- Baseline Heuristic Mechanics, Q&As & Failure Analysis
    │   ├── PATTERN_2_STATISTICAL_ML.md        <-- Classical ML Pattern Miner Deep-Dive
    │   ├── PATTERN_3_HYBRID_LLM.md            <-- Hybrid Semantic Model Deep-Dive & Calibration
    │
    ├── 02_SECONDARY_TRACK_INTENT_DIVERGENCE/  <-- Track 2 Dedicated Learning Guides
    │   ├── 01_SECONDARY_TRACK_OVERVIEW.md     <-- Triplet Comparison Pipeline Overview
    │   └── TRIPLET_DIVERGENCE_ENGINE.md       <-- Divergence Metric, Injection Recall & Benign FPR
    │
    ├── 03_EVALUATION_AND_METRICS/             <-- Evaluation & Math Metrics Guides
    │   ├── HELD_OUT_BENCHMARK_SPEC.md         <-- 60 Gold Traces (20 Train / 40 Test) + 35 Seeded Cases
    │   └── MATHEMATICAL_FORMULAS_GUIDE.md     <-- Complete Math Formulas: Over-Permissioning, Recall, ECE
    │
    ├── ARCHITECTURE_SPEC.md                   <-- Enterprise Architecture & JWT/OPA Integration Blueprint
    └── RECOMMENDATION_REPORT.md               <-- Written Executive Recommendation & Trade-Off Analysis
```

---

## 🎯 Detailed Folder & Component Descriptions

### 1️⃣ Primary Track — Manifest Inference (`src/track1_manifest_inference/`)
- **Objective**: Given an agent's observed activity trace + context (tool history, resource ARNs, arguments, declared goal), infer what its authoritative `IntentManifest` should be.
- **Candidate Models**:
  1. `FrequencyBaselineModel` ([`baseline_frequency.py`](file:///c:/Users/chint/Music/Intent%20Agent/src/track1_manifest_inference/baseline_frequency.py)): Frequency counting heuristic ($\text{count} \ge 1$).
  2. `StatisticalPatternModel` ([`statistical_ml.py`](file:///c:/Users/chint/Music/Intent%20Agent/src/track1_manifest_inference/statistical_ml.py)): TF-IDF + Naive Bayes + discrete limits + regex pattern induction.
  3. `LLMHybridManifestModel` ([`llm_hybrid.py`](file:///c:/Users/chint/Music/Intent%20Agent/src/track1_manifest_inference/llm_hybrid.py)): Schema-bounded domain ontology extractor (Recommended Winner).

---

### 2️⃣ Secondary Track — Intent Divergence (`src/track2_intent_divergence/`)
- **Objective**: Extract structured intent from user prompt and compare against agent declared intent and observed behavior.
- **Key Engine**:
  - `IntentDivergenceEngine` ([`intent_divergence.py`](file:///c:/Users/chint/Music/Intent%20Agent/src/track2_intent_divergence/intent_divergence.py)): Computes multi-dimensional divergence score across `{Requested, Declared, Observed}` views. Anchored on Observed behavior.

---

### 3️⃣ Reproducible Evaluation Harness (`src/evaluation/`)
- `EvaluationHarness` ([`harness.py`](file:///c:/Users/chint/Music/Intent%20Agent/src/evaluation/harness.py)): Calculates Over-Permissioning Rate, Under-Permissioning Rate, Macro Scope Precision/Recall, Constraint Exact Match, Constraint MAE, Purpose F1, and ECE Calibration Error.
- Benchmark Execution Script ([`run_benchmark.py`](file:///c:/Users/chint/Music/Intent%20Agent/src/evaluation/run_benchmark.py)): Evaluates models on held-out test split and writes JSON metrics export.

---

### 4️⃣ Document Map (`docs/`)
- **`01_PRIMARY_TRACK_MANIFEST_INFERENCE/`**: Detailed guides for Track 1 pipeline, untrusted prompt principle, and individual pattern deep dives.
- **`02_SECONDARY_TRACK_INTENT_DIVERGENCE/`**: Guides for Track 2 triplet pipeline, risk verb lists, and benign paraphrase filtering.
- **`03_EVALUATION_AND_METRICS/`**: Held-out benchmark dataset specification and step-by-step mathematical formulas guide.
