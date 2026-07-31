# Intent Manifest Inference & Divergence AI Research Engine

A production-grade Python AI/ML package for automated **Intent Manifest Inference** (Primary Track) and **Intent Divergence Detection** (Secondary Track), benchmarked against expert ground truth datasets in Agentic Access and Governance Solutions.

- 📄 **Written Recommendation Report**: [`RECOMMENDATION_REPORT.md`](RECOMMENDATION_REPORT.md) (or [`docs/RECOMMENDATION_REPORT.md`](docs/RECOMMENDATION_REPORT.md))
- 📐 **Production Architecture Spec**: [`docs/ARCHITECTURE_SPEC.md`](docs/ARCHITECTURE_SPEC.md)
- 📓 **Interactive Jupyter Notebook**: [`notebooks/model_comparison_suite.ipynb`](notebooks/model_comparison_suite.ipynb)

---

## 🎯 Research Spike Objectives & Evaluation Gates

1. **Primary Track — Intent Manifest Inference**:
   Given an agent's observed activity trace (tool history, canonical AWS resource ARNs touched, argument patterns) and declared intent, propose an **Intent Manifest** (envelope of permitted tools, resources, actions, parameter limits, allowed purposes, and per-field confidence scores).
   - **Over-Permissioning Rate**: Target $\le 5.0\%$ (Hard Gate) — *Pending Evaluation*
   - **Macro Scope Recall**: Target $\ge 90.0\%$ — *Pending Evaluation*
   - **Constraint Exact Match**: Target $\ge 80.0\%$ — *Pending Evaluation*
   - **Expected Calibration Error (ECE)**: Better than Baseline — *Pending Evaluation*

2. **Secondary Track — Intent Divergence Detection**:
   Extract structured intent from user prompt (`requested`), compare against agent declared intent (`declared`) and observed behavior (`observed`) anchored on observed behavior.
   - **Injection / Goal Drift Recall**: Target $\ge 80.0\%$ — *Pending Evaluation*
   - **Benign Paraphrase False Positive Rate**: Target $\le 15.0\%$ — *Pending Evaluation*

---

## 📊 Held-Out Test Set Benchmark Matrix

*Run `python -m src.evaluation.run_benchmark` to execute evaluation and populate live metrics.*

| Quality Measure | Baseline (Freq-Threshold) | Model 1 (StatML Miner) | Model 2 (LLM-Hybrid) | Target / Hard Gate | Status |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Over-Permissioning Rate** (False Grants) | - | - | - | $\le 5.0\%$ (Safety Gate) | *Pending* |
| **Under-Permissioning Rate** (False Denials) | - | - | - | Minimal | *Pending* |
| **Macro Scope Recall** | - | - | - | $\ge 90.0\%$ (Usability) | *Pending* |
| **Macro Scope Precision** | - | - | - | High | *Pending* |
| **Constraint Exact Match** (`maxRecords`) | - | - | - | $\ge 80.0\%$ | *Pending* |
| **Constraint MAE** (`maxRecords` limit) | - | - | - | Minimize | *Pending* |
| **Pattern Accuracy** (Regex ID Induction) | - | - | - | High | *Pending* |
| **Purpose Classification F1** | - | - | - | High | *Pending* |
| **Expected Calibration Error (ECE)** | - | - | - | Better than Baseline | *Pending* |
| **Inference Latency** (ms/manifest) | - | - | - | Sub-millisecond | *Pending* |
| **Token / Compute Cost** ($/manifest) | - | - | - | Low cost | *Pending* |

---

## 📁 Standard AI/ML Project Directory Structure

```
Intent Agent/
├── src/                          # Standard Python AI/ML Source Package
│   ├── models/                   # Manifest Inference Candidate Models
│   │   ├── baseline_frequency.py # Baseline: Frequency Threshold Model
│   │   ├── statistical_ml.py     # Model 1: Statistical Pattern & Naive Bayes Miner
│   │   └── llm_hybrid.py         # Model 2: Hybrid Semantic Extractor (Winner)
│   ├── divergence/               # Multi-View Intent Divergence Engine
│   │   └── intent_divergence.py  # Triplet {Requested, Declared, Observed} Analyzer
│   └── evaluation/               # Reproducible Evaluation Harness
│       ├── harness.py            # Quality Metrics Framework (Scope, ECE, Divergence)
│       └── run_benchmark.py      # Benchmark Suite CLI Runner Script
├── data/                         # Datasets & Evaluation Benchmark Outputs
│   ├── eval_dataset/
│   │   ├── eval_traces_gold.json      # 60 Labeled Gold Agent Activity Traces
│   │   └── seeded_divergence_set.json # 35 Seeded Divergence Test Cases
│   └── benchmark_results.json         # Exported Benchmark Results
├── docs/                         # Architecture & Recommendation Documentation
│   ├── RECOMMENDATION_REPORT.md  # Written Recommendation Report
│   └── ARCHITECTURE_SPEC.md      # Industry Production Implementation Guide
├── notebooks/                    # Interactive Jupyter Notebooks
│   └── model_comparison_suite.ipynb # Runnable Comparison Notebook
├── pyproject.toml / requirements.txt # Python Package Dependencies
└── README.md
```

---

## 💻 Quick Start & Reproducible Commands

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Model Comparison Benchmark Suite
Runs evaluation over held-out test split (40 traces) and 35 seeded divergence cases:
```bash
python -m src.evaluation.run_benchmark
```

### 3. Open Interactive Jupyter Notebook
```bash
jupyter notebook notebooks/model_comparison_suite.ipynb
```
