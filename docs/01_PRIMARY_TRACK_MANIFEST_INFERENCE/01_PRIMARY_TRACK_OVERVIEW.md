# Primary Track: Intent Manifest Inference

📁 **Python Source Package**: [`src/models/`](file:///c:/Users/chint/Music/Intent%20Agent/src/models/)  
📊 **Held-Out Test Dataset**: [`data/eval_dataset/eval_traces_gold.json`](file:///c:/Users/chint/Music/Intent%20Agent/data/eval_dataset/eval_traces_gold.json) (60 Gold Traces)

---

## 🎯 Track Objective & Core Research Question

The Primary Track answers the core question:
> **"Given an agent's observed activity trace and context (tool history, canonical resource ARNs touched, argument patterns, and declared goal), can a model infer what its authoritative Intent Manifest should be?"**

An **Intent Manifest** is an expert-pre-approved cryptographic envelope specifying:
- `allowedPurposes`: Business purpose categories permitted for this task.
- `scope.tools`: Whitelist of permitted tool functions.
- `scope.resources`: Whitelist of canonical resource identifiers (e.g. AWS ARNs).
- `scope.actions`: Whitelist of permitted API operations (e.g. `dynamodb:GetItem`, `s3:GetObject`).
- `constraints`: Parametric limits (`maxRecords`, customer ID regex pattern, `allowPagination`).
- `dataHandling`: Security classification ceilings (`INTERNAL`, `RESTRICTED`, `CONFIDENTIAL_PHI`).

---

## 📐 Primary Track Pipeline Architecture

```
                 AGENT ACTIVITY + CONTEXT
                            │
             ┌──────────────┼──────────────┐
             │              │              │
        Tool history    Resources      Arguments
             │              │              │
             └──────────────┼──────────────┘
                            │
                      Declared goal
                            │
                            ▼
                 ┌────────────────────┐
                 │ Manifest           │
                 │ Inference Model    │
                 └──────────┬─────────┘
                            ▼
                    PROPOSED MANIFEST
                            │
                            ▼
                      GOLD MANIFEST
                    (expert-authored)
                            │
                            ▼
                Accuracy / Safety Metrics
```

---

## ⚠️ Critical Rule: The Untrusted Prompt Principle

> **"Natural-language goals and prompts are treated as untrusted features, never as ground truth. Observed behavior is the anchor."**

Why?
- A user prompt might say: *"Just read customer C1000."*
- But the user prompt could be a **prompt injection attack** or **covert exfiltration attempt**.
- The model must NOT blindly trust whatever natural language prompt text says.
- Instead, **Observed Activity (tools executed, ARNs touched, parameters passed)** is treated as the ground truth anchor to infer the actual least-privilege manifest.

---

## 🎓 Training Set vs. Held-Out Test Set Split

To ensure scientific reproducibility, the dataset of 60 expert gold activity traces is divided:

```
                      60 Expert Gold Traces
                               │
            ┌──────────────────┴──────────────────┐
            ▼                                     ▼
     Training Set                           Held-Out Test Set
     (20 Traces)                            (40 Traces)
            │                                     │
            ▼                                     ▼
   Model learns patterns /             Used purely for evaluation.
   trains Naive Bayes classifier       Model NEVER sees test gold
            │                          answers during inference!
            ▼                                     │
    Predicted Manifest ───────────────────────────┘
                                │
                                ▼
                       Evaluation Harness
                 (Computes Over-permissioning,
                  Recall, ECE, MAE metrics)
```

---

## 🔬 Investigating Candidate Models in this Track

This learning guide focuses on the three locally runnable primary patterns:

1. **Pattern 1 — Baseline Frequency Model** ([`src/models/baseline_frequency.py`](file:///c:/Users/chint/Music/Intent%20Agent/src/models/baseline_frequency.py)): Frequency threshold heuristic.
2. **Pattern 2 — Statistical ML Miner** ([`src/models/statistical_ml.py`](file:///c:/Users/chint/Music/Intent%20Agent/src/models/statistical_ml.py)): TF-IDF + Naive Bayes + discrete limits + regex pattern induction.
3. **Pattern 3 — Hybrid LLM Extractor** ([`src/models/llm_hybrid.py`](file:///c:/Users/chint/Music/Intent%20Agent/src/models/llm_hybrid.py)): Domain ontology schema graph + Bayesian calibration (Winner).
