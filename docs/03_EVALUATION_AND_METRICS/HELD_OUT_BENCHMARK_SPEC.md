# Evaluation Spec — Held-Out Benchmark & Dataset Division

📁 **Harness Code**: [`src/evaluation/harness.py`](file:///c:/Users/chint/Music/Intent%20Agent/src/evaluation/harness.py)  
📁 **Runner Code**: [`src/evaluation/run_benchmark.py`](file:///c:/Users/chint/Music/Intent%20Agent/src/evaluation/run_benchmark.py)

---

## 📊 Dataset Structure & Split

1. **Gold Activity Traces Dataset** (`data/eval_dataset/eval_traces_gold.json`):
   - **Total Traces**: 60 expert gold activity traces across 6 enterprise domain categories.
   - **Train Set**: 20 Traces (used to train Naive Bayes classifier in Model 1).
   - **Held-Out Test Set**: 40 Traces (used purely for model evaluation — models never see gold answers during inference).

2. **Seeded Divergence Dataset** (`data/eval_dataset/seeded_divergence_set.json`):
   - **Total Test Cases**: 35 seeded test cases.
   - **Benign Paraphrase Cases**: 15 cases.
   - **Prompt Injection & Goal Drift Cases**: 20 cases.
