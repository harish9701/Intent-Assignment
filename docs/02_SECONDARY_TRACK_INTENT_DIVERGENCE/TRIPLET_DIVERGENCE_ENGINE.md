# Secondary Track — Triplet Divergence Detection Engine

📁 **Source Code**: [`src/divergence/intent_divergence.py`](file:///c:/Users/chint/Music/Intent%20Agent/src/divergence/intent_divergence.py)  
🏷️ **Class Name**: `IntentDivergenceEngine`  
📊 **Track Role**: Secondary Track Divergence Detector

---

## 🎯 Model Objective
Detects when an agent's observed behavior strays from requested/declared intent:
- **Target Drift**: Requested target `C1000`, observed target `C2999` or `*`.
- **Action Escalation**: Requested `READ`, observed `DELETE`, `EXPORT`, `DUMP_ALL`.
- **Scope Expansion**: Parameter `limit > 10000` or injection payloads (`;`, `DROP TABLE`, `ftp://`).

---

## ⚙️ How It Works

1. **Structured Intent Extraction**: Parses target IDs, action verbs, and entity categories from requested/declared text.
2. **Anchor Verification**: Treats **Observed Behavior as the anchor**.
3. **Drift Scoring**: Computes max divergence score across Target Drift, Action Escalation, and Scope Expansion.
4. **Benign Paraphrase Filtering**: Normalizes synonyms (`lookup`, `fetch`, `retrieve`, `get` $\rightarrow$ `READ`) to achieve **0.00% False Positive Rate on benign rewording**.

---

## 📊 Benchmark Results (35 Seeded Cases)

- **Injection / Goal Drift Recall**: **95.00% (PASSED $\ge 80\%$)**
- **Benign Paraphrase FPR**: **0.00% (PASSED $\le 15\%$)**
- **Divergence F1 Score**: **0.9744**
- **Confusion Matrix**: TP=19, FP=0, TN=15, FN=1
