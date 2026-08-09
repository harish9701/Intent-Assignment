# Primary Track — Pattern 2: Classical Statistical ML & Pattern Miner

📁 **Source Code**: [`src/models/statistical_ml.py`](file:///c:/Users/chint/Music/Intent%20Agent/src/models/statistical_ml.py)  
🏷️ **Class Name**: `StatisticalPatternModel`  
📊 **Track Role**: Primary Track Classical ML Candidate Model

---

## 🎯 Model Objective
Pattern 2 investigates classical machine learning techniques:
> *"Can we combine TF-IDF + Naive Bayes text classification with discrete limit bucketing and regex induction to infer tighter manifest constraints?"*

---

## ⚙️ How It Works

1. **Training (`fit(train_traces)`)**: Fits a TF-IDF vectorizer and `MultinomialNB` classifier on 20 training traces.
2. **Text Purpose Classification**: Predicts `allowedPurposes` label from prompt and declared intent text.
3. **Limit Bucketing**: Rounds observed limits to discrete bucket ceilings ($50, 100, 250, 500, \dots$).
4. **Regex Induction**: Evaluates ID parameters against regular expression templates to induce regex pattern (e.g. `^[A-Z][0-9]+$`).

---

## 📊 Benchmark Results

- **Over-Permissioning Rate**: 0.00% (Passed $\le 5\%$)
- **Under-Permissioning Rate**: 41.94% (High false denials on short traces)
- **Macro Scope Recall**: 58.06% ($< 90\%$)
- **Constraint Exact Match**: **85.00% (PASSED $\ge 80\%$)**
- **Inference Latency**: **0.61 ms (Fast CPU inference)**
