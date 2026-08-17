# Primary Track — Pattern 2: Classical Statistical ML & Pattern Miner

📁 **Source Code**: [`src/track1_manifest_inference/statistical_ml.py`](file:///c:/Users/chint/Music/Intent%20Agent/src/track1_manifest_inference/statistical_ml.py)  
📦 **ML Classifier Variants**: [`src/track1_manifest_inference/statistical_ml_variants.py`](file:///c:/Users/chint/Music/Intent%20Agent/src/track1_manifest_inference/statistical_ml_variants.py)  
🏷️ **Class Name**: `StatisticalPatternModel`  
📊 **Track Role**: Primary Track Classical ML Candidate Model

---

## 🎯 Model Objective
Pattern 2 investigates classical machine learning and deterministic pattern induction techniques:
> *"Can we combine TF-IDF feature extraction, probabilistic text classification, discrete limit quantiles, and regular expression pattern induction to infer authoritative Intent Manifests?"*

## Plain-language summary: Pattern 2

### How it works

Pattern 2 has two parts. A trained **TF-IDF + Naive Bayes** classifier reads the user prompt and declared intent, then predicts the task's business purpose. Fixed rules inspect the observed tool calls to fill in the manifest scope, record limit, identifier pattern, pagination setting, and data classification.

It is like teaching a small text classifier that “ledger” often means finance and “patient” often means healthcare, while still taking concrete permissions only from the trace.

### Pros

- **Learns from labelled examples:** it can recognise purpose from wording instead of only exact keywords.
- **Fast and inexpensive:** TF-IDF and Naive Bayes run quickly on a CPU.
- **Explainable:** you can inspect the input text, labels, vocabulary, and deterministic rules.
- **No external service:** training and prediction can run locally.
- **Safer separation of duties:** purpose is learned, while concrete scope is based on observed calls.

### Cons

- **Needs good labelled data:** incorrect, too-small, or unrepresentative examples lead to weak predictions.
- **Limited language understanding:** it sees token patterns, not deep meaning; unfamiliar wording may be misclassified.
- **Single-label purpose:** it trains only on the first allowed purpose, even when a task has several purposes.
- **Rules need maintenance:** new resource types or ID formats require code/policy updates.
- **Unknown ID fallback is broad:** `.*` should be replaced by a fail-closed rule in production.

### Best use

Use Pattern 2 when you have labelled historical traces, need a fast local baseline, and want to explain why the system predicted a purpose. It is a strong learning and baseline model, not a replacement for policy review.

---

## 📐 Mathematical Formulation & Core Equations

### 1. TF-IDF Text Feature Extraction
The natural language text $d$ (concatenation of `user_prompt` and `agent_declared_intent`) is converted into a numerical feature vector $\mathbf{x}$ using Term Frequency-Inverse Document Frequency (TF-IDF):

$$TF(t, d) = \frac{f_{t,d}}{\sum_{t' \in d} f_{t',d}}$$

$$IDF(t, D) = \ln \left( \frac{1 + |D|}{1 + |\{d \in D : t \in d\}|} \right) + 1$$

$$\text{TF-IDF}(t, d, D) = TF(t, d) \times IDF(t, D)$$

Where:
- $f_{t,d}$ is the frequency of token $t$ in text document $d$.
- $|D|$ is the total number of traces in the training dataset ($|D| = 20$).
- $|\{d \in D : t \in d\}|$ is the document frequency of token $t$.

---

### 2. Multinomial Naive Bayes Purpose Classifier (Default)
The model predicts the primary business purpose category $\hat{y} \in Y$ (`allowedPurposes`) using Bayes' Theorem under the conditional independence assumption:

$$P(y \mid \mathbf{x}) = \frac{P(y) \prod_{i=1}^{n} P(x_i \mid y)^{x_i}}{P(\mathbf{x})}$$

Applying the $\log$ transformation for numerical stability, the decision rule becomes:

$$\hat{y} = \arg\max_{y \in Y} \left[ \ln P(y) + \sum_{i=1}^{n} x_i \cdot \ln \hat{\theta}_{yi} \right]$$

Where the class-conditional word probabilities $\hat{\theta}_{yi}$ are estimated using **Laplace Smoothing** ($\alpha = 1.0$):

$$\hat{\theta}_{yi} = P(x_i \mid y) = \frac{N_{yi} + \alpha}{N_y + \alpha n}$$

- $N_{yi} = \sum_{d \in y} x_{d,i}$ is the total TF-IDF weight of word $i$ in class $y$.
- $N_y = \sum_{i=1}^{n} N_{yi}$ is the total weight of all words in class $y$.
- $n$ is the total vocabulary size.

---

### 3. Quantized Limit Bucketing Formula
Rather than guessing arbitrary numbers for `constraints.maxRecords`, Pattern 2 collects all integer `limit` parameters observed in trace calls $L = \{l_1, l_2, \dots, l_m\}$ and rounds the peak limit $\max(L)$ up to the nearest security ceiling bucket $B$:

$$B = \{50, 100, 250, 300, 500, 1000, 2500, 5000, 10000\}$$

$$\text{maxRecords}_{\text{pred}} = \min \{ b \in B : b \ge \max(L) \}$$

If $L = \emptyset$, it defaults to standard ceiling $500$.

---

### 4. Regular Expression Pattern Induction
Instead of using unsafe wildcards (`.*`), Pattern 2 samples parameter values $s$ (e.g. `"C1000"`, `"CUST-10010"`) and matches them against regular expression templates $T = \{(R_k, P_k)\}_{k=1}^K$:

$$\text{Pattern}(s) = \begin{cases} P_k & \text{if } s \text{ matches regex } R_k \\ ".*" & \text{otherwise} \end{cases}$$

Supported Induced Templates:
- `"C1000"` $\rightarrow$ `^[A-Z][0-9]+$`
- `"CUST-10010"` $\rightarrow$ `^CUST-[0-9]{5}$`
- `"MED-123456"` $\rightarrow$ `^MED-[0-9]{6}$`
- `"SEC-9012-US"` $\rightarrow$ `^SEC-[0-9]{4}-[A-Z]{2}$`
- `"ORD-12345678"` $\rightarrow$ `^ORD-[0-9]{8}$`

---

### 5. Multi-Factor Confidence Score Calibration
Confidence scores $C \in [0.85, 0.90]$ are assigned based on probabilistic feature support:

$$C_{\text{field}} = w_{\text{base}} + w_{\text{support}} \cdot \mathbb{I}(\text{evidence\_present})$$

---

## 🧪 Classical Machine Learning Classifier Variants

To compare how different classical ML algorithms perform on prompt text classification, Pattern 2 includes alternative classifier models implemented in [`src/track1_manifest_inference/statistical_ml_variants.py`](file:///c:/Users/chint/Music/Intent%20Agent/src/track1_manifest_inference/statistical_ml_variants.py) (re-exported in [`src/models/statistical_ml_variants.py`](file:///c:/Users/chint/Music/Intent%20Agent/src/models/statistical_ml_variants.py)):

### Variant 1: Logistic Regression (`LogisticRegressionPatternModel`)
Uses a linear model with L2 regularization to estimate purpose class probabilities via the Softmax function:

$$P(y = k \mid \mathbf{x}) = \frac{e^{\mathbf{w}_k^T \mathbf{x} + b_k}}{\sum_{j=1}^K e^{\mathbf{w}_j^T \mathbf{x} + b_j}}$$

$$\hat{y} = \arg\max_{k} P(y = k \mid \mathbf{x})$$

- **Strengths**: Smooth probabilistic decision boundary, robust against small weight variations.

---

### Variant 2: Linear Support Vector Machine (`LinearSVCPatternModel`)
Finds maximum-margin hyperplanes separating purpose classes in TF-IDF feature space by minimizing hinge loss with L2 penalty:

$$\min_{\mathbf{w}, b} \frac{1}{2} \|\mathbf{w}\|^2 + C \sum_{i=1}^M \max\left(0, 1 - y_i (\mathbf{w}^T \mathbf{x}_i + b)\right)$$

$$\hat{y} = \arg\max_{k} (\mathbf{w}_k^T \mathbf{x} + b_k)$$

- **Strengths**: Highly effective in high-dimensional text sparse feature spaces.

---

### Variant 3: Random Forest Classifier (`RandomForestPatternModel`)
Constructs an ensemble of $N=100$ decision trees, each trained on bootstrap samples of the training data using random feature sub-selection (Gini Impurity):

$$\text{Gini}(D) = 1 - \sum_{k=1}^K p_k^2$$

$$\hat{y} = \text{mode} \left\{ T_1(\mathbf{x}), T_2(\mathbf{x}), \dots, T_N(\mathbf{x}) \right\}$$

- **Strengths**: Captures non-linear term interactions across prompt keywords.

---

## ⚙️ How It Works (Step-by-Step Execution Flow)

1. **Training (`fit(train_traces)`)**: Fits `TfidfVectorizer` + Classifier (Naive Bayes / Logistic Regression / SVM / Random Forest) on training traces to map prompt text $\rightarrow$ primary purpose label.
2. **Trace Log Scanning**: Extracts tools, resource ARNs, actions, limits, and sample ID parameters.
3. **Quantile Bucketing**: Quantizes max limit into standard ceiling $b \in B$.
4. **Regex Induction**: Maps sample ID to formal parameter regex pattern.
5. **Security Classification Mapping**: Assigns `RESTRICTED` or `CONFIDENTIAL_PHI` based on resource keywords (`redshift`, `healthlake`, etc.).

---

## 📊 Benchmark Results (Held-Out Test Set)

| Model Variant | Class Name | Purpose F1 | Over-Perm. | Exact Match (`maxRecords`) | Latency (ms) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Multinomial Naive Bayes** (Default) | `StatisticalPatternModel` | **0.6667** | **0.00%** | **85.00%** | **0.61 ms** |
| **Logistic Regression** | `LogisticRegressionPatternModel` | **0.6667** | **0.00%** | **85.00%** | **0.82 ms** |
| **Linear Support Vector Machine** | `LinearSVCPatternModel` | **0.6667** | **0.00%** | **85.00%** | **0.75 ms** |
| **Random Forest** | `RandomForestPatternModel` | **0.6667** | **0.00%** | **85.00%** | **3.40 ms** |

---

## Beginner guide: exactly how the dataset is used

Pattern 2 is a **supervised classifier plus deterministic rules**. It learns only the task purpose from labelled examples. It does not learn the entire manifest from scratch.

```text
Training:   prompt + declared intent  ->  known purpose label
Prediction: new prompt + declaration  ->  predicted purpose

Observed tool calls -> scope, limits, identifier pattern, pagination, data policy
```

### One row in the dataset

`data/eval_dataset/eval_traces_gold.json` is a JSON list. Each object is one agent trace, for example:

```json
{
  "trace_id": "trace_001",
  "agent_id": "agt_001",
  "user_prompt": "Audit financial ledger entries for account CUST-10001",
  "agent_declared_intent": "Executing financial payroll auditing...",
  "domain_category": "financial_payroll_auditing",
  "tool_call_history": [{"tool_name": "financial_ledger.query", "parameters": {"limit": 100}}],
  "observed_summary": {},
  "gold_manifest": {"allowedPurposes": ["financial_payroll_auditing"]}
}
```

| Dataset field | Plain-English meaning | Current Pattern 2 use |
| --- | --- | --- |
| `trace_id` | Unique example number. | Used to name the generated manifest. |
| `agent_id` | Agent that performed the task. | Copied to the generated manifest. |
| `user_prompt` | The user's natural-language request. | **Classifier input.** |
| `agent_declared_intent` | The agent's statement of what it plans to do. | **Classifier input.** |
| `domain_category` | Domain supplied by the synthetic dataset. | Used only when the classifier has not been trained. |
| `tool_call_history` | Every actual tool call. | Used by rules, not passed to Naive Bayes. |
| `observed_summary` | Convenient summary of calls. | Not read by the current implementation. |
| `gold_manifest` | Expert-approved expected answer. | Supplies the training label and held-out evaluation answer. |

Each `tool_call_history` item contains these important fields:

| Call field | Example | How it becomes manifest data |
| --- | --- | --- |
| `tool_name` | `crm_report_tool.read` | Added to `scope.tools`. |
| `resource_arn` | `arn:aws:healthlake:...` | Added to `scope.resources`; keywords help set classification. |
| `action` | `dynamodb:Query` | Added to `scope.actions`. |
| `parameters.limit` | `100` | The highest observed value is rounded up to a policy bucket. |
| ID parameters such as `customer_id` or `patient_id` | `MED-100001` | Matched to a known safe regex template. |
| `parameters.offset` | `20` | A positive value enables pagination. |

### Train/test split: do not let answers leak

The benchmark has 60 traces. It trains on the first 20 and tests on the remaining 40:

```python
train_traces = gold_traces[:20]
test_traces = gold_traces[20:]
```

Only the training traces give the model a gold label. During evaluation, the test trace's `gold_manifest` is kept away from `predict_manifest`; it is used only afterwards to score the prediction. In a larger project, shuffle first and use a stratified split so each purpose is represented in train and test sets.

### The two values sent into training

For every training trace, `fit(train_traces)` creates exactly this pair:

```python
text = trace["user_prompt"] + " " + trace["agent_declared_intent"]
label = trace["gold_manifest"]["allowedPurposes"][0]
```

For the example above, the input text contains words such as `audit`, `financial`, and `ledger`; the label is `financial_payroll_auditing`. The first purpose only is used, so the current model is **single-label**, even if a gold manifest contains additional purposes.

### TF-IDF: turning words into numbers

`TfidfVectorizer` builds a vocabulary from the training text and turns each sentence into a numeric vector. Words that distinguish a topic receive more weight than words that appear in almost every example.

```text
Vocabulary: [ledger, patient, order]
"audit ledger"     -> [0.82, 0.00, 0.00]
"retrieve patient" -> [0.00, 0.91, 0.00]
```

The real vector contains all learned training tokens. A new trace uses the same learned vocabulary; it does not create new features while being predicted.

### Naive Bayes: choosing the purpose

`MultinomialNB` learns which weighted words are associated with each label. For a new TF-IDF vector, it scores every purpose seen in training and chooses the highest-scoring label. The word “naive” refers to its simplifying assumption that word features are independent once the purpose is known. It is fast and understandable, which is useful for this small dataset.

### What happens after classification

The classifier produces only `allowedPurposes`. The following fields come from deterministic inspection of the observed calls:

| Manifest field | How it is generated |
| --- | --- |
| `scope.tools`, `scope.resources`, `scope.actions` | Unique values actually observed. |
| `constraints.maxRecords` | Largest observed limit, rounded to a configured bucket. |
| `constraints.allowedCustomerIdPattern` | First matching known ID template; defaults to `.*` when unknown. |
| `constraints.allowPagination` | True when any call has `offset > 0`. |
| `dataHandling` | Resource-keyword rules, for example healthcare resources map to `CONFIDENTIAL_PHI`. |

This design makes the model easier to understand: text classification decides the business purpose, while the trace determines concrete permissions.

### Minimal experiment

```python
import json
from src.track1_manifest_inference.statistical_ml import StatisticalPatternModel

with open("data/eval_dataset/eval_traces_gold.json", encoding="utf-8") as file:
    traces = json.load(file)

model = StatisticalPatternModel()
model.fit(traces[:20])
manifest = model.predict_manifest(traces[20])
print(manifest["allowedPurposes"])
```

### Limitations to learn from

- The dataset is small, so a different split can change results.
- Confidence scores are fixed values; they are not calibrated probabilities from Naive Bayes.
- `.*` is broad for an unknown identifier. A production system should fail closed or require review instead.
- Add more traces, multi-label classification, stratified cross-validation, and calibrated probabilities to improve this baseline.
