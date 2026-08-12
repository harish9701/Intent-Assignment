# Primary Track — Pattern 2: Classical Statistical ML & Pattern Miner

📁 **Source Code**: [`src/track1_manifest_inference/statistical_ml.py`](file:///c:/Users/chint/Music/Intent%20Agent/src/track1_manifest_inference/statistical_ml.py)  
📦 **ML Classifier Variants**: [`src/track1_manifest_inference/statistical_ml_variants.py`](file:///c:/Users/chint/Music/Intent%20Agent/src/track1_manifest_inference/statistical_ml_variants.py)  
🏷️ **Class Name**: `StatisticalPatternModel`  
📊 **Track Role**: Primary Track Classical ML Candidate Model

---

## 🎯 Model Objective
Pattern 2 investigates classical machine learning and deterministic pattern induction techniques:
> *"Can we combine TF-IDF feature extraction, probabilistic text classification, discrete limit quantiles, and regular expression pattern induction to infer authoritative Intent Manifests?"*

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
