# Primary Track — Pattern 1: Frequency-Threshold Baseline

📁 **Source Code**: [`src/track1_manifest_inference/baseline_frequency.py`](file:///c:/Users/chint/Music/Intent%20Agent/src/track1_manifest_inference/baseline_frequency.py)  
🏷️ **Class Name**: `FrequencyBaselineModel`  
📊 **Track Role**: Primary Track Baseline Candidate Model

---

## 🎯 Model Objective
Pattern 1 is the baseline heuristic model. It operates on the simple assumption that **an agent's future permissible scope should strictly mirror what it has already executed in its recent trace log**.

---

## ⚙️ How It Works (Step-by-Step Code Mechanics)

1. **Reading Trace History**: Extracts `tool_call_history` from the input trace dictionary.
2. **Item Frequency Tallying**: Maintains short-lived transient dictionaries (`tool_counts`, `res_counts`, `act_counts`) and increments counters for every observed tool, ARN, and action.
3. **Threshold Filtering ($N=1$)**: Includes tools, ARNs, and actions that appeared $\ge N$ times ($N=1$ by default).
4. **Parameter Limit Mining**: Tracks the peak integer `limit` parameter observed in the trace. If no limit is seen, defaults to $500$.
5. **Wildcard & Static Defaults**:
   - Defaults parameter regex constraint to `"allowedCustomerIdPattern": ".*"` (wildcard matching anything).
   - Defaults data security classification to `"maxClassification": "INTERNAL"`.
   - Emits a flat uncalibrated confidence score of $0.50$ ($50\%$) across all manifest fields.

---

## 💾 Internal State Stored Across Requests
- **Stateless Architecture**: Pattern 1 does **not** persist any database entries, trained weights, or vectorizers across requests.
- **Transient In-Memory State**: During `predict_manifest(trace)`, it uses temporary counters:
  - `tool_counts: Dict[str, int]`
  - `res_counts: Dict[str, int]`
  - `act_counts: Dict[str, int]`
  - `max_records_seen: int`
  - `has_pagination: bool`

---

## ❓ Frequently Asked Questions (Q&A)

### Q1: What is Pattern 1 officially named?
**Answer**: `FrequencyBaselineModel` (or *Baseline Frequency-Threshold Model*).

### Q2: What does Pattern 1 store internally across requests?
**Answer**: Nothing. Pattern 1 is completely stateless. It does not persist any trained weights or vectorizers across requests.

### Q3: How does Pattern 1 decide which manifest to create for a NEW input trace?
**Answer**: It scans the `tool_call_history` in the new trace, counts how many times each tool/resource/action was invoked, and includes any item with count $\ge 1$.

### Q4: When a user prompt arrives (e.g. "Look up support cases for C1000"), how does Pattern 1 know which tools/actions should be done?
**Answer**:
- **It DOES NOT read or parse the prompt text at all!** Pattern 1 completely ignores the user prompt's natural language text.
- **How it decides**: It relies **100% on what the agent has already executed in past `tool_call_history`**.
  - If the trace log shows the agent already called `crm_report_tool.read` in the past, Pattern 1 says: *"I see `crm_report_tool.read` in past calls, so I will add `crm_report_tool.read` to the manifest."*
  - If the agent has NOT YET executed `support_ticket_tool.lookup`, Pattern 1 **has no idea** that `support_ticket_tool.lookup` is needed, because it cannot read the user prompt to infer future intent.

### Q5: What happens when a NEW tool is needed for Step 2 of a task, but wasn't in the past trace?
**Answer**: **Pattern 1 excludes/denies it.** Because it cannot predict unobserved tools, it blocks Step 2 of the agent's job. This is the root cause of its **41.94% false denial under-permissioning error rate**.

### Q6: Why is Pattern 1 flawed for security governance?
**Answer**:
1. **Ignores Prompt Intent**: Cannot read text prompts to predict upcoming required tools.
2. **Severe Under-Permissioning (41.94%)**: Blocks legitimate multi-step agent workflows.
3. **Wildcard Security Hole (`.*`)**: Accepts any parameter value, allowing prompt injection attacks.
4. **Blind Data Classification**: Marks sensitive PHI or financial records as standard `"INTERNAL"` data.
5. **Meaningless Confidence ($0.50$)**: Emits flat $50\%$ confidence with high error ($\text{ECE} = 0.3875$).

---

## 📊 Benchmark Performance

| Metric | Score | Status |
| :--- | :---: | :---: |
| **Over-Permissioning Rate** (False Grants) | 0.00% | PASSED ($\le 5\%$) |
| **Under-Permissioning Rate** (False Denials) | 41.94% | **FAILED (Too High)** |
| **Macro Scope Recall** | 58.06% | **FAILED ($< 90\%$)** |
| **Constraint Exact Match** (`maxRecords`) | 2.50% | **FAILED ($< 80\%$)** |
| **ECE Calibration Error** | 0.3875 | **FAILED (High Error)** |
| **Inference Latency** | 0.01 ms | Fast |
| **Compute / Token Cost** | $0.00 | Free |

---

## Plain-language summary: Pattern 1

### How it works

Pattern 1 is a **counting rule**, not a trained AI model. It looks only at tool calls that already happened in one trace. If a tool, resource, or action appears at least once, it adds it to the manifest. It ignores the user prompt and the agent's declared intent.

Think of it as making a checklist from an agent's past actions:

```text
Observed: crm_report_tool.read was called twice
Output:   allow crm_report_tool.read
```

### Pros

- **Very simple:** no dataset, training, API key, or model download is required.
- **Very fast:** it mainly counts values in a list.
- **Easy to audit:** every granted tool comes directly from an observed call.
- **Low cost:** it has no model-compute cost.

### Cons

- **Cannot plan ahead:** it cannot add a legitimate next-step tool that has not yet been observed.
- **Does not understand language:** “retrieve a patient chart” and “read an EHR record” have no meaning to it.
- **Weak constraints:** unknown identifiers become `.*`, which is a broad wildcard.
- **Weak data policy:** it defaults to `INTERNAL`, so it can miss financial or health sensitivity.
- **No learned confidence:** its fixed 0.50 confidence does not explain how reliable a field really is.

### Best use

Use Pattern 1 only as a baseline for comparison, or as a simple audit of already-observed activity. Do not use it alone to grant access for multi-step workflows.
