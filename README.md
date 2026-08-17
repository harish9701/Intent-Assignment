# Intent Manifest Inference — Local Pattern 3 Guide

The recommended manifest-inference path is **Pattern 3**, `LLMHybridManifestModel`. It can run deterministically on the local machine, through a local Ollama model, or through a configured frontier model.

With no provider setting, Pattern 3 uses deterministic schema-bounded extraction: it identifies an intent domain, selects that domain's approved envelope, and returns an Intent Manifest. This default requires no API key, Ollama service, downloaded model, or training.

## Run locally

```bash
pip install -r requirements.txt
```

Then use the Pattern 3 example below in a Python file or interpreter. The separate benchmark command evaluates all research candidates; it is not needed to run Pattern 3 locally.

## Use Pattern 3 directly

```python
from src.track1_manifest_inference.llm_hybrid import LLMHybridManifestModel

model = LLMHybridManifestModel()
manifest = model.predict_manifest(trace)
print(manifest)
```

`trace` should contain a user prompt, declared agent intent, and observed `tool_call_history`. If `domain_category` is available and is one of the supported domains, Pattern 3 uses it; otherwise it infers the domain from the prompt, declared intent, and tool names.

## How Pattern 3 works

1. Determine the domain, for example healthcare, finance, security, ecommerce, DevOps, or customer support.
2. Read that domain's safe envelope from `DOMAIN_ENVELOPES` in `src/track1_manifest_inference/llm_hybrid.py`.
3. Generate a manifest using only that envelope's tools, resources, actions, record limit, identifier pattern, and data-handling rules.
4. Read pagination from the observed calls.

This makes Pattern 3 reproducible and suitable for local demonstrations. Its domain envelopes are policy data in source code, so they should be reviewed and maintained as policies change.

## Pattern 2 learning guide

Pattern 2 is the classical supervised-ML alternative. Its beginner-focused dataset and training explanation is in [its own Pattern 2 guide](docs/01_PRIMARY_TRACK_MANIFEST_INFERENCE/PATTERN_2_STATISTICAL_ML.md).

## Project layout

```text
src/track1_manifest_inference/
  statistical_ml.py       # Pattern 2: TF-IDF + Naive Bayes classifier
  llm_hybrid.py           # Pattern 3: deterministic, local, or frontier router
data/eval_dataset/
  eval_traces_gold.json   # labelled traces for training and evaluation
docs/01_PRIMARY_TRACK_MANIFEST_INFERENCE/
  PATTERN_2_STATISTICAL_ML.md # beginner guide to Pattern 2 and the dataset
```

## Additional documentation

- [Primary-track overview](docs/01_PRIMARY_TRACK_MANIFEST_INFERENCE/01_PRIMARY_TRACK_OVERVIEW.md)
- [Pattern 2 technical reference](docs/01_PRIMARY_TRACK_MANIFEST_INFERENCE/PATTERN_2_STATISTICAL_ML.md)
- [Pattern 3 technical reference](docs/01_PRIMARY_TRACK_MANIFEST_INFERENCE/PATTERN_3_HYBRID_LLM.md)
