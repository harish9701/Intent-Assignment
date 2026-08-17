# Pattern 3: Configurable Local / Frontier Manifest Extractor

**Source:** `src/track1_manifest_inference/llm_hybrid.py`
**Class:** `LLMHybridManifestModel`

## Is Pattern 3 the same as Pattern 4?

No. Before this update, Pattern 3 was deterministic policy code and Pattern 4 was a direct local-Ollama caller. Pattern 4 used Pattern 3 whenever Ollama was unavailable, so the **fallback output was the same** but the normal execution path was not.

Pattern 3 is now the single supported routing interface:

| `inference_provider` value | What runs | Needs credentials/service? | If unavailable |
| --- | --- | --- | --- |
| omitted or `"deterministic"` | Local `DOMAIN_ENVELOPES` policy rules | No | Always works locally |
| `"local"` | An Ollama model, default `llama3` | Local Ollama service and model | Deterministic Pattern 3 rules |
| `"frontier"` | An OpenAI frontier model, default `gpt-5.6` | `OPENAI_API_KEY` and network access | Deterministic Pattern 3 rules |

The provider is explicit configuration in the trace, not something inferred from the user's prompt. A prompt must never be able to switch a governance system from local to frontier processing.

## Why use different providers?

All three paths create the same manifest schema, but they obtain the proposed fields differently:

- **Deterministic:** selects a pre-approved domain envelope. It is fastest and fully local, but only covers the domains configured in source code.
- **Local:** lets an installed open-weights model interpret novel language while keeping the trace on the local machine. Its JSON is parsed before being accepted.
- **Frontier:** sends the trace to a stronger hosted model through the OpenAI Responses API. It can be more flexible with language, but needs an API key, network access, and has an external-data boundary.

The deterministic fallback exists so the system returns a reproducible manifest even when a configured model is offline, misconfigured, or returns invalid JSON.

## Plain-language summary: Pattern 3

### How it works

Pattern 3 is a **router**. A trusted configuration field chooses one of three ways to propose the same manifest: deterministic rules from approved domain envelopes, a local Ollama model, or a hosted frontier model. If a model path fails, it uses deterministic rules rather than failing open. The user prompt helps describe the task, but it cannot choose the provider.

### Pros

- **Flexible deployment:** one interface supports offline, local-model, and hosted-model use.
- **Reliable fallback:** deterministic mode still works during network, credential, or local-service failures.
- **Consistent output:** every provider returns the same manifest structure.
- **Privacy choice:** local/deterministic modes can avoid sending traces to an external service.
- **Broader language handling when needed:** local or frontier models can interpret wording outside the hard-coded domain keywords.

### Cons

- **More moving parts:** provider configuration, model installation, credentials, and fallback behaviour must be understood.
- **Model output is not automatically policy-safe:** valid JSON can still contain inappropriate permissions, so production systems must validate it against policy.
- **Local models need resources:** Ollama requires a running service and enough memory/CPU or GPU for its model.
- **Frontier models have external dependency:** they require network access, an API key, and an explicit data-sharing decision.
- **Deterministic coverage is finite:** it only understands the domain envelopes defined in code.

### Best use

Use deterministic mode for reproducible, offline governance. Use local mode where privacy requires on-device processing but language flexibility helps. Use frontier mode only when its stronger language interpretation is needed and external processing is approved.

## Choose a provider

### Fully local, no model service

```python
trace["inference_provider"] = "deterministic"
manifest = LLMHybridManifestModel().predict_manifest(trace)
```

### Local Ollama model

```python
trace["inference_provider"] = "local"
trace["local_model"] = "llama3"  # Optional; llama3 is already the default.
manifest = LLMHybridManifestModel().predict_manifest(trace)
```

### OpenAI frontier model

Set `OPENAI_API_KEY` in your environment, then choose the provider and, optionally, a model:

```python
trace["inference_provider"] = "frontier"
trace["frontier_model"] = "gpt-5.6"
manifest = LLMHybridManifestModel().predict_manifest(trace)
```

The frontier path uses structured JSON output through the OpenAI Responses API. It requires the `openai` Python package, which is listed in this project's dependencies.

## Deterministic path, step by step

1. Read `domain_category` if the trace provides a supported value.
2. Otherwise inspect the prompt, declared intent, and observed tool names for domain terms.
3. Read the matching approved policy from `DOMAIN_ENVELOPES`.
4. Produce the allowed tools, resources, actions, record limit, identifier regex, data classification, and export policy from that policy.
5. Set `allowPagination` from observed calls with `offset > 0`.

The deterministic path does not train on the dataset and does not call an LLM.

## Safety note

Model-generated JSON is a proposal, not proof of authorization. The model path currently validates that the required manifest shape exists, and it falls back if it cannot parse a valid result. A production deployment should additionally enforce every returned field against an approved policy registry before granting access.
