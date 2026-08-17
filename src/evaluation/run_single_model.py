"""
Single Model Execution & Evaluation CLI Runner.
Runs a specific candidate model on train/test traces loaded from separate dataset files.

Usage:
    python -m src.evaluation.run_single_model --model baseline --single-trace --trace-id trace_043
    python -m src.evaluation.run_single_model --model statml --single-trace --trace-index 5
    python -m src.evaluation.run_single_model --model hybrid
    python -m src.evaluation.run_single_model --model authbench
"""
import os
import sys
import json
import argparse
from typing import Dict, Any, List, Optional

from src.track1_manifest_inference.baseline_frequency import FrequencyBaselineModel
from src.track1_manifest_inference.statistical_ml import StatisticalPatternModel
from src.track1_manifest_inference.llm_hybrid import LLMHybridManifestModel
from src.track1_manifest_inference.ollama_llm import OllamaLLMManifestModel
from src.track1_manifest_inference.authbench_sufficiency_tightness import AuthBenchInspiredManifestModel
from src.evaluation.harness import EvaluationHarness

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data")
EVAL_DATASET_DIR = os.path.join(DATA_DIR, "eval_dataset")
GOLD_PATH = os.path.join(EVAL_DATASET_DIR, "eval_traces_gold.json")
TRAIN_PATH = os.path.join(EVAL_DATASET_DIR, "train_traces.json")
TEST_PATH = os.path.join(EVAL_DATASET_DIR, "test_traces.json")

def load_train_test_splits() -> (List[Dict[str, Any]], List[Dict[str, Any]]):
    """Loads train_traces.json and test_traces.json, auto-creating them from gold set if needed."""
    if os.path.exists(TRAIN_PATH) and os.path.exists(TEST_PATH):
        with open(TRAIN_PATH, "r") as f:
            train_traces = json.load(f)
        with open(TEST_PATH, "r") as f:
            test_traces = json.load(f)
        return train_traces, test_traces

    if not os.path.exists(GOLD_PATH):
        print(f"[Error] Gold evaluation dataset not found at {GOLD_PATH}")
        sys.exit(1)

    print(f"[*] Splitting {GOLD_PATH} into train_traces.json (20) and test_traces.json (40)...")
    with open(GOLD_PATH, "r") as f:
        gold_traces = json.load(f)

    train_traces = gold_traces[:20]
    test_traces = gold_traces[20:]

    os.makedirs(EVAL_DATASET_DIR, exist_ok=True)
    with open(TRAIN_PATH, "w") as f:
        json.dump(train_traces, f, indent=2)
    with open(TEST_PATH, "w") as f:
        json.dump(test_traces, f, indent=2)

    print(f"[+] Saved train traces to: {TRAIN_PATH}")
    print(f"[+] Saved test traces to:  {TEST_PATH}")
    return train_traces, test_traces

def select_test_trace(test_traces: List[Dict[str, Any]], trace_index: Optional[int] = None, trace_id: Optional[str] = None) -> Dict[str, Any]:
    """Selects a test trace by trace_id or trace_index with fallback bounds protection."""
    if not test_traces:
        raise ValueError("Test traces dataset is empty.")

    available_ids = [t.get("trace_id", f"idx_{i}") for i, t in enumerate(test_traces)]

    # 1. Selection by trace_id
    if trace_id:
        for t in test_traces:
            if t.get("trace_id") == trace_id:
                print(f"[+] Matched requested Trace ID: {trace_id}")
                return t
        print(f"[Warning] Trace ID '{trace_id}' not found in test set.")
        print(f"    Available test Trace IDs: {available_ids[:5]} ... {available_ids[-5:]}")

    # 2. Selection by trace_index with bounds check
    if trace_index is not None:
        if 0 <= trace_index < len(test_traces):
            print(f"[+] Selected test trace at index {trace_index} (ID: {test_traces[trace_index].get('trace_id')})")
            return test_traces[trace_index]
        else:
            print(f"[Warning] Index {trace_index} out of bounds for test set size {len(test_traces)} (valid indices: 0..{len(test_traces)-1}).")

    # 3. Fallback default: first test trace
    fallback_trace = test_traces[0]
    print(f"[+] Fallback: Defaulting to first test trace (Index 0, ID: {fallback_trace.get('trace_id')})")
    return fallback_trace

def main():
    parser = argparse.ArgumentParser(description="Run an individual candidate model on Track 1 evaluation dataset.")
    parser.add_argument(
        "--model",
        type=str,
        default="baseline",
        choices=["baseline", "statml", "hybrid", "ollama", "authbench"],
        help="Which model to run: baseline | statml | hybrid | ollama | authbench"
    )
    parser.add_argument(
        "--single-trace",
        action="store_true",
        help="If set, outputs the predicted JSON manifest for a single test trace."
    )
    parser.add_argument(
        "--trace-index",
        type=int,
        default=None,
        help="0-based index of the test trace to predict (e.g. 0, 5, 20)."
    )
    parser.add_argument(
        "--trace-id",
        type=str,
        default=None,
        help="Exact Trace ID of the test trace to predict (e.g. trace_021, trace_043)."
    )
    args = parser.parse_args()

    train_traces, test_traces = load_train_test_splits()
    print(f"[*] Dataset loaded: {len(train_traces)} train traces | {len(test_traces)} test traces.")

    # Instantiate chosen model
    if args.model == "baseline":
        model = FrequencyBaselineModel(threshold_n=1)
    elif args.model == "statml":
        model = StatisticalPatternModel()
        print(f"[*] Training Statistical ML model on {len(train_traces)} train traces...")
        model.fit(train_traces)
    elif args.model == "hybrid":
        model = LLMHybridManifestModel()
    elif args.model == "ollama":
        model = OllamaLLMManifestModel()
    elif args.model == "authbench":
        model = AuthBenchInspiredManifestModel()
    else:
        raise ValueError(f"Unknown model choice: {args.model}")

    print(f"\n================================================================================")
    print(f" EXECUTING MODEL: {model.name}")
    print(f"================================================================================\n")

    if args.single_trace or args.trace_index is not None or args.trace_id is not None:
        sample_trace = select_test_trace(test_traces, trace_index=args.trace_index, trace_id=args.trace_id)
        print(f"\n--- INPUT TRACE DETAILS ---")
        print(f"Trace ID              : {sample_trace.get('trace_id')}")
        print(f"Agent ID              : {sample_trace.get('agent_id')}")
        print(f"Domain Category       : {sample_trace.get('domain_category')}")
        print(f"User Prompt           : {sample_trace.get('user_prompt')}")
        print(f"Agent Declared Intent : {sample_trace.get('agent_declared_intent')}")
        print(f"Tool Calls Count      : {len(sample_trace.get('tool_call_history', []))}")

        manifest = model.predict_manifest(sample_trace)
        print("\n--- PREDICTED INTENT MANIFEST JSON ---")
        print(json.dumps(manifest, indent=2))
    else:
        harness = EvaluationHarness()
        print(f"[*] Running full evaluation harness over {len(test_traces)} held-out test traces...\n")
        results = harness.evaluate_manifest_model(model, test_traces)

        print(f"{'Metric Name':<38} | {'Value':<15}")
        print("-" * 58)
        for k, v in results.items():
            val_str = f"{v:.4f}" if isinstance(v, float) else str(v)
            print(f"{k:<38} | {val_str:<15}")
        print("\n[+] Evaluation complete.\n")

if __name__ == "__main__":
    main()
