"""
Model Comparison & Evaluation Benchmarking Runner Script.
Loads gold traces and seeded divergence set, evaluates models on held-out test split,
prints full comparison metrics, and outputs data/benchmark_results.json.
"""
import os
import json
import numpy as np
from typing import Dict, Any, List

from src.models.baseline_frequency import FrequencyBaselineModel
from src.models.statistical_ml import StatisticalPatternModel
from src.models.llm_hybrid import LLMHybridManifestModel
from src.divergence.intent_divergence import IntentDivergenceEngine
from src.evaluation.harness import EvaluationHarness

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data")
GOLD_PATH = os.path.join(DATA_DIR, "eval_dataset", "eval_traces_gold.json")
SEEDED_PATH = os.path.join(DATA_DIR, "eval_dataset", "seeded_divergence_set.json")
OUTPUT_BENCHMARK = os.path.join(DATA_DIR, "benchmark_results.json")

def run_benchmark():
    print("=" * 80)
    print(" INTENT MANIFEST INFERENCE & ACCURACY MODEL BENCHMARK HARNESS ")
    print("=" * 80)

    with open(GOLD_PATH, "r") as f:
        gold_traces = json.load(f)
    with open(SEEDED_PATH, "r") as f:
        seeded_cases = json.load(f)

    print(f"[*] Loaded {len(gold_traces)} gold activity traces.")
    print(f"[*] Loaded {len(seeded_cases)} seeded divergence cases.")

    train_traces = gold_traces[:20]
    test_traces = gold_traces[20:]
    print(f"[*] Train set: {len(train_traces)} traces | Held-out Test set: {len(test_traces)} traces.")

    baseline_model = FrequencyBaselineModel(threshold_n=1)
    model_stat_ml = StatisticalPatternModel()

    print("[*] Training Statistical ML Purpose Classifier on train set...")
    model_stat_ml.fit(train_traces)

    model_llm_hybrid = LLMHybridManifestModel()

    models = [baseline_model, model_stat_ml, model_llm_hybrid]
    harness = EvaluationHarness()

    results_primary = {}
    print("\n" + "-" * 80)
    print(" PRIMARY TRACK: MANIFEST INFERENCE BENCHMARK RESULTS (HELD-OUT TEST SET) ")
    print("-" * 80)

    for m in models:
        res = harness.evaluate_manifest_model(m, test_traces)
        results_primary[m.name] = res

    baseline_res = results_primary[baseline_model.name]
    
    header = f"{'Metric':<32} | {'Baseline (Freq)':<16} | {'Model 1 (StatML)':<16} | {'Model 2 (LLM-Hyb)':<16}"
    print(header)
    print("-" * len(header))

    metrics_to_show = [
        ("Over-Permissioning Rate (<=5%)", "over_permissioning_rate", True),
        ("Under-Permissioning Rate", "under_permissioning_rate", True),
        ("Macro Scope Recall (>=0.90)", "scope_recall_macro", False),
        ("Macro Scope Precision", "scope_precision_macro", False),
        ("Constraint Exact Match (>=0.80)", "constraint_exact_match", False),
        ("Constraint MAE (maxRecords)", "constraint_mae", True),
        ("Pattern Accuracy (Regex ID)", "pattern_accuracy", False),
        ("Purpose Classification F1", "purpose_f1", False),
        ("ECE Calibration Error (Lower)", "ece_calibration_error", True),
        ("Avg Latency (ms)", "avg_latency_ms", True),
        ("Cost per Manifest ($)", "cost_per_manifest_usd", True)
    ]

    for label, key, is_lower_better in metrics_to_show:
        b_val = baseline_res[key]
        m1_val = results_primary[model_stat_ml.name][key]
        m2_val = results_primary[model_llm_hybrid.name][key]

        b_str = f"{b_val:.4f}" if isinstance(b_val, float) else str(b_val)
        m1_str = f"{m1_val:.4f}" if isinstance(m1_val, float) else str(m1_val)
        m2_str = f"{m2_val:.4f}" if isinstance(m2_val, float) else str(m2_val)

        print(f"{label:<32} | {b_str:<16} | {m1_str:<16} | {m2_str:<16}")

    print("\n" + "-" * 80)
    print(" SECONDARY TRACK: INTENT DIVERGENCE DETECTION BENCHMARK ")
    print("-" * 80)
    
    div_engine = IntentDivergenceEngine()
    results_divergence = harness.evaluate_divergence_engine(div_engine, seeded_cases)

    print(f"Divergence Recall (Injection/Drift >=0.80) : {results_divergence['divergence_recall_injection'] * 100:.2f}% (Target: >=80%)")
    print(f"Divergence Precision                       : {results_divergence['divergence_precision_injection'] * 100:.2f}%")
    print(f"Divergence F1 Score                        : {results_divergence['divergence_f1_injection']:.4f}")
    print(f"Benign Paraphrase FP Rate (<=15%)          : {results_divergence['benign_paraphrase_fpr'] * 100:.2f}% (Target: <=15%)")
    print(f"Confusion Matrix (TP/FP/TN/FN)             : TP={results_divergence['true_positives']} | FP={results_divergence['false_positives']} | TN={results_divergence['true_negatives']} | FN={results_divergence['false_negatives']}")

    rec_model_res = results_primary[model_llm_hybrid.name]
    
    print("\n" + "=" * 80)
    print(" ACCEPTANCE CRITERIA AUDIT ")
    print("=" * 80)
    gate_over_perm = rec_model_res["over_permissioning_rate"] <= 0.05
    gate_recall = rec_model_res["scope_recall_macro"] >= 0.90
    gate_constraint = rec_model_res["constraint_exact_match"] >= 0.80
    gate_ece = rec_model_res["ece_calibration_error"] < baseline_res["ece_calibration_error"]
    gate_div_recall = results_divergence["divergence_recall_injection"] >= 0.80
    gate_div_fpr = results_divergence["benign_paraphrase_fpr"] <= 0.15

    print(f"1. Over-permissioning rate <= 5% (Hard Gate): {'PASSED' if gate_over_perm else 'FAILED'} ({rec_model_res['over_permissioning_rate']*100:.2f}%)")
    print(f"2. Scope recall >= 0.90 (Usability)          : {'PASSED' if gate_recall else 'FAILED'} ({rec_model_res['scope_recall_macro']*100:.2f}%)")
    print(f"3. Constraint exact match >= 0.80            : {'PASSED' if gate_constraint else 'FAILED'} ({rec_model_res['constraint_exact_match']*100:.2f}%)")
    print(f"4. Calibration ECE better than baseline      : {'PASSED' if gate_ece else 'FAILED'} (ECE={rec_model_res['ece_calibration_error']:.4f} vs Baseline ECE={baseline_res['ece_calibration_error']:.4f})")
    print(f"5. Divergence recall >= 0.80 on injection    : {'PASSED' if gate_div_recall else 'FAILED'} ({results_divergence['divergence_recall_injection']*100:.2f}%)")
    print(f"6. Benign paraphrase FPR <= 15%              : {'PASSED' if gate_div_fpr else 'FAILED'} ({results_divergence['benign_paraphrase_fpr']*100:.2f}%)")

    benchmark_payload = {
        "dataset_summary": {
            "total_gold_traces": len(gold_traces),
            "train_set_size": len(train_traces),
            "held_out_test_set_size": len(test_traces),
            "seeded_divergence_cases": len(seeded_cases)
        },
        "primary_track_models": results_primary,
        "secondary_track_divergence": results_divergence,
        "acceptance_criteria_audit": {
            "over_permissioning_gate_passed": gate_over_perm,
            "scope_recall_gate_passed": gate_recall,
            "constraint_exact_match_gate_passed": gate_constraint,
            "calibration_ece_gate_passed": gate_ece,
            "divergence_recall_gate_passed": gate_div_recall,
            "benign_fpr_gate_passed": gate_div_fpr
        }
    }

    with open(OUTPUT_BENCHMARK, "w") as f:
        json.dump(benchmark_payload, f, indent=2)

    print(f"\n[+] Successfully saved benchmark results to {OUTPUT_BENCHMARK}\n")

if __name__ == "__main__":
    run_benchmark()
