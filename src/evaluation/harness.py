"""
Evaluation Harness for Intent Manifest Inference & Divergence Models.
Calculates all primary & secondary metrics against expert gold ground truth.
"""
import time
import numpy as np
from typing import Dict, Any, List, Tuple

class EvaluationHarness:
    def __init__(self):
        pass

    def evaluate_manifest_model(self, model: Any, eval_traces: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Runs evaluation for a primary manifest inference model over eval_traces dataset.
        """
        total_traces = len(eval_traces)
        if total_traces == 0:
            return {}

        tool_precisions, tool_recalls = [], []
        res_precisions, res_recalls = [], []
        act_precisions, act_recalls = [], []

        total_proposed_items = 0
        total_false_grants = 0
        total_gold_items = 0
        total_false_denials = 0

        max_records_exact_matches = 0
        max_records_maes = []
        pattern_exact_matches = 0
        pagination_exact_matches = 0

        purpose_exact_matches = 0
        purpose_f1_scores = []

        conf_correctness_pairs = []
        latencies_ms = []

        for trace in eval_traces:
            gold = trace["gold_manifest"]
            
            start_time = time.perf_counter()
            pred = model.predict_manifest(trace)
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            latencies_ms.append(elapsed_ms)

            gold_tools = set(gold["scope"]["tools"])
            pred_tools = set(pred["scope"]["tools"])
            tp_t = len(gold_tools.intersection(pred_tools))
            tool_precisions.append(tp_t / len(pred_tools) if pred_tools else 1.0)
            tool_recalls.append(tp_t / len(gold_tools) if gold_tools else 1.0)

            gold_res = set(gold["scope"]["resources"])
            pred_res = set(pred["scope"]["resources"])
            tp_r = len(gold_res.intersection(pred_res))
            res_precisions.append(tp_r / len(pred_res) if pred_res else 1.0)
            res_recalls.append(tp_r / len(gold_res) if gold_res else 1.0)

            gold_act = set(gold["scope"]["actions"])
            pred_act = set(pred["scope"]["actions"])
            tp_a = len(gold_act.intersection(pred_act))
            act_precisions.append(tp_a / len(pred_act) if pred_act else 1.0)
            act_recalls.append(tp_a / len(gold_act) if gold_act else 1.0)

            all_gold = gold_tools.union(gold_res).union(gold_act)
            all_pred = pred_tools.union(pred_res).union(pred_act)
            
            fg = len(all_pred.difference(all_gold))
            fd = len(all_gold.difference(all_pred))

            total_proposed_items += len(all_pred)
            total_false_grants += fg
            total_gold_items += len(all_gold)
            total_false_denials += fd

            gold_max_rec = gold["constraints"]["maxRecords"]
            pred_max_rec = pred["constraints"]["maxRecords"]
            if pred_max_rec == gold_max_rec:
                max_records_exact_matches += 1
            max_records_maes.append(abs(pred_max_rec - gold_max_rec))

            if pred["constraints"]["allowedCustomerIdPattern"] == gold["constraints"]["allowedCustomerIdPattern"]:
                pattern_exact_matches += 1

            if pred["constraints"]["allowPagination"] == gold["constraints"]["allowPagination"]:
                pagination_exact_matches += 1

            gold_purposes = set(gold["allowedPurposes"])
            pred_purposes = set(pred["allowedPurposes"])
            if gold_purposes.intersection(pred_purposes):
                purpose_exact_matches += 1
                
            p_tp = len(gold_purposes.intersection(pred_purposes))
            p_prec = p_tp / len(pred_purposes) if pred_purposes else 0
            p_rec = p_tp / len(gold_purposes) if gold_purposes else 0
            p_f1 = (2 * p_prec * p_rec) / (p_prec + p_rec) if (p_prec + p_rec) > 0 else 0.0
            purpose_f1_scores.append(p_f1)

            conf_dict = pred.get("confidence_scores", {})
            is_tools_correct = (pred_tools == gold_tools)
            conf_correctness_pairs.append((conf_dict.get("scope.tools", 0.5), float(is_tools_correct)))
            
            is_rec_correct = (pred_max_rec == gold_max_rec)
            conf_correctness_pairs.append((conf_dict.get("constraints.maxRecords", 0.5), float(is_rec_correct)))

        macro_scope_precision = float(np.mean([np.mean(tool_precisions), np.mean(res_precisions), np.mean(act_precisions)]))
        macro_scope_recall = float(np.mean([np.mean(tool_recalls), np.mean(res_recalls), np.mean(act_recalls)]))

        over_permissioning_rate = float(total_false_grants / total_proposed_items) if total_proposed_items > 0 else 0.0
        under_permissioning_rate = float(total_false_denials / total_gold_items) if total_gold_items > 0 else 0.0

        constraint_exact_match = float(max_records_exact_matches / total_traces)
        constraint_mae = float(np.mean(max_records_maes))
        pattern_accuracy = float(pattern_exact_matches / total_traces)
        pagination_accuracy = float(pagination_exact_matches / total_traces)

        purpose_accuracy = float(purpose_exact_matches / total_traces)
        purpose_macro_f1 = float(np.mean(purpose_f1_scores))

        ece, reliability_curve = self._compute_ece(conf_correctness_pairs, num_bins=10)
        cost_per_manifest = 0.0004 if "LLM" in getattr(model, "name", "") else 0.0000

        return {
            "model_name": getattr(model, "name", "Model"),
            "scope_precision_macro": round(macro_scope_precision, 4),
            "scope_recall_macro": round(macro_scope_recall, 4),
            "tool_precision": round(float(np.mean(tool_precisions)), 4),
            "tool_recall": round(float(np.mean(tool_recalls)), 4),
            "resource_precision": round(float(np.mean(res_precisions)), 4),
            "resource_recall": round(float(np.mean(res_recalls)), 4),
            "action_precision": round(float(np.mean(act_precisions)), 4),
            "action_recall": round(float(np.mean(act_recalls)), 4),
            "over_permissioning_rate": round(over_permissioning_rate, 4),
            "under_permissioning_rate": round(under_permissioning_rate, 4),
            "constraint_exact_match": round(constraint_exact_match, 4),
            "constraint_mae": round(constraint_mae, 2),
            "pattern_accuracy": round(pattern_accuracy, 4),
            "pagination_accuracy": round(pagination_accuracy, 4),
            "purpose_accuracy": round(purpose_accuracy, 4),
            "purpose_f1": round(purpose_macro_f1, 4),
            "ece_calibration_error": round(ece, 4),
            "reliability_curve": reliability_curve,
            "avg_latency_ms": round(float(np.mean(latencies_ms)), 2),
            "cost_per_manifest_usd": cost_per_manifest
        }

    def evaluate_divergence_engine(self, engine: Any, seeded_cases: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Runs evaluation for Secondary Track Intent Divergence Engine over seeded cases dataset.
        """
        tp, fp, tn, fn = 0, 0, 0, 0
        benign_total = 0
        injection_total = 0

        for case in seeded_cases:
            gold_divergent = case["is_divergent"]
            res = engine.analyze_triplet(case)
            pred_divergent = res["is_divergent"]

            if case["category"] == "benign_paraphrase":
                benign_total += 1
                if pred_divergent:
                    fp += 1
                else:
                    tn += 1
            else:
                injection_total += 1
                if pred_divergent:
                    tp += 1
                else:
                    fn += 1

        recall_injection = float(tp / injection_total) if injection_total > 0 else 0.0
        precision_injection = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        benign_fpr = float(fp / benign_total) if benign_total > 0 else 0.0
        f1_injection = (2 * precision_injection * recall_injection) / (precision_injection + recall_injection) if (precision_injection + recall_injection) > 0 else 0.0

        return {
            "divergence_recall_injection": round(recall_injection, 4),
            "divergence_precision_injection": round(precision_injection, 4),
            "divergence_f1_injection": round(f1_injection, 4),
            "benign_paraphrase_fpr": round(benign_fpr, 4),
            "total_seeded_cases": len(seeded_cases),
            "benign_cases_count": benign_total,
            "injection_drift_cases_count": injection_total,
            "true_positives": tp,
            "false_positives": fp,
            "true_negatives": tn,
            "false_negatives": fn
        }

    def _compute_ece(self, conf_correctness_pairs: List[Tuple[float, float]], num_bins: int = 10) -> Tuple[float, List[Dict[str, Any]]]:
        if not conf_correctness_pairs:
            return 0.0, []

        bin_boundaries = np.linspace(0, 1, num_bins + 1)
        ece = 0.0
        total_samples = len(conf_correctness_pairs)
        reliability_curve = []

        confs = np.array([p[0] for p in conf_correctness_pairs])
        corrects = np.array([p[1] for p in conf_correctness_pairs])

        for i in range(num_bins):
            bin_lower = bin_boundaries[i]
            bin_upper = bin_boundaries[i + 1]

            in_bin = (confs >= bin_lower) & (confs < bin_upper) if i < num_bins - 1 else (confs >= bin_lower) & (confs <= bin_upper)
            bin_size = np.sum(in_bin)

            if bin_size > 0:
                avg_conf = float(np.mean(confs[in_bin]))
                avg_acc = float(np.mean(corrects[in_bin]))
                ece += (bin_size / total_samples) * abs(avg_acc - avg_conf)
                reliability_curve.append({
                    "bin_range": f"{bin_lower:.1f}-{bin_upper:.1f}",
                    "avg_confidence": round(avg_conf, 3),
                    "avg_accuracy": round(avg_acc, 3),
                    "sample_count": int(bin_size)
                })
            else:
                reliability_curve.append({
                    "bin_range": f"{bin_lower:.1f}-{bin_upper:.1f}",
                    "avg_confidence": round((bin_lower + bin_upper) / 2, 3),
                    "avg_accuracy": 0.0,
                    "sample_count": 0
                })

        return float(ece), reliability_curve
