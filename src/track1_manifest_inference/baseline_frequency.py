"""
Primary Track 1: Baseline Frequency-Threshold Heuristic Model for Intent Manifest Inference.
Rules:
- Includes tools, resources, and actions observed >= N times in trace history.
- Sets standard default fallback constraints and flat uncalibrated confidence (0.50).
"""
from typing import Dict, Any, List

class FrequencyBaselineModel:
    def __init__(self, threshold_n: int = 1):
        self.name = f"Baseline_Frequency_N{threshold_n}"
        self.threshold_n = threshold_n

    def predict_manifest(self, trace: Dict[str, Any]) -> Dict[str, Any]:
        tool_calls = trace.get("tool_call_history", [])
        
        tool_counts: Dict[str, int] = {}
        res_counts: Dict[str, int] = {}
        act_counts: Dict[str, int] = {}
        max_records_seen = 0
        has_pagination = False
        
        for call in tool_calls:
            t_name = call.get("tool_name")
            r_arn = call.get("resource_arn")
            action = call.get("action")
            
            if t_name:
                tool_counts[t_name] = tool_counts.get(t_name, 0) + 1
            if r_arn:
                res_counts[r_arn] = res_counts.get(r_arn, 0) + 1
            if action:
                act_counts[action] = act_counts.get(action, 0) + 1
                
            params = call.get("parameters", {})
            if "limit" in params:
                max_records_seen = max(max_records_seen, int(params["limit"]))
            if params.get("offset", 0) > 0:
                has_pagination = True

        sel_tools = sorted([t for t, count in tool_counts.items() if count >= self.threshold_n])
        sel_resources = sorted([r for r, count in res_counts.items() if count >= self.threshold_n])
        sel_actions = sorted([a for a, count in act_counts.items() if count >= self.threshold_n])
        
        purpose_guess = trace.get("domain_category", "general_query")
        
        confidences = {
            "scope.tools": 0.50,
            "scope.resources": 0.50,
            "scope.actions": 0.50,
            "constraints.maxRecords": 0.50,
            "constraints.allowedCustomerIdPattern": 0.50,
            "constraints.allowPagination": 0.50,
            "allowedPurposes": 0.50,
            "dataHandling.maxClassification": 0.50,
            "dataHandling.allowExport": 0.50
        }
        
        proposed_manifest = {
            "intentManifestId": f"im_baseline_{trace.get('trace_id', '000')}",
            "version": "1.0",
            "agentId": trace.get("agent_id", "agt_unknown"),
            "allowedPurposes": [purpose_guess],
            "scope": {
                "tools": sel_tools,
                "resources": sel_resources,
                "actions": sel_actions
            },
            "constraints": {
                "maxRecords": max_records_seen if max_records_seen > 0 else 500,
                "allowedCustomerIdPattern": ".*",
                "allowPagination": has_pagination
            },
            "dataHandling": {
                "maxClassification": "INTERNAL",
                "allowExport": False
            },
            "validity": {
                "validFrom": "2026-01-01T00:00:00Z",
                "validTo": "2026-12-31T23:59:59Z"
            },
            "confidence_scores": confidences
        }
        return proposed_manifest
