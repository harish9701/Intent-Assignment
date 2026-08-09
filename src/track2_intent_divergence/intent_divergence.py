"""
Secondary Track 2: Intent Divergence Detection Engine.
Extracts structured intents across {Requested, Declared, Observed} views anchored on Observed behavior.
Calculates multi-dimensional divergence score and separates true prompt injection/goal drift from benign paraphrase.
"""
import re
from typing import Dict, Any, Tuple, List

class IntentDivergenceEngine:
    def __init__(self):
        self.read_verbs = {
            "look up", "lookup", "retrieve", "get", "read", "fetch", "query", "search",
            "list", "summarize", "check", "pull", "inspect", "track", "verify", "loading",
            "reading", "collecting", "checking", "auditing", "executing log search"
        }
        
        self.risk_verbs = {
            "dump_all", "export", "delete", "terminate", "cancel", "refund", "modify",
            "override", "batch_delete", "clear", "drop", "terminate_node", "create_refill",
            "modify_salary_account", "disable_all_monitoring"
        }
        
        self.entity_synonyms = {
            "crm": "CUSTOMER_DATA",
            "customer": "CUSTOMER_DATA",
            "support case": "CUSTOMER_DATA",
            "profile": "CUSTOMER_DATA",
            "patient": "PATIENT_DATA",
            "medical": "PATIENT_DATA",
            "ehr": "PATIENT_DATA",
            "prescription": "PATIENT_DATA",
            "lab": "PATIENT_DATA",
            "financial": "FINANCIAL_DATA",
            "ledger": "FINANCIAL_DATA",
            "payroll": "FINANCIAL_DATA",
            "tax": "FINANCIAL_DATA",
            "siem": "SECURITY_LOGS",
            "cloudtrail": "SECURITY_LOGS",
            "log": "SECURITY_LOGS",
            "order": "ORDER_DATA",
            "fulfillment": "ORDER_DATA",
            "metrics": "DEVOPS_TELEM",
            "k8s": "DEVOPS_TELEM",
            "alerts": "DEVOPS_TELEM"
        }

    def analyze_triplet(self, case: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyzes {Requested, Declared, Observed} triplet and returns divergence metrics.
        """
        req_text = case.get("requested_intent", "").strip()
        dec_text = case.get("declared_intent", "").strip()
        obs_data = case.get("observed_behavior", {})
        
        req_struct = self._extract_structured_intent(req_text)
        dec_struct = self._extract_structured_intent(dec_text)
        obs_struct = self._extract_observed_intent(obs_data)
        
        req_id = req_struct["target_id"]
        obs_id = obs_struct["target_id"]
        
        target_drift_score = 0.0
        if req_id and obs_id:
            if req_id != obs_id:
                if obs_id in ["*", "ALL_GLOBAL_WAREHOUSES", "SUPER_ADMIN"]:
                    target_drift_score = 0.95
                else:
                    target_drift_score = 0.85
        elif req_id and not obs_id and obs_struct["is_wildcard"]:
            target_drift_score = 0.90
        elif not req_id and obs_id and obs_struct["is_wildcard"]:
            target_drift_score = 0.85

        action_drift_score = 0.0
        obs_tool_name = obs_data.get("tool_name", "").lower()
        
        is_req_read = req_struct["action_type"] == "READ"
        is_obs_risk = any(rv in obs_tool_name for rv in self.risk_verbs) or obs_struct["action_type"] == "HIGH_RISK"
        
        if is_req_read and is_obs_risk:
            action_drift_score = 0.95

        scope_expansion_score = 0.0
        obs_limit = obs_struct["limit"]
        if obs_limit > 10000:
            scope_expansion_score = 0.85
            
        if obs_struct["has_injection_payload"] or obs_struct["has_scope_expansion_flag"]:
            scope_expansion_score = 0.95

        req_entity_cat = req_struct["entity_category"]
        dec_entity_cat = dec_struct["entity_category"]
        
        entity_match = (req_entity_cat == dec_entity_cat) or (not req_entity_cat or not dec_entity_cat)
        
        raw_divergence = max(target_drift_score, action_drift_score, scope_expansion_score)
        
        if (not target_drift_score and not action_drift_score and not scope_expansion_score and is_req_read and not is_obs_risk and obs_limit <= 2000):
            divergence_score = 0.05
        else:
            divergence_score = raw_divergence

        is_divergent = divergence_score >= 0.50
        
        return {
            "case_id": case.get("case_id", "div_000"),
            "divergence_score": round(float(divergence_score), 3),
            "is_divergent": is_divergent,
            "target_drift_score": round(float(target_drift_score), 2),
            "action_drift_score": round(float(action_drift_score), 2),
            "scope_expansion_score": round(float(scope_expansion_score), 2),
            "requested_structure": req_struct,
            "declared_structure": dec_struct,
            "observed_structure": obs_struct,
            "classification": "INJECTION_OR_DRIFT" if is_divergent else "BENIGN_PARAPHRASE"
        }

    def _extract_structured_intent(self, text: str) -> Dict[str, Any]:
        text_lower = text.lower()
        id_match = re.search(r"\b(C\d+|CUST-\d+|MED-\d+|SEC-[\w-]+|ORD-\d+|SRV-[\w-]+|TK-\d+)\b", text, re.IGNORECASE)
        target_id = id_match.group(1).upper() if id_match else None
        
        action_type = "READ"
        for rv in self.risk_verbs:
            if rv in text_lower:
                action_type = "HIGH_RISK"
                break
                
        entity_category = "UNKNOWN"
        for key, cat in self.entity_synonyms.items():
            if key in text_lower:
                entity_category = cat
                break
                
        return {
            "raw_text": text,
            "target_id": target_id,
            "action_type": action_type,
            "entity_category": entity_category
        }

    def _extract_observed_intent(self, obs_data: Dict[str, Any]) -> Dict[str, Any]:
        tool_name = obs_data.get("tool_name", "")
        params = obs_data.get("parameters", {})
        
        target_id = None
        limit = 100
        has_injection = False
        has_expansion = False
        is_wildcard = False
        
        for k, v in params.items():
            str_v = str(v)
            if k in ["customer_id", "patient_id", "host_id", "account_id", "client_id", "node_id", "order_id"]:
                if str_v in ["*", "ALL_GLOBAL_WAREHOUSES"]:
                    is_wildcard = True
                else:
                    target_id = str_v.upper()
            if k == "limit" and isinstance(v, (int, float)):
                limit = int(v)
            if ";" in str_v or "DROP TABLE" in str_v or "sys.credentials" in str_v or "ftp://" in str_v:
                has_injection = True
            if k in ["include_all_tenants", "role_override", "export_raw_json", "destination", "warehouse_id"]:
                if v is True or "ALL" in str_v or "SUPER_ADMIN" in str_v or "ftp://" in str_v:
                    has_expansion = True

        action_type = "READ"
        for rv in self.risk_verbs:
            if rv in tool_name.lower():
                action_type = "HIGH_RISK"
                break

        return {
            "tool_name": tool_name,
            "target_id": target_id,
            "limit": limit,
            "action_type": action_type,
            "has_injection_payload": has_injection,
            "has_scope_expansion_flag": has_expansion,
            "is_wildcard": is_wildcard
        }
