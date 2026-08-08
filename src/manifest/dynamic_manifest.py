"""
Dynamic Intent Manifest Boundary Engine & PEP/PDP Policy Enforcer.
Key Features:
- Real-time zero-trust security gateway enforcing the Intent Manifest as an active runtime boundary for agent execution.
- Evaluates proposed agent tool calls across 8 security dimensions before execution.
- Supports ARN wildcard matching, parametric regex constraint validation, record limit ceilings, and data exfiltration guardrails.
- Provides dynamic boundary updates and security audit logging.
"""

import re
import fnmatch
from typing import Dict, Any, List, Optional, Tuple

CLASSIFICATION_LEVELS = {
    "INTERNAL": 1,
    "RESTRICTED": 2,
    "CONFIDENTIAL_PHI": 3
}

class BoundaryDecision:
    def __init__(self, permitted: bool, decision_code: str, violations: List[str], manifest_id: str, tool_call: Dict[str, Any]):
        self.permitted = permitted
        self.decision_code = decision_code
        self.violations = violations
        self.manifest_id = manifest_id
        self.tool_call = tool_call

    def to_dict(self) -> Dict[str, Any]:
        return {
            "permitted": self.permitted,
            "decision_code": self.decision_code,
            "violations": self.violations,
            "manifest_id": self.manifest_id,
            "evaluated_tool_call": self.tool_call
        }

class DynamicManifestBoundaryEnforcer:
    def __init__(self):
        self.audit_logs: List[Dict[str, Any]] = []

    def enforce_boundary(self, manifest: Dict[str, Any], proposed_tool_call: Dict[str, Any]) -> BoundaryDecision:
        """
        Evaluates a proposed agent tool call against an active Intent Manifest.
        Returns a BoundaryDecision detailing permission status and any boundary violations.
        """
        manifest_id = manifest.get("intentManifestId", "unknown_manifest")
        violations = []

        scope = manifest.get("scope", {})
        allowed_tools = set(scope.get("tools", []))
        allowed_resources = scope.get("resources", [])
        allowed_actions = set(scope.get("actions", []))
        constraints = manifest.get("constraints", {})
        data_handling = manifest.get("dataHandling", {})

        tool_name = proposed_tool_call.get("tool_name", "").strip()
        resource_arn = proposed_tool_call.get("resource_arn", "").strip()
        action = proposed_tool_call.get("action", "").strip()
        params = proposed_tool_call.get("parameters", {})

        # Dimension 1: Tool Authorization
        if tool_name not in allowed_tools:
            violations.append(f"Tool '{tool_name}' is not granted in manifest scope.tools: {sorted(list(allowed_tools))}")

        # Dimension 2: Canonical Resource ARN Authorization (with wildcard matching)
        if resource_arn:
            resource_authorized = self._check_resource_arn_permitted(resource_arn, allowed_resources)
            if not resource_authorized:
                violations.append(f"Resource ARN '{resource_arn}' violates manifest scope.resources envelope: {allowed_resources}")

        # Dimension 3: API Action Authorization
        if action and action not in allowed_actions:
            violations.append(f"API Action '{action}' is not granted in manifest scope.actions: {sorted(list(allowed_actions))}")

        # Dimension 4: maxRecords Limit Constraint
        max_records_limit = constraints.get("maxRecords", 500)
        requested_limit = params.get("limit") or params.get("maxRecords") or params.get("batch_size")
        if requested_limit is not None:
            try:
                req_int = int(requested_limit)
                if req_int > max_records_limit:
                    violations.append(f"Requested record limit ({req_int}) exceeds manifest maxRecords ceiling ({max_records_limit}).")
            except (ValueError, TypeError):
                pass

        # Dimension 5: Parametric Regex Customer/Entity ID Pattern Constraint
        id_pattern = constraints.get("allowedCustomerIdPattern")
        if id_pattern and id_pattern != ".*":
            for param_key in ["customer_id", "patient_id", "account_id", "host_id", "client_id", "order_id", "ticket_id"]:
                if param_key in params:
                    val_str = str(params[param_key])
                    if val_str in ["*", "ALL_GLOBAL_WAREHOUSES", "SUPER_ADMIN"] or not re.match(id_pattern, val_str):
                        violations.append(f"Parameter '{param_key}' value '{val_str}' violates manifest regex constraint '{id_pattern}'.")

        # Dimension 6: Pagination Constraint
        allow_pagination = constraints.get("allowPagination", True)
        requested_offset = params.get("offset", 0)
        try:
            if int(requested_offset) > 0 and not allow_pagination:
                violations.append("Pagination offset > 0 requested, but allowPagination is set to False in manifest constraints.")
        except (ValueError, TypeError):
            pass

        # Dimension 7: Data Exfiltration Policy
        allow_export = data_handling.get("allowExport", False)
        is_export_action = "export" in tool_name.lower() or "dump" in tool_name.lower() or params.get("export_raw_json") is True
        if is_export_action and not allow_export:
            violations.append("Data export operation attempted, but allowExport is set to False in manifest dataHandling.")

        # Dimension 8: Data Classification Ceiling Constraint
        max_class = data_handling.get("maxClassification", "INTERNAL")
        resource_class = self._infer_resource_classification(resource_arn, tool_name)
        if CLASSIFICATION_LEVELS.get(resource_class, 1) > CLASSIFICATION_LEVELS.get(max_class, 1):
            violations.append(f"Resource sensitivity '{resource_class}' exceeds manifest maxClassification ceiling '{max_class}'.")

        permitted = len(violations) == 0
        decision_code = "PERMITTED" if permitted else self._determine_decision_code(violations)

        decision = BoundaryDecision(
            permitted=permitted,
            decision_code=decision_code,
            violations=violations,
            manifest_id=manifest_id,
            tool_call=proposed_tool_call
        )

        self.audit_logs.append(decision.to_dict())
        return decision

    def _check_resource_arn_permitted(self, resource_arn: str, allowed_resources: List[str]) -> bool:
        """Validates if resource_arn matches any allowed resource or prefix wildcard."""
        for pattern in allowed_resources:
            if pattern == resource_arn:
                return True
            if pattern.endswith("*") and resource_arn.startswith(pattern[:-1]):
                return True
            if fnmatch.fnmatch(resource_arn, pattern):
                return True
        return False

    def _infer_resource_classification(self, resource_arn: str, tool_name: str) -> str:
        """Determines the data classification level of a target resource."""
        text = f"{resource_arn} {tool_name}".lower()
        if "healthlake" in text or "patient" in text or "lab" in text or "ehr" in text:
            return "CONFIDENTIAL_PHI"
        elif "redshift" in text or "payroll" in text or "finance" in text or "ledger" in text:
            return "RESTRICTED"
        return "INTERNAL"

    def _determine_decision_code(self, violations: List[str]) -> str:
        """Categorizes primary violation reason into a machine-readable code."""
        v_text = " ".join(violations)
        if "scope.tools" in v_text:
            return "DENIED_UNAUTHORIZED_TOOL"
        if "scope.resources" in v_text:
            return "DENIED_RESOURCE_OUT_OF_BOUNDS"
        if "scope.actions" in v_text:
            return "DENIED_UNAUTHORIZED_ACTION"
        if "maxRecords" in v_text:
            return "DENIED_PARAM_LIMIT_EXCEEDED"
        if "regex constraint" in v_text:
            return "DENIED_PATTERN_MISMATCH"
        if "allowExport" in v_text:
            return "DENIED_EXFILTRATION_PREVENTED"
        if "maxClassification" in v_text:
            return "DENIED_CLASSIFICATION_CEILING_EXCEEDED"
        return "DENIED_BOUNDARY_VIOLATION"
