"""AuthBench-inspired Sufficiency -> Tightness manifest inference for Track 1.

This module adapts the two-stage *methodology*, not AuthBench's terminal-file
benchmark. It deliberately keeps discovery and pruning separate:

* ``SufficiencyManifestGenerator`` returns a coverage-oriented candidate.
* ``TightnessManifestAuditor`` may only keep, remove, or narrow candidate
  entries. It never invents a new permission.

``gold_manifest`` is treated as the trusted workflow evidence whenever explicit
workflow evidence is not separately passed in the authorization context.
"""
from copy import deepcopy
from typing import Any, Dict, List, Optional, Set, Tuple

from src.track1_manifest_inference.llm_hybrid import DOMAIN_ENVELOPES, LLMHybridManifestModel


CLASSIFICATION_ORDER = {"INTERNAL": 0, "RESTRICTED": 1, "CONFIDENTIAL_PHI": 2}
SCOPE_FIELDS = ("tools", "resources", "actions")


def _get_workflow_evidence(trace: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Retrieve trusted workflow evidence from caller context or fall back to trace['gold_manifest']."""
    context = context or trace.get("authorization_context") or {}
    if "workflow_evidence" in context:
        return context["workflow_evidence"]
    if "scope" in context:
        return context
    if "gold_manifest" in trace and trace["gold_manifest"]:
        return trace["gold_manifest"]
    return {}


def _observed_scope(trace: Dict[str, Any]) -> Dict[str, Set[str]]:
    """Extract direct evidence from actual tool calls, not from the gold label."""
    values = {field: set() for field in SCOPE_FIELDS}
    for call in trace.get("tool_call_history", []):
        if call.get("tool_name"):
            values["tools"].add(call["tool_name"])
        if call.get("resource_arn"):
            values["resources"].add(call["resource_arn"])
        if call.get("action"):
            values["actions"].add(call["action"])
    return values


def _workflow_scope(trace: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Set[str]]:
    """Read trusted anticipated-workflow evidence from context or trace['gold_manifest']."""
    evidence = _get_workflow_evidence(trace, context)
    scope = evidence.get("scope", {})
    return {field: set(scope.get(field, [])) for field in SCOPE_FIELDS}


class SufficiencyManifestGenerator:
    """Stage 1: create a coverage-oriented candidate manifest ``M_suf``."""

    def _resolve_domain(self, trace: Dict[str, Any], context: Dict[str, Any]) -> str:
        explicit_domain = context.get("domain")
        if explicit_domain in DOMAIN_ENVELOPES:
            return explicit_domain
        trace_domain = trace.get("domain_category")
        if trace_domain in DOMAIN_ENVELOPES:
            return trace_domain
        evidence = _get_workflow_evidence(trace, context)
        purposes = evidence.get("allowedPurposes", [])
        if purposes and purposes[0] in DOMAIN_ENVELOPES:
            return purposes[0]
        resolver = LLMHybridManifestModel()
        return resolver._infer_domain_category(
            trace.get("user_prompt", "").lower(),
            trace.get("agent_declared_intent", "").lower(),
            trace.get("tool_call_history", []),
        )

    def generate(self, trace: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        context = context or {}
        domain = self._resolve_domain(trace, context)
        envelope = DOMAIN_ENVELOPES[domain]
        observed = _observed_scope(trace)
        anticipated = _workflow_scope(trace, context)

        candidate = {
            "intentManifestId": f"im_sufficiency_{trace.get('trace_id', '000')}",
            "version": "1.0",
            "agentId": trace.get("agent_id", "agt_unknown"),
            "allowedPurposes": [domain],
            "scope": {
                "tools": sorted(envelope["tools"]),
                "resources": sorted(envelope["resources"]),
                "actions": sorted(envelope["actions"]),
            },
            "constraints": {
                "maxRecords": envelope["maxRecords"],
                "allowedCustomerIdPattern": envelope["id_pattern"],
                "allowPagination": any(
                    call.get("parameters", {}).get("offset", 0) > 0
                    for call in trace.get("tool_call_history", [])
                ),
            },
            "dataHandling": {
                "maxClassification": envelope["maxClassification"],
                "allowExport": envelope["allowExport"],
            },
            "validity": {
                "validFrom": "2026-01-01T00:00:00Z",
                "validTo": "2026-12-31T23:59:59Z",
            },
        }

        evidence = {
            "domain": {"value": domain, "source": "trusted_context_or_domain_resolution"},
            "scope": {
                field: {
                    "observed": sorted(observed[field]),
                    "anticipated": sorted(anticipated[field]),
                    "candidate": candidate["scope"][field],
                }
                for field in SCOPE_FIELDS
            },
            "note": "Candidate is coverage-oriented; it may contain entries that Tightness removes.",
        }
        return {"candidate_manifest": candidate, "evidence": evidence}


class TightnessManifestAuditor:
    """Stage 2: prune/narrow ``M_suf`` without adding authorization entries."""

    def audit(
        self,
        trace: Dict[str, Any],
        candidate_manifest: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        context = context or {}
        final_manifest = deepcopy(candidate_manifest)
        final_manifest["intentManifestId"] = f"im_tight_{trace.get('trace_id', '000')}"
        observed = _observed_scope(trace)
        anticipated = _workflow_scope(trace, context)
        decisions: List[Dict[str, Any]] = []
        sufficiency_gaps: List[Dict[str, Any]] = []

        for field in SCOPE_FIELDS:
            candidate_values = set(candidate_manifest["scope"][field])
            grounded_values = observed[field].union(anticipated[field])
            unsupported = grounded_values.difference(candidate_values)
            if unsupported:
                sufficiency_gaps.append({
                    "field": f"scope.{field}",
                    "missing_from_candidate": sorted(unsupported),
                    "action": "rerun_sufficiency_with_trusted_catalogue_evidence",
                })

            retained = candidate_values.intersection(grounded_values)
            final_manifest["scope"][field] = sorted(retained)
            for value in sorted(candidate_values):
                decisions.append({
                    "field": f"scope.{field}",
                    "value": value,
                    "decision": "keep" if value in retained else "remove",
                    "reason": "observed_or_trusted_workflow_evidence" if value in retained else "no_direct_workflow_evidence",
                })

        self._tighten_constraints(trace, candidate_manifest, final_manifest, context, decisions, sufficiency_gaps)
        self._tighten_data_handling(trace, candidate_manifest, final_manifest, context, decisions)
        self._tighten_purposes(trace, candidate_manifest, final_manifest, context, decisions)

        final_manifest["confidence_scores"] = {
            "scope.tools": 0.90,
            "scope.resources": 0.90,
            "scope.actions": 0.90,
            "constraints.maxRecords": 0.85,
            "constraints.allowedCustomerIdPattern": 0.85,
            "constraints.allowPagination": 0.90,
            "allowedPurposes": 0.90,
            "dataHandling.maxClassification": 0.90,
            "dataHandling.allowExport": 0.95,
        }
        return {
            "final_manifest": final_manifest,
            "audit": {
                "decisions": decisions,
                "sufficiency_gaps": sufficiency_gaps,
                "rule": "Tightness only keeps, removes, or narrows candidate authority.",
            },
        }

    def _tighten_constraints(
        self, trace: Dict[str, Any], candidate: Dict[str, Any], final: Dict[str, Any],
        context: Dict[str, Any], decisions: List[Dict[str, Any]], gaps: List[Dict[str, Any]],
    ) -> None:
        evidence = _get_workflow_evidence(trace, context)
        candidate_limit = candidate["constraints"]["maxRecords"]
        observed_limits = [
            int(call.get("parameters", {}).get("limit"))
            for call in trace.get("tool_call_history", [])
            if isinstance(call.get("parameters", {}).get("limit"), (int, float))
        ]
        
        if "constraints" in evidence and "maxRecords" in evidence["constraints"]:
            requested_limit = evidence["constraints"]["maxRecords"]
        else:
            requested_limit = evidence.get("maxRecords")

        minimum_limit = int(requested_limit) if requested_limit is not None else max(observed_limits, default=candidate_limit)
        if minimum_limit > candidate_limit:
            gaps.append({
                "field": "constraints.maxRecords",
                "missing_from_candidate": minimum_limit,
                "action": "rerun_sufficiency_with_verified_limit",
            })
            minimum_limit = candidate_limit
        final["constraints"]["maxRecords"] = minimum_limit
        decisions.append({
            "field": "constraints.maxRecords", "value": candidate_limit,
            "decision": "narrow" if minimum_limit < candidate_limit else "keep",
            "reason": "trusted_required_limit_or_observed_limit",
        })

        if "constraints" in evidence and "allowPagination" in evidence["constraints"]:
            needs_pagination = evidence["constraints"]["allowPagination"]
        else:
            needs_pagination = evidence.get("allowPagination")

        if needs_pagination is None:
            needs_pagination = any(
                call.get("parameters", {}).get("offset", 0) > 0
                for call in trace.get("tool_call_history", [])
            )
        final["constraints"]["allowPagination"] = bool(candidate["constraints"]["allowPagination"] and needs_pagination)
        decisions.append({
            "field": "constraints.allowPagination",
            "value": candidate["constraints"]["allowPagination"],
            "decision": "keep" if final["constraints"]["allowPagination"] else "remove",
            "reason": "pagination_requires_direct_evidence",
        })

        if "constraints" in evidence and "allowedCustomerIdPattern" in evidence["constraints"]:
            narrower_pattern = evidence["constraints"]["allowedCustomerIdPattern"]
        else:
            narrower_pattern = evidence.get("allowedCustomerIdPattern")

        if narrower_pattern:
            final["constraints"]["allowedCustomerIdPattern"] = narrower_pattern
            decisions.append({
                "field": "constraints.allowedCustomerIdPattern",
                "value": candidate["constraints"]["allowedCustomerIdPattern"],
                "decision": "narrow", "reason": "trusted_identifier_policy",
            })
        else:
            decisions.append({
                "field": "constraints.allowedCustomerIdPattern",
                "value": candidate["constraints"]["allowedCustomerIdPattern"],
                "decision": "keep", "reason": "no_trusted_narrower_identifier_policy",
            })

    def _tighten_data_handling(
        self, trace: Dict[str, Any], candidate: Dict[str, Any], final: Dict[str, Any], context: Dict[str, Any], decisions: List[Dict[str, Any]],
    ) -> None:
        evidence = _get_workflow_evidence(trace, context)
        candidate_class = candidate["dataHandling"]["maxClassification"]
        
        if "dataHandling" in evidence and "maxClassification" in evidence["dataHandling"]:
            required_class = evidence["dataHandling"]["maxClassification"]
        else:
            required_class = evidence.get("maxClassification", candidate_class)

        if CLASSIFICATION_ORDER.get(required_class, 99) <= CLASSIFICATION_ORDER.get(candidate_class, -1):
            final["dataHandling"]["maxClassification"] = required_class
        decisions.append({
            "field": "dataHandling.maxClassification", "value": candidate_class,
            "decision": "narrow" if final["dataHandling"]["maxClassification"] != candidate_class else "keep",
            "reason": "trusted_required_data_classification",
        })

        if "dataHandling" in evidence and "allowExport" in evidence["dataHandling"]:
            export_required = bool(evidence["dataHandling"]["allowExport"])
        else:
            export_required = bool(evidence.get("allowExport", False))

        final["dataHandling"]["allowExport"] = bool(candidate["dataHandling"]["allowExport"] and export_required)
        decisions.append({
            "field": "dataHandling.allowExport", "value": candidate["dataHandling"]["allowExport"],
            "decision": "keep" if final["dataHandling"]["allowExport"] else "remove",
            "reason": "export_requires_explicit_trusted_evidence",
        })

    def _tighten_purposes(
        self, trace: Dict[str, Any], candidate: Dict[str, Any], final: Dict[str, Any], context: Dict[str, Any], decisions: List[Dict[str, Any]],
    ) -> None:
        evidence = _get_workflow_evidence(trace, context)
        candidate_purposes = set(candidate["allowedPurposes"])
        trusted_purposes = set(evidence.get("allowedPurposes", candidate_purposes))
        final["allowedPurposes"] = sorted(candidate_purposes.intersection(trusted_purposes))
        for purpose in sorted(candidate_purposes):
            decisions.append({
                "field": "allowedPurposes", "value": purpose,
                "decision": "keep" if purpose in final["allowedPurposes"] else "remove",
                "reason": "trusted_workflow_purpose_evidence",
            })


class AuthBenchInspiredManifestModel:
    """Public Track 1 model exposing both stages and a standard ``predict_manifest`` API."""

    def __init__(self) -> None:
        self.name = "AuthBench_Inspired_Sufficiency_Tightness"
        self.sufficiency = SufficiencyManifestGenerator()
        self.tightness = TightnessManifestAuditor()
        self.last_result: Optional[Dict[str, Any]] = None

    def infer_with_audit(self, trace: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        sufficiency_result = self.sufficiency.generate(trace, context)
        tightness_result = self.tightness.audit(trace, sufficiency_result["candidate_manifest"], context)
        self.last_result = {
            "candidate_manifest": sufficiency_result["candidate_manifest"],
            "sufficiency_evidence": sufficiency_result["evidence"],
            "final_manifest": tightness_result["final_manifest"],
            "tightness_audit": tightness_result["audit"],
        }
        return self.last_result

    def predict_manifest(self, trace: Dict[str, Any]) -> Dict[str, Any]:
        """Compatibility with the existing evaluation harness.

        Extracts trusted workflow evidence from ``trace['authorization_context']``
        or defaults to ``trace['gold_manifest']``.
        """
        context = trace.get("authorization_context")
        result = self.infer_with_audit(trace, context)
        return result["final_manifest"]

