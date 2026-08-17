"""
Primary Track 1: Ollama Local LLM Extractor for Intent Manifest Inference.
Key Features:
- Direct HTTP integration with local Ollama API (http://localhost:11434).
- Zero-shot / Few-shot JSON structured prompting for manifest extraction.
- Automatic fallback to schema extraction & domain ontology graph if Ollama server is offline.
"""
import json
import urllib.request
import urllib.error
import re
from typing import Dict, Any, List, Optional

from src.track1_manifest_inference.llm_hybrid import LLMHybridManifestModel

OLLAMA_DEFAULT_URL = "http://localhost:11434/api/generate"
OLLAMA_DEFAULT_MODEL = "llama3"

class OllamaLLMManifestModel:
    def __init__(self, ollama_url: str = OLLAMA_DEFAULT_URL, model_name: str = OLLAMA_DEFAULT_MODEL, timeout_seconds: float = 2.0):
        self.name = f"Ollama_LLM_{model_name}"
        self.ollama_url = ollama_url
        self.model_name = model_name
        self.timeout_seconds = timeout_seconds

    def predict_manifest(self, trace: Dict[str, Any]) -> Dict[str, Any]:
        """Infers an Intent Manifest for the given agent activity trace using local Ollama LLM."""
        prompt_text = self._build_llm_prompt(trace)
        ollama_response = self._query_ollama_api(prompt_text)

        if ollama_response:
            parsed_manifest = self._parse_llm_json(ollama_response, trace)
            if parsed_manifest:
                return parsed_manifest

        return self._fallback_manifest_extraction(trace)

    def _build_llm_prompt(self, trace: Dict[str, Any]) -> str:
        prompt = trace.get("user_prompt", "")
        declared = trace.get("agent_declared_intent", "")
        tool_calls = trace.get("tool_call_history", [])

        prompt_payload = {
            "task": "Extract an Intent Manifest JSON envelope specifying allowed scope, constraints, and data handling for an AI agent.",
            "user_prompt": prompt,
            "agent_declared_intent": declared,
            "observed_tool_calls": tool_calls,
            "required_format": {
                "allowedPurposes": ["string"],
                "scope": {
                    "tools": ["string"],
                    "resources": ["string"],
                    "actions": ["string"]
                },
                "constraints": {
                    "maxRecords": "int",
                    "allowedCustomerIdPattern": "regex_string",
                    "allowPagination": "bool"
                },
                "dataHandling": {
                    "maxClassification": "INTERNAL | RESTRICTED | CONFIDENTIAL_PHI",
                    "allowExport": "bool"
                }
            }
        }
        return f"Respond ONLY with a valid JSON object matching the required manifest schema.\n\nInput Data:\n{json.dumps(prompt_payload)}"

    def _query_ollama_api(self, prompt: str) -> Optional[str]:
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "options": {
                "temperature": 0.1
            }
        }
        try:
            data = json.dumps(payload).encode('utf-8')
            req = urllib.request.Request(self.ollama_url, data=data, headers={'Content-Type': 'application/json'})
            with urllib.request.urlopen(req, timeout=self.timeout_seconds) as response:
                if response.status == 200:
                    resp_json = json.loads(response.read().decode('utf-8'))
                    return resp_json.get("response", "")
        except Exception:
            pass
        return None

    def _parse_llm_json(self, raw_text: str, trace: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        try:
            cleaned = raw_text.strip()
            if cleaned.startswith("```json"):
                cleaned = cleaned[7:]
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]
            cleaned = cleaned.strip()

            parsed = json.loads(cleaned)

            if "scope" in parsed and "tools" in parsed["scope"]:
                return {
                    "intentManifestId": f"im_ollama_{trace.get('trace_id', '000')}",
                    "version": "1.0",
                    "agentId": trace.get("agent_id", "agt_unknown"),
                    "allowedPurposes": parsed.get("allowedPurposes", [trace.get("domain_category", "general_query")]),
                    "scope": {
                        "tools": sorted(parsed["scope"].get("tools", [])),
                        "resources": sorted(parsed["scope"].get("resources", [])),
                        "actions": sorted(parsed["scope"].get("actions", []))
                    },
                    "constraints": {
                        "maxRecords": int(parsed.get("constraints", {}).get("maxRecords", 500)),
                        "allowedCustomerIdPattern": str(parsed.get("constraints", {}).get("allowedCustomerIdPattern", ".*")),
                        "allowPagination": bool(parsed.get("constraints", {}).get("allowPagination", False))
                    },
                    "dataHandling": {
                        "maxClassification": parsed.get("dataHandling", {}).get("maxClassification", "INTERNAL"),
                        "allowExport": bool(parsed.get("dataHandling", {}).get("allowExport", False))
                    },
                    "validity": {
                        "validFrom": "2026-01-01T00:00:00Z",
                        "validTo": "2026-12-31T23:59:59Z"
                    },
                    "confidence_scores": {
                        "scope.tools": 0.95,
                        "scope.resources": 0.95,
                        "scope.actions": 0.95,
                        "constraints.maxRecords": 0.92,
                        "constraints.allowedCustomerIdPattern": 0.94,
                        "constraints.allowPagination": 0.95,
                        "allowedPurposes": 0.95,
                        "dataHandling.maxClassification": 0.95,
                        "dataHandling.allowExport": 0.95
                    }
                }
        except Exception:
            pass
        return None

    def _fallback_manifest_extraction(self, trace: Dict[str, Any]) -> Dict[str, Any]:
        """Use Pattern 3 as the single deterministic fallback for local inference."""
        manifest = LLMHybridManifestModel().predict_manifest(trace)
        manifest["intentManifestId"] = f"im_ollama_fallback_{trace.get('trace_id', '000')}"
        return manifest
