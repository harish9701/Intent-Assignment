"""
Lightweight Inference REST API Server for Intent Manifest Inference & Divergence Models.
Allows querying the models programmatically via HTTP JSON endpoints.
"""
import os
import json
import sys
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse

# Ensure src is in sys.path
SYS_BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if SYS_BASE not in sys.path:
    sys.path.insert(0, SYS_BASE)

from src.models.baseline_frequency import FrequencyBaselineModel
from src.models.statistical_ml import StatisticalPatternModel
from src.models.llm_hybrid import LLMHybridManifestModel
from src.divergence.intent_divergence import IntentDivergenceEngine

BENCHMARK_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "data", "benchmark_results.json")

model_baseline = FrequencyBaselineModel()
model_statml = StatisticalPatternModel()
model_hybrid = LLMHybridManifestModel()
divergence_engine = IntentDivergenceEngine()

class ModelInferenceRequestHandler(BaseHTTPRequestHandler):
    def _send_json(self, data, status_code=200):
        body = json.dumps(data, indent=2).encode('utf-8')
        self.send_response(status_code)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == '/api/benchmark':
            if os.path.exists(BENCHMARK_PATH):
                with open(BENCHMARK_PATH, 'r') as f:
                    self._send_json(json.load(f))
            else:
                self._send_json({"error": "Benchmark data not generated yet."}, status_code=404)
        else:
            self._send_json({"message": "Intent Manifest Inference ML API Active. Use /api/infer-manifest or /api/analyze-divergence."})

    def do_POST(self):
        parsed = urlparse(self.path)
        content_length = int(self.headers.get('Content-Length', 0))
        body_bytes = self.rfile.read(content_length)
        
        try:
            body = json.loads(body_bytes.decode('utf-8')) if body_bytes else {}
        except Exception as e:
            self._send_json({"error": f"Invalid JSON: {str(e)}"}, status_code=400)
            return

        if parsed.path == '/api/infer-manifest':
            model_type = body.get("model", "hybrid")
            trace_data = body.get("trace", {})
            
            if model_type == "baseline":
                res = model_baseline.predict_manifest(trace_data)
            elif model_type == "statml":
                res = model_statml.predict_manifest(trace_data)
            else:
                res = model_hybrid.predict_manifest(trace_data)
                
            self._send_json(res)
            
        elif parsed.path == '/api/analyze-divergence':
            res = divergence_engine.analyze_triplet(body)
            self._send_json(res)
            
        else:
            self._send_json({"error": "Not Found"}, status_code=404)

def run_server(port=8000):
    server_address = ('', port)
    httpd = HTTPServer(server_address, ModelInferenceRequestHandler)
    print(f"[+] Intent Manifest Model API running on http://localhost:{port}")
    httpd.serve_forever()

if __name__ == '__main__':
    run_server(8000)
