# src/orchestration/logger.py
import json
import time
from pathlib import Path
from typing import Any, Dict, Optional

LOGS_DIR = Path("logs")
LOGS_DIR.mkdir(exist_ok=True)
AUDIT_LOG_FILE = LOGS_DIR / "qdc_audit_trail.jsonl"

def log_event(
    event_type: str,
    node_name: str,
    payload: Dict[str, Any],
    prompt: Optional[str] = None,
    response: Optional[str] = None,
    token_usage: Optional[Dict[str, int]] = None
):
    """
    Logs an event, prompt, or decision into an immutable JSONL audit trail.
    Ensures complete transparency for QDC judges and QuantumBlack evaluation.
    """
    record = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "event_type": event_type,
        "node_name": node_name,
        "prompt": prompt,
        "response": response,
        "token_usage": token_usage,
        "payload": payload
    }
    with open(AUDIT_LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, default=str) + "\n")
