# src/orchestration/logger.py
import json
import time

def log_trade_event(event_type: str, payload: dict, log_path="logs/qdc_audit_trail.jsonl"):
    entry = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "event_type": event_type,
        "payload": payload
    }
    with open(log_path, "a") as f:
        f.write(json.dumps(entry) + "\n")