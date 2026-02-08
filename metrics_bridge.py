"""
Metrics bridge: connects PPO training output to the Dominator Training web app.

Two modes:
1. FILE MODE (default): Writes metrics to a JSONL file that the API server tails
2. API MODE: POSTs metrics directly to the API (requires auth)

Usage — run in a separate terminal while training:

    # Terminal 1: training
    python train_fast.py 2>&1 | tee training.log

    # Terminal 2: metrics bridge
    tail -f training.log | python metrics_bridge.py

Or pipe directly:
    python train_fast.py 2>&1 | tee >(python metrics_bridge.py)

Environment variables:
    METRICS_FILE       — Path to shared metrics file (default: data/metrics/live_metrics.jsonl)
    DOMINATOR_API_URL  — API base URL for API mode (default: http://localhost:8000)
    DOMINATOR_RUN_ID   — Training run UUID from the web app
    DOMINATOR_API_KEY  — Admin access token (optional, for API mode)
"""

import sys
import os
import re
import json
import time
import signal
from pathlib import Path
from datetime import datetime, timezone

# ─── Config ──────────────────────────────────────────────────────────
METRICS_DIR = Path(os.environ.get("METRICS_FILE", "data/metrics")).parent
METRICS_FILE = Path(os.environ.get("METRICS_FILE", "data/metrics/live_metrics.jsonl"))
API_URL = os.environ.get("DOMINATOR_API_URL", "http://localhost:8000")
RUN_ID = os.environ.get("DOMINATOR_RUN_ID", "")
API_KEY = os.environ.get("DOMINATOR_API_KEY", "")

# ─── Regex patterns for rlgym-ppo output ─────────────────────────────
PATTERNS = {
    "policy_reward": re.compile(r"Policy Reward:\s*([\-\d.]+)"),
    "entropy": re.compile(r"Policy Entropy:\s*([\-\d.]+)"),
    "loss_v": re.compile(r"Value Function Loss:\s*([\-\d.]+)"),
    "kl_divergence": re.compile(r"Mean KL Divergence:\s*([\-\d.]+)"),
    "clip_fraction": re.compile(r"SB3 Clip Fraction:\s*([\-\d.]+)"),
    "policy_update_mag": re.compile(r"Policy Update Magnitude:\s*([\-\d.]+)"),
    "value_update_mag": re.compile(r"Value Function Update Magnitude:\s*([\-\d.]+)"),
    "collected_sps": re.compile(r"Collected Steps per Second:\s*([\-\d.,]+)"),
    "overall_sps": re.compile(r"Overall Steps per Second:\s*([\-\d.,]+)"),
    "collection_time": re.compile(r"Timestep Collection Time:\s*([\-\d.]+)"),
    "consumption_time": re.compile(r"Timestep Consumption Time:\s*([\-\d.]+)"),
    "ppo_batch_time": re.compile(r"PPO Batch Consumption Time:\s*([\-\d.]+)"),
    "iteration_time": re.compile(r"Total Iteration Time:\s*([\-\d.]+)"),
    "model_updates": re.compile(r"Cumulative Model Updates:\s*([\-\d.,]+)"),
    "cumulative_steps": re.compile(r"Cumulative Timesteps:\s*([\-\d.,]+)"),
    "steps_collected": re.compile(r"Timesteps Collected:\s*([\-\d.,]+)"),
}

BEGIN_MARKER = "--------BEGIN ITERATION REPORT--------"
END_MARKER = "--------END ITERATION REPORT--------"
CHECKPOINT_PATTERN = re.compile(r"Saving checkpoint (\d+)")
CHECKPOINT_SAVED = re.compile(r"Checkpoint (\d+) saved")


def parse_number(s: str) -> float:
    """Parse a number string, removing commas."""
    return float(s.replace(",", ""))


class MetricsBridge:
    def __init__(self):
        self.in_report = False
        self.current_report = {}
        self.iteration_count = 0
        self.start_time = time.time()

        # Ensure metrics directory exists
        METRICS_DIR.mkdir(parents=True, exist_ok=True)
        METRICS_FILE.parent.mkdir(parents=True, exist_ok=True)

        # Write status file
        status = {
            "status": "running",
            "started_at": datetime.now(timezone.utc).isoformat(),
            "pid": os.getpid(),
            "run_id": RUN_ID or None,
        }
        status_path = METRICS_FILE.parent / "status.json"
        with open(status_path, "w") as f:
            json.dump(status, f)

        print(f"[MetricsBridge] Writing metrics to {METRICS_FILE}")
        if RUN_ID:
            print(f"[MetricsBridge] Run ID: {RUN_ID}")
        if API_KEY:
            print(f"[MetricsBridge] API mode enabled → {API_URL}")

    def process_line(self, line: str):
        line = line.strip()
        if not line:
            return

        # Check for checkpoint saves
        m = CHECKPOINT_PATTERN.search(line)
        if m:
            self._emit_checkpoint(int(m.group(1)), saving=True)
            return

        m = CHECKPOINT_SAVED.search(line)
        if m:
            self._emit_checkpoint(int(m.group(1)), saving=False)
            return

        # Check for iteration report boundaries
        if BEGIN_MARKER in line:
            self.in_report = True
            self.current_report = {}
            return

        if END_MARKER in line:
            self.in_report = False
            if self.current_report:
                self._emit_metrics(self.current_report)
            self.current_report = {}
            return

        # Parse metrics within a report
        if self.in_report:
            for key, pattern in PATTERNS.items():
                m = pattern.search(line)
                if m:
                    self.current_report[key] = parse_number(m.group(1))
                    break

        # Also catch error lines
        if "ERROR" in line or "Traceback" in line:
            self._emit_event("error", {"message": line})

    def _emit_metrics(self, report: dict):
        """Write a metrics point to the JSONL file and optionally POST to API."""
        self.iteration_count += 1
        elapsed = time.time() - self.start_time

        point = {
            "type": "metrics",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "iteration": self.iteration_count,
            "elapsed_seconds": round(elapsed, 1),
            # Core metrics (matching what the web app expects)
            "step": int(report.get("cumulative_steps", 0)),
            "avg_reward": round(report.get("policy_reward", 0), 4),
            "entropy": round(report.get("entropy", 0), 4),
            "loss_pi": round(report.get("kl_divergence", 0), 6),  # KL as proxy for policy loss
            "loss_v": round(report.get("loss_v", 0), 6),
            # Extended metrics
            "clip_fraction": round(report.get("clip_fraction", 0), 5),
            "policy_update_mag": round(report.get("policy_update_mag", 0), 5),
            "value_update_mag": round(report.get("value_update_mag", 0), 5),
            "collected_sps": round(report.get("collected_sps", 0), 1),
            "overall_sps": round(report.get("overall_sps", 0), 1),
            "iteration_time": round(report.get("iteration_time", 0), 3),
            "model_updates": int(report.get("model_updates", 0)),
        }

        # Write to JSONL file
        with open(METRICS_FILE, "a") as f:
            f.write(json.dumps(point) + "\n")

        # Print summary
        step = point["step"]
        reward = point["avg_reward"]
        entropy = point["entropy"]
        sps = point["overall_sps"]
        print(
            f"[MetricsBridge] iter={self.iteration_count:>5} "
            f"step={step:>12,} "
            f"reward={reward:>8.3f} "
            f"entropy={entropy:>6.3f} "
            f"sps={sps:>10,.0f} "
            f"elapsed={elapsed:>7.1f}s"
        )

        # POST to API if configured
        if RUN_ID and API_KEY:
            self._post_to_api("/api/training-runs/metrics", point)

    def _emit_checkpoint(self, step: int, saving: bool):
        """Write a checkpoint event."""
        event = {
            "type": "checkpoint",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "step": step,
            "status": "saving" if saving else "saved",
            "path": f"data/checkpoints/{step}/",
        }
        with open(METRICS_FILE, "a") as f:
            f.write(json.dumps(event) + "\n")

        if not saving:
            print(f"[MetricsBridge] 💾 Checkpoint saved: step {step:,}")

    def _emit_event(self, event_type: str, data: dict):
        """Write a generic event."""
        event = {
            "type": event_type,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **data,
        }
        with open(METRICS_FILE, "a") as f:
            f.write(json.dumps(event) + "\n")

    def _post_to_api(self, path: str, data: dict):
        """POST data to the API (best-effort, no crash on failure)."""
        try:
            import urllib.request
            req = urllib.request.Request(
                f"{API_URL}{path}",
                data=json.dumps(data).encode(),
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {API_KEY}",
                },
                method="POST",
            )
            urllib.request.urlopen(req, timeout=5)
        except Exception:
            pass  # Don't crash training if API is down

    def finalize(self):
        """Write final status."""
        status = {
            "status": "stopped",
            "stopped_at": datetime.now(timezone.utc).isoformat(),
            "total_iterations": self.iteration_count,
            "elapsed_seconds": round(time.time() - self.start_time, 1),
        }
        status_path = METRICS_FILE.parent / "status.json"
        with open(status_path, "w") as f:
            json.dump(status, f)

        event = {
            "type": "done",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "status": "stopped",
            "total_iterations": self.iteration_count,
        }
        with open(METRICS_FILE, "a") as f:
            f.write(json.dumps(event) + "\n")

        print(f"\n[MetricsBridge] Stopped. {self.iteration_count} iterations recorded.")


def main():
    bridge = MetricsBridge()

    def handle_signal(sig, frame):
        bridge.finalize()
        sys.exit(0)

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    print("[MetricsBridge] Reading from stdin... (pipe training output here)")

    try:
        for line in sys.stdin:
            bridge.process_line(line)
    except KeyboardInterrupt:
        pass
    finally:
        bridge.finalize()


if __name__ == "__main__":
    main()