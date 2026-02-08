"""
Training Server — runs on the bot machine, accepts commands from the web app.

Endpoints:
    GET  /status          — is training running? what step?
    POST /start           — start training with a config JSON body
    POST /stop            — stop current training
    GET  /checkpoints     — list available checkpoints
    GET  /metrics          — get latest metrics
    GET  /config          — get active config

Usage:
    cd E:\DominanceBot_v2
    .\venv\Scripts\Activate.ps1
    python training_server.py

The web app calls this server to start/stop training remotely.
Set TRAINING_SERVER_PORT env var to change port (default 9000).
"""

import os
import sys
import json
import time
import signal
import subprocess
import threading
from pathlib import Path
from datetime import datetime, timezone

from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# ─── State ────────────────────────────────────────────────────────────
BOT_DIR = Path(os.environ.get("BOT_DIR", os.path.dirname(os.path.abspath(__file__))))
METRICS_FILE = BOT_DIR / "data" / "metrics" / "live_metrics.jsonl"
STATUS_FILE = BOT_DIR / "data" / "metrics" / "status.json"
CONFIG_FILE = BOT_DIR / "data" / "train_config.json"
VENV_PYTHON = BOT_DIR / "venv" / "Scripts" / "python.exe"  # Windows
if not VENV_PYTHON.exists():
    VENV_PYTHON = BOT_DIR / "venv" / "bin" / "python"  # Linux/Mac
if not VENV_PYTHON.exists():
    VENV_PYTHON = Path(sys.executable)  # Fallback to current python


class TrainingManager:
    def __init__(self):
        self.process = None
        self.bridge_process = None
        self.log_lines = []
        self.max_log_lines = 1000
        self.started_at = None
        self.config = None
        self._lock = threading.Lock()

    @property
    def is_running(self):
        return self.process is not None and self.process.poll() is None

    def start(self, config: dict) -> dict:
        with self._lock:
            if self.is_running:
                return {"error": "Training already running", "status": "running"}

            # Save config
            CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
            with open(CONFIG_FILE, "w") as f:
                json.dump(config, f, indent=2)

            # Clear old metrics
            METRICS_FILE.parent.mkdir(parents=True, exist_ok=True)
            if METRICS_FILE.exists():
                METRICS_FILE.unlink()

            self.config = config
            self.log_lines = []
            self.started_at = datetime.now(timezone.utc).isoformat()

            # Start training process
            cmd = [
                str(VENV_PYTHON),
                str(BOT_DIR / "train_configurable.py"),
                "--config", str(CONFIG_FILE),
            ]

            print(f"[TrainingServer] Starting: {' '.join(cmd)}")

            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                cwd=str(BOT_DIR),
                bufsize=1,
                universal_newlines=True,
            )

            # Start bridge in a thread that reads stdout and pipes to metrics_bridge
            self.bridge_process = None
            bridge_thread = threading.Thread(target=self._run_bridge, daemon=True)
            bridge_thread.start()

            return {
                "status": "started",
                "pid": self.process.pid,
                "config": config,
            }

    def _run_bridge(self):
        """Read training stdout, log it, and pipe to metrics_bridge."""
        from metrics_bridge import MetricsBridge
        bridge = MetricsBridge()

        try:
            for line in self.process.stdout:
                line = line.rstrip('\n')
                # Store in log buffer
                self.log_lines.append({
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "message": line,
                })
                if len(self.log_lines) > self.max_log_lines:
                    self.log_lines = self.log_lines[-self.max_log_lines:]

                # Pass to metrics bridge
                bridge.process_line(line)

                # Also print to server console
                print(f"[Training] {line}")

        except Exception as e:
            print(f"[TrainingServer] Bridge error: {e}")
        finally:
            bridge.finalize()
            print("[TrainingServer] Training process ended.")

    def stop(self) -> dict:
        with self._lock:
            if not self.is_running:
                return {"status": "not_running"}

            print("[TrainingServer] Stopping training...")
            # Send SIGTERM (or terminate on Windows)
            try:
                self.process.terminate()
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()

            self.process = None
            return {"status": "stopped"}

    def get_status(self) -> dict:
        status = {
            "running": self.is_running,
            "started_at": self.started_at,
            "config": self.config,
            "pid": self.process.pid if self.is_running else None,
        }

        # Read latest metrics from file
        if METRICS_FILE.exists():
            try:
                with open(METRICS_FILE, "r") as f:
                    lines = f.readlines()
                # Find last metrics line
                for line in reversed(lines):
                    try:
                        data = json.loads(line.strip())
                        if data.get("type") == "metrics":
                            status["latest_metrics"] = data
                            break
                    except json.JSONDecodeError:
                        continue
                status["total_events"] = len(lines)
            except IOError:
                pass

        return status

    def get_logs(self, last_n: int = 100) -> list:
        return self.log_lines[-last_n:]


manager = TrainingManager()


# ─── Endpoints ────────────────────────────────────────────────────────

@app.route("/status", methods=["GET"])
def get_status():
    return jsonify(manager.get_status())


@app.route("/start", methods=["POST"])
def start_training():
    config = request.get_json()
    if not config:
        # Use defaults
        config = {
            "mode": "1v1",
            "checkpoint_path": None,
            "rewards": {
                "goal": 10.0, "touch": 3.0, "velocity_ball_to_goal": 5.0,
                "velocity_player_to_ball": 1.0, "speed": 0.1,
                "boost_penalty": 0.0, "demo": 0.0, "aerial": 0.0,
            },
            "hyperparameters": {
                "policy_lr": 5e-4, "critic_lr": 5e-4, "n_proc": 16,
                "ppo_batch_size": 50000, "ts_per_iteration": 50000,
                "ppo_epochs": 3, "ppo_ent_coef": 0.01, "gamma": 0.99, "tick_skip": 8,
            },
            "training": {
                "save_every_ts": 5000000, "timestep_limit": 10000000000,
                "timeout_seconds": 15, "log_to_wandb": False,
            },
        }
    result = manager.start(config)
    status_code = 409 if "error" in result else 200
    return jsonify(result), status_code


@app.route("/stop", methods=["POST"])
def stop_training():
    result = manager.stop()
    return jsonify(result)


@app.route("/checkpoints", methods=["GET"])
def list_checkpoints():
    found = []
    for base_name in ["checkpoints", "checkpoints_v1", "checkpoints_v2"]:
        base = BOT_DIR / "data" / base_name
        if not base.exists():
            continue
        for run_dir in sorted(base.iterdir()):
            if not run_dir.is_dir():
                continue
            for step_dir in sorted(run_dir.iterdir()):
                if not step_dir.is_dir() or not step_dir.name.isdigit():
                    continue
                policy_file = step_dir / "PPO_POLICY.pt"
                if policy_file.exists():
                    label = run_dir.name
                    if base_name != "checkpoints":
                        label += f" ({base_name.replace('checkpoints_', '')})"
                    found.append({
                        "path": str(step_dir),
                        "step": int(step_dir.name),
                        "run": label,
                        "size_mb": round(policy_file.stat().st_size / 1024 / 1024, 1),
                    })
    found.sort(key=lambda x: x["step"], reverse=True)
    return jsonify(found)


@app.route("/logs", methods=["GET"])
def get_logs():
    last_n = request.args.get("n", 100, type=int)
    return jsonify(manager.get_logs(last_n))


@app.route("/metrics", methods=["GET"])
def get_metrics():
    """Return all metrics from the JSONL file."""
    if not METRICS_FILE.exists():
        return jsonify([])
    metrics = []
    with open(METRICS_FILE, "r") as f:
        for line in f:
            try:
                data = json.loads(line.strip())
                if data.get("type") == "metrics":
                    metrics.append(data)
            except json.JSONDecodeError:
                continue
    return jsonify(metrics)


@app.route("/config", methods=["GET"])
def get_config():
    if CONFIG_FILE.exists():
        with open(CONFIG_FILE) as f:
            return jsonify(json.load(f))
    return jsonify(None)


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "service": "training-server"})


# ─── Main ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.environ.get("TRAINING_SERVER_PORT", 9000))
    print(f"[TrainingServer] Bot directory: {BOT_DIR}")
    print(f"[TrainingServer] Python: {VENV_PYTHON}")
    print(f"[TrainingServer] Starting on http://0.0.0.0:{port}")
    print(f"[TrainingServer] Endpoints:")
    print(f"  GET  /status       — check if training is running")
    print(f"  POST /start        — start training (send config JSON)")
    print(f"  POST /stop         — stop training")
    print(f"  GET  /checkpoints  — list checkpoints")
    print(f"  GET  /logs?n=100   — get recent log lines")
    print(f"  GET  /metrics      — get all metrics")
    print()
    app.run(host="0.0.0.0", port=port, debug=False)