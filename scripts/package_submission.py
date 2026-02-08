"""Package the bot for RLBot v5 tournament submission.

Tournament requires a zip containing:
- Source code
- bot.toml
- requirements.txt

Usage:
    python -m scripts.package_submission
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

INCLUDE = [
    "src/",
    "configs/",
    "models/stable/",
    "requirements.txt",
    "README.md",
]

EXCLUDE = [
    "__pycache__",
    "*.pyc",
    ".git",
    "data/",
    "eval_results/",
    "models/bc/",
    "models/ppo/",
    "models/league/",
    "models/opponent_pool/",
    "docker/",
    "tests/",
    ".ruff_cache",
    ".pytest_cache",
    "venv/",
    "dist/",
]


def main():
    parser = argparse.ArgumentParser(description="Package bot for tournament")
    parser.add_argument("--version", type=str, default="0.2.0")
    parser.add_argument("--output", type=str, default="dist")
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent
    output_dir = project_root / args.output
    output_dir.mkdir(parents=True, exist_ok=True)

    bot_name = f"DominanceBot_v{args.version}"
    staging_dir = output_dir / bot_name
    if staging_dir.exists():
        shutil.rmtree(staging_dir)
    staging_dir.mkdir()

    # Copy included files
    for include_path in INCLUDE:
        src_path = project_root / include_path
        if not src_path.exists():
            logger.warning(f"Skipping missing: {include_path}")
            continue
        dst = staging_dir / include_path
        if src_path.is_dir():
            shutil.copytree(src_path, dst, ignore=shutil.ignore_patterns(*EXCLUDE))
        else:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_path, dst)

    # Verify required tournament files exist
    bot_toml = staging_dir / "src" / "bot.toml"
    req_txt = staging_dir / "requirements.txt"

    if bot_toml.exists():
        logger.info(f"✓ bot.toml present")
    else:
        logger.error("✗ bot.toml MISSING — tournament will reject this")

    if req_txt.exists():
        logger.info(f"✓ requirements.txt present")
    else:
        logger.error("✗ requirements.txt MISSING")

    model_path = staging_dir / "models" / "stable" / "strategy_model.pt"
    if model_path.exists():
        logger.info(f"✓ ML model included")
    else:
        logger.warning("⚠ No ML model — bot will run in expert (deterministic) mode")

    # Create zip
    zip_path = output_dir / bot_name
    shutil.make_archive(str(zip_path), "zip", str(staging_dir))
    shutil.rmtree(staging_dir)

    final_zip = f"{zip_path}.zip"
    size_mb = Path(final_zip).stat().st_size / (1024 * 1024)
    logger.info(f"✓ Package created: {final_zip} ({size_mb:.1f} MB)")
    logger.info(f"  Submit this zip to @goosefairy on Discord")


if __name__ == "__main__":
    main()
