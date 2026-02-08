"""Launch an RLBot match with the PPO bot.

Usage:
    python run.py              # 1v1 vs Psyonix AllStar
    python run.py --3v3        # 3v3 (add more bots to rlbot.toml)
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from rlbot import runner

if __name__ == "__main__":
    runner.main()
