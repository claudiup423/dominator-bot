# DominanceBot — Tournament-Ready Rocket League Bot

A production-grade, hierarchical Rocket League bot built on RLBot + rlgym-sim.
Designed for **dominance-oriented play** (~90% win rate target) with hard safety
overrides, deterministic low-level controllers, and an ML strategy layer.

## Architecture

```
Tick Pipeline:
  GamePacket → StateBuilder → StrategyPolicy → TacticsPlanner → SafetySupervisor → Controller → Controls
                                (ML or Expert)    (plan)           (veto/override)    (deterministic)
```

The ML policy **only** outputs high-level decisions (intent, target position, risk scalar).
All mechanical execution is handled by deterministic, tuned controllers.
A `SafetySupervisor` enforces hard dominance rules that **cannot** be overridden by ML.

## Quick Start

### Prerequisites
- Python 3.11+
- RLBot framework installed (for live matches)
- rlgym-sim (for training)

### Install
```bash
git clone <this-repo>
cd rl-bot
pip install -e ".[dev]"
```

### Run the bot in RLBot
```bash
# Start RLBot framework, then add the bot via:
# RLBot GUI → Add → path to rlbot_config/dominance_bot.cfg
# Or use the RLBot runner:
python -m scripts.run_bot
```

### Training Pipeline
```bash
# Phase 1: Run deterministic expert (no training needed)
python -m scripts.evaluate --bot expert --opponent allstar --matches 20

# Phase 2: Generate imitation dataset + train BC
python -m scripts.generate_dataset --episodes 5000
python -m scripts.train_bc --epochs 50

# Phase 3: PPO reinforcement learning
python -m scripts.train_ppo --timesteps 50_000_000

# Phase 4: League training
python -m scripts.train_league --rounds 100
```

### Evaluate
```bash
python -m scripts.evaluate --bot ml --opponent expert --matches 50
```

### Package for tournament
```bash
python -m scripts.package_submission
# Output: dist/DominanceBot_v<VERSION>.zip
```

## Project Structure
```
rl-bot/
├── src/
│   ├── state/         # Feature extraction, game state representation
│   ├── strategy/      # ML policy + expert policy (intent/target/risk)
│   ├── tactics/       # Tactical planner (converts intent→plan)
│   ├── control/       # Deterministic controllers (drive, aerial, dodge)
│   ├── safety/        # SafetySupervisor — hard dominance rules
│   ├── training/      # BC, PPO, league training pipelines
│   ├── eval/          # Arena, metrics, regression gates
│   ├── kickoff/       # Kickoff-specific logic
│   └── utils/         # Physics, math, geometry helpers
├── configs/           # YAML configs for training, eval, bot params
├── models/            # Trained model checkpoints
│   └── stable/        # Currently deployed "stable" model
├── rlbot_config/      # RLBot package (cfg, appearance, entrypoint)
├── scripts/           # CLI scripts (train, eval, package, run)
├── docker/            # Dockerfile for reproducible training
├── docs/              # Guides: architecture, training, deployment
└── tests/             # Unit + integration tests
```

## Dominance Rules (Hard-Coded Safety)
These rules are enforced by `SafetySupervisor` and **cannot** be overridden by ML:

| Rule | Description |
|------|-------------|
| Last Man | If sole defender, forbid direct challenge unless imminent threat |
| Back Post | Defensive rotations target back-post, never front-post |
| Shot Quality | Block low-quality shots (bad angle/covered net) |
| Boost Discipline | Last man cannot detour for corner boost |
| Stability | NaN/invalid outputs → instant fallback to deterministic expert |
| No Free Goals | Conservative play when single mistake = open net |

## Intents
The strategy layer outputs one of these intents per tick:
- `DEFEND_SHADOW` — Shadow the ball carrier defensively
- `DEFEND_CLEAR` — Clear the ball from danger
- `CHALLENGE` — Commit to a 50/50 or contest
- `ATTACK_SHOOT` — Take a shot on goal
- `ATTACK_POSSESSION` — Keep/gain possession safely
- `ROTATE_BACK` — Rotate to defensive position
- `GRAB_BOOST_SAFE` — Collect boost on a safe path

## License
MIT
