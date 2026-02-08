# Training Guide

## Overview

Training proceeds in four phases:

1. **Deterministic Expert** — hand-crafted, no training needed
2. **Behavior Cloning (BC)** — learn to mimic the expert
3. **PPO Reinforcement Learning** — fine-tune with rewards
4. **League Training** — train against diverse opponent pool

## Phase 1: Deterministic Expert

The expert (`src/strategy/__init__.py`) is a hand-crafted decision tree:

- **If ball threatening goal** → DEFEND_CLEAR or DEFEND_SHADOW
- **If we're much closer to ball** → ATTACK_SHOOT (if good quality) or ATTACK_POSSESSION
- **If opponent closer** → ROTATE_BACK or GRAB_BOOST_SAFE
- **If contested** → CHALLENGE (if in their half) or DEFEND_SHADOW

Tunable parameters in `configs/bot_config.yaml` under `expert:`:
- `aggression`: 0-1 scale for risk tolerance
- `challenge_distance`: Max distance to attempt a challenge
- `shadow_offset`: Distance behind ball for shadow defense

Run the expert:
```bash
python -m scripts.run_bot --mode expert --ticks 1000
python -m scripts.evaluate --bot expert --matches 20
```

## Phase 2: Behavior Cloning

### Generate Dataset
```bash
python -m scripts.generate_dataset --episodes 5000 --output data/bc_dataset
```

This runs the expert through randomized game states and logs:
- `obs.npy`: Observation tensors (N × 34)
- `intents.npy`: Expert intent labels (N,)
- `targets.npy`: Expert target positions (N × 3)
- `risks.npy`: Expert risk scalars (N,)

### Train
```bash
python -m scripts.train_bc --dataset data/bc_dataset --epochs 50 --output models/bc
```

Loss function:
- Cross-entropy for intent classification
- MSE × 5.0 for target regression
- MSE for risk regression

### Evaluate
```bash
python -m scripts.evaluate --bot ml --model models/bc/strategy_model.pt --matches 50
```

Expected: ML policy should match expert intent accuracy >85% and produce
similar win rates.

## Phase 3: PPO Reinforcement Learning

### Prerequisites
- rlgym-sim installed
- BC checkpoint as starting point

### Training
```bash
python -m scripts.train_ppo --timesteps 50000000 --resume models/bc/strategy_model.pt
```

### Reward Design

| Event | Reward | Rationale |
|-------|--------|-----------|
| Goal scored | +10.0 | Primary objective |
| Goal conceded | -15.0 | Punish more than reward (dominance) |
| Save | +3.0 | Encourage defensive solidity |
| Shot on goal | +1.5 | Reward threatening plays |
| Clear | +1.0 | Reward removing danger |
| Open-net conceded | -20.0 | Severely punish preventable goals |
| Overcommit | -2.0 | Discourage reckless play |
| Boost waste | -0.05/unit | Discourage wasteful boost usage |
| Good rotation | +0.2 | Reward positional discipline |

### Curriculum
Start with no aerial/jump mechanics (disabled in config). After 10M timesteps,
enable basic jumps. After 25M, enable dodge mechanics.

### Hyperparameters
See `configs/bot_config.yaml` under `training.ppo:`:
- `learning_rate: 0.0003`
- `gamma: 0.99`
- `clip_range: 0.2`
- `n_steps: 4096` (rollout buffer size)
- `n_epochs: 4` (PPO epochs per update)

### Resuming
```bash
python -m scripts.train_ppo --resume models/ppo/strategy_model.pt
```

## Phase 4: League Training

### Setup
```bash
python -m scripts.train_league --rounds 100
```

### How It Works
1. Maintains a pool of opponent checkpoints + scripted opponents
2. Each round samples an opponent
3. Trains against that opponent for N matches
4. Updates ELO ratings
5. Promotes current model to pool if win rate > threshold

### Scripted Opponents
- `expert_conservative` (aggression=0.3): passive, never overcommits
- `expert_aggressive` (aggression=0.8): pushes hard, takes risks
- `expert_balanced` (aggression=0.5): default expert behavior

### Preventing Exploit-Overfitting
- Diverse opponent pool prevents specializing against one style
- ELO tracking identifies weak opponents to avoid over-training against
- Pool size limit forces pruning of old checkpoints

## Regression Gates

After training, always evaluate:
```bash
python -m scripts.evaluate --bot ml --model models/ppo/strategy_model.pt \
    --matches 50 --baseline eval_results/baseline.json
```

Gates (configurable in `configs/bot_config.yaml`):
- Goals conceded must not increase by more than 0.1/game
- Open-net conceded must not increase by more than 0.05/game
- Win rate must not decrease by more than 3%

If gates fail, the model is rejected and the previous "stable" model is kept.

## Promoting to Stable

When a model passes all regression gates:
```bash
cp models/ppo/strategy_model.pt models/stable/strategy_model.pt
```

The bot will automatically load the stable model at runtime.
