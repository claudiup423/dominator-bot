# Competition & Deployment Guide

## Prerequisites

1. RLBot framework installed: https://rlbot.org
2. Rocket League running
3. Python 3.11+

## Running the Bot Locally

### Option A: RLBot GUI
1. Open RLBot GUI
2. Click "Add" → navigate to `rlbot_config/dominance_bot.cfg`
3. Select opponent (RLBot built-in bots or other bots)
4. Start match

### Option B: Command Line Testing
```bash
# Run pipeline smoke test (no Rocket League needed)
python -m scripts.run_bot --mode expert --ticks 1000

# Run with ML model
python -m scripts.run_bot --mode ml --model models/stable/strategy_model.pt
```

## Packaging for Tournament

```bash
python -m scripts.package_submission --version 1.0.0
```

This creates `dist/DominanceBot_v1.0.0.zip` containing:
- Bot source code
- Configuration files
- Model weights (if available)
- RLBot config entry point

## Tournament Submission Steps

### RLBot Community Tournaments
1. Package the bot: `python -m scripts.package_submission`
2. Upload the zip to the tournament submission form
3. Verify the bot loads by testing in RLBot GUI first
4. Entry point: `rlbot_config/dominance_bot.cfg`

### RLBot Ladder
1. Fork/clone this repo
2. Ensure `models/stable/strategy_model.pt` exists (or bot runs in expert mode)
3. Submit via the RLBot ladder platform

## Bot Modes

| Mode | When Used | Description |
|------|-----------|-------------|
| `expert` | Default, or if no ML model | Deterministic decision tree |
| `ml` | When trained model exists | Neural network strategy |
| `safe` | Fallback on any error | Ultra-conservative deterministic |

The bot automatically falls back:
- `ml` → `expert` if model fails to load
- `expert` → safe fallback if tick throws exception
- Any mode → `ControlOutput(throttle=0.3)` on catastrophic failure

## Model Versioning

Models are stored in `models/stable/`:
- `strategy_model.pt` — current production model
- Includes MD5 checksum for integrity validation
- Bot validates checksum on load; falls back to expert if mismatch

## Performance

Target: <2ms per tick on CPU (achieved by the expert; ML adds ~0.5ms).

Monitor performance:
```bash
python -m scripts.run_bot --ticks 5000
# Outputs average/max tick times
```

## Troubleshooting

### Bot doesn't move
- Check RLBot console for errors
- Verify `rlbot_config/dominance_bot.cfg` points to correct Python file
- Ensure Python 3.11+ is being used

### Model not loading
- Check `models/stable/strategy_model.pt` exists
- Check logs for checksum mismatch errors
- Bot will fall back to expert mode automatically

### High latency
- Disable ML mode: set `mode: expert` in `configs/bot_config.yaml`
- Check CPU usage — other processes may be competing
- Reduce opponent pool size for training

### Bot plays too passively
- Increase `expert.aggression` in config
- Lower `safety.shot_quality_threshold`
- Retrain with higher reward for scoring
