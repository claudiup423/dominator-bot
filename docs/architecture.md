# Architecture Overview

## Design Philosophy

DominanceBot uses a **hierarchical pipeline** where ML only controls high-level
strategy while all low-level mechanics are deterministic. This provides:

1. **Safety**: Hard rules cannot be bypassed by ML exploration
2. **Debuggability**: Each layer's decisions are inspectable
3. **Modularity**: Layers can be developed and tested independently
4. **Extensibility**: Adding 2v2/3v3 support requires changes mainly in strategy

## Tick Pipeline

```
┌─────────────┐     ┌──────────────┐     ┌──────────────┐
│ GamePacket   │────▶│ StateBuilder │────▶│  GameState   │
│ (RLBot/Sim) │     │              │     │  (structured)│
└─────────────┘     └──────────────┘     └──────┬───────┘
                                                │
                    ┌──────────────┐             │
                    │   Kickoff    │◄────────────┤ (if kickoff)
                    │   Handler    │             │
                    └──────┬───────┘             │
                           │              ┌──────▼───────┐
                           │              │   Strategy   │
                           │              │ (Expert/ML)  │
                           │              └──────┬───────┘
                           │                     │
                           │    StrategyOutput: Intent + Target + Risk
                           │                     │
                           ▼              ┌──────▼───────┐
                    ┌──────────────┐      │   Tactics    │
                    │              │      │   Planner    │
                    │              │      └──────┬───────┘
                    │              │             │
                    │              │    TacticalPlan: position, speed, mode
                    │              │             │
                    │              │      ┌──────▼───────┐
                    │              │      │   Safety     │
                    │   Recovery   │      │  Supervisor  │
                    │  Controller  │      └──────┬───────┘
                    │              │             │
                    │              │    SafePlan (possibly overridden)
                    │              │             │
                    │              │      ┌──────▼───────┐
                    └──────┬───────┘      │  Controller  │
                           │              │ (deterministic)│
                           │              └──────┬───────┘
                           │                     │
                           ▼              ControlOutput: throttle, steer, ...
                    ┌──────────────┐             │
                    │   RLBot      │◄────────────┘
                    │   Output     │
                    └──────────────┘
```

## Layers

### StateBuilder (`src/state/`)
- Converts raw game data (RLBot packets or rlgym obs) to structured `GameState`
- Computes derived fields: distances, time-to-ball, last-man detection
- Exports feature tensors for ML policy

### Strategy (`src/strategy/`)
- **ExpertStrategy**: Hand-crafted decision tree (the "boring lethal" baseline)
- **MLStrategy**: Neural network trained via BC then PPO
- Outputs: `Intent` (enum), `Target` (Vec3), `Risk` (float 0-1)
- ML can be swapped without touching any other layer

### Tactics (`src/tactics/`)
- Converts abstract intent into concrete plan
- Computes intercept points, positioning targets, approach angles
- Each intent has a dedicated planner function

### Safety Supervisor (`src/safety/`)
- Enforces hard dominance rules (cannot be overridden by ML)
- Checks every plan before execution
- Rules: last man, back post, shot quality gate, boost discipline, stability, no free goals

### Controller (`src/control/`)
- Pure deterministic: PD steering, throttle management, boost gating
- RecoveryController handles awkward states (airborne, upside down)
- Never controlled by ML directly

## Intent Set

| Intent | When Used | Tactical Effect |
|--------|-----------|-----------------|
| DEFEND_SHADOW | Opponent has ball, we shadow | Stay between ball and goal |
| DEFEND_CLEAR | Ball threatening goal, we can reach it | Intercept + hit away from goal |
| CHALLENGE | Contested ball, we go for 50/50 | Drive at ball contact point |
| ATTACK_SHOOT | Good shot opportunity | Intercept + aim at goal |
| ATTACK_POSSESSION | Have time but bad shot | Approach ball from behind |
| ROTATE_BACK | Need to retreat | Go to back-post position |
| GRAB_BOOST_SAFE | Low boost, safe to detour | Find nearest safe boost pad |

## Extending to 2v2 / 3v3

Key changes needed:
1. **State**: Add teammate awareness, team formations
2. **Strategy**: Add team-level intents (SUPPORT, ROTATE_MID, etc.)
3. **Safety**: Add double-commit detection, rotation ordering
4. **Tactics**: Add passing targets, team positioning

The architecture isolates these changes to specific layers.
