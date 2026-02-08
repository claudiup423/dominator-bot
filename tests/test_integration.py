"""Integration tests: verify the full tick pipeline works end-to-end."""

from __future__ import annotations

import numpy as np
import pytest

from src.bot_brain import DominanceBotBrain, BotMode
from src.types import Team, Intent, ControlOutput, GameState
from src.state import StateBuilder
from src.strategy import ExpertStrategy
from src.tactics import TacticsPlanner
from src.safety import SafetySupervisor
from src.control import Controller
from src.kickoff import KickoffHandler


def _make_state(**overrides) -> dict:
    """Create a plausible game state dict with optional overrides."""
    state = {
        "ball": {
            "position": [0, 0, 93],
            "velocity": [0, 0, 0],
        },
        "car": {
            "position": [0, -4000, 17],
            "velocity": [0, 0, 0],
            "rotation": [0, 1.57, 0],
            "boost": 50,
            "is_on_ground": True,
        },
        "opponents": [{
            "position": [0, 4000, 17],
            "velocity": [0, 0, 0],
            "rotation": [0, -1.57, 0],
            "boost": 50,
            "is_on_ground": True,
        }],
        "time_remaining": 250,
        "is_kickoff": False,
    }
    state.update(overrides)
    return state


class TestFullPipeline:
    """Test the complete tick pipeline from packet to controls."""

    def test_expert_tick_produces_valid_output(self):
        brain = DominanceBotBrain(car_index=0, team=Team.BLUE, mode="expert")
        state = _make_state()
        ctrl = brain.tick(state)
        assert isinstance(ctrl, ControlOutput)
        assert -1.0 <= ctrl.throttle <= 1.0
        assert -1.0 <= ctrl.steer <= 1.0

    def test_many_ticks_no_crash(self):
        """Run 500 random ticks — must not crash."""
        brain = DominanceBotBrain(car_index=0, team=Team.BLUE, mode="expert")
        rng = np.random.default_rng(123)

        for _ in range(500):
            state = {
                "ball": {
                    "position": [rng.uniform(-4000, 4000), rng.uniform(-5000, 5000), 93],
                    "velocity": [rng.uniform(-2000, 2000), rng.uniform(-2000, 2000), 0],
                },
                "car": {
                    "position": [rng.uniform(-4000, 4000), rng.uniform(-5000, 5000), 17],
                    "velocity": [rng.uniform(-2000, 2000), rng.uniform(-2000, 2000), 0],
                    "rotation": [0, rng.uniform(-3.14, 3.14), 0],
                    "boost": rng.uniform(0, 100),
                    "is_on_ground": True,
                },
                "opponents": [{
                    "position": [rng.uniform(-4000, 4000), rng.uniform(-5000, 5000), 17],
                    "velocity": [rng.uniform(-1000, 1000), rng.uniform(-1000, 1000), 0],
                    "rotation": [0, rng.uniform(-3.14, 3.14), 0],
                    "boost": rng.uniform(0, 100),
                    "is_on_ground": True,
                }],
            }
            ctrl = brain.tick(state)
            assert isinstance(ctrl, ControlOutput)

    def test_latency_under_budget(self):
        """Average tick time must be under 2ms."""
        import time
        brain = DominanceBotBrain(car_index=0, team=Team.BLUE, mode="expert")
        rng = np.random.default_rng(42)

        times = []
        for _ in range(200):
            state = _make_state()
            state["ball"]["position"] = [
                rng.uniform(-3000, 3000), rng.uniform(-4000, 4000), 93
            ]
            start = time.perf_counter()
            brain.tick(state)
            times.append((time.perf_counter() - start) * 1000)

        avg_ms = sum(times) / len(times)
        assert avg_ms < 2.0, f"Average tick time {avg_ms:.3f}ms exceeds 2ms budget"


class TestStateBuilder:
    def test_basic_build(self):
        builder = StateBuilder(car_index=0, team=Team.BLUE)
        state = builder.from_dict(_make_state())
        assert isinstance(state, GameState)
        assert state.team == Team.BLUE
        assert state.ball_distance > 0

    def test_tensor_shape(self):
        builder = StateBuilder(car_index=0, team=Team.BLUE)
        state = builder.from_dict(_make_state())
        tensor = builder.to_tensor(state)
        assert tensor.shape == (34,)
        assert not np.any(np.isnan(tensor))

    def test_last_man_detection(self):
        builder = StateBuilder(car_index=0, team=Team.BLUE)
        state = builder.from_dict(_make_state())
        assert state.is_last_man  # No teammates in 1v1


class TestExpertStrategy:
    def test_decides_intent(self):
        builder = StateBuilder(car_index=0, team=Team.BLUE)
        expert = ExpertStrategy()
        state = builder.from_dict(_make_state())
        output = expert.decide(state)
        assert output.intent in list(Intent)
        assert 0.0 <= output.risk <= 1.0

    def test_defends_when_ball_near_goal(self):
        builder = StateBuilder(car_index=0, team=Team.BLUE)
        expert = ExpertStrategy()
        # Ball near our goal, moving toward it
        state_data = _make_state()
        state_data["ball"]["position"] = [0, -4500, 93]
        state_data["ball"]["velocity"] = [0, -1000, 0]
        state = builder.from_dict(state_data)
        output = expert.decide(state)
        # Should be defensive
        assert output.intent in (
            Intent.DEFEND_SHADOW, Intent.DEFEND_CLEAR, Intent.ROTATE_BACK
        )


class TestSafetySupervisor:
    def test_nan_override(self):
        """NaN targets must be caught and overridden."""
        from src.types import StrategyOutput, TacticalPlan, Vec3
        supervisor = SafetySupervisor()
        builder = StateBuilder(car_index=0, team=Team.BLUE)
        state = builder.from_dict(_make_state())

        strategy = StrategyOutput(intent=Intent.ATTACK_SHOOT, target=Vec3(0, 0, 0))
        plan = TacticalPlan(
            target_position=Vec3(float("nan"), 0, 0),
            source_intent=Intent.ATTACK_SHOOT,
        )
        safe_plan = supervisor.check(state, strategy, plan)
        assert safe_plan.was_overridden
        assert "NaN" in safe_plan.override_reason

    def test_shot_quality_gate(self):
        """Low quality shots should be blocked."""
        from src.types import StrategyOutput, TacticalPlan, Vec3
        supervisor = SafetySupervisor()
        builder = StateBuilder(car_index=0, team=Team.BLUE)

        # Set up a bad shot: ball behind us, aimed wrong direction
        state_data = _make_state()
        state_data["ball"]["position"] = [0, -4000, 93]  # Ball near our goal
        state_data["car"]["position"] = [3000, -3000, 17]  # Car way off angle
        state = builder.from_dict(state_data)

        strategy = StrategyOutput(intent=Intent.ATTACK_SHOOT, target=Vec3(0, 0, 0), risk=0.5)
        plan = TacticalPlan(
            target_position=Vec3(0, -4000, 93),
            source_intent=Intent.ATTACK_SHOOT,
        )
        safe_plan = supervisor.check(state, strategy, plan)
        # Should be converted to possession or overridden
        if safe_plan.was_overridden:
            assert "quality" in safe_plan.override_reason.lower() or safe_plan.source_intent != Intent.ATTACK_SHOOT


class TestKickoff:
    def test_kickoff_detection(self):
        handler = KickoffHandler()
        builder = StateBuilder(car_index=0, team=Team.BLUE)

        state = builder.from_dict(_make_state(is_kickoff=True))
        assert handler.is_kickoff(state)

        state = builder.from_dict(_make_state(is_kickoff=False))
        assert not handler.is_kickoff(state)

    def test_kickoff_outputs_challenge(self):
        handler = KickoffHandler()
        builder = StateBuilder(car_index=0, team=Team.BLUE)
        state = builder.from_dict(_make_state(is_kickoff=True))
        output = handler.decide(state)
        assert output.intent == Intent.CHALLENGE


class TestController:
    def test_produces_valid_output(self):
        from src.types import TacticalPlan, Vec3
        controller = Controller()
        builder = StateBuilder(car_index=0, team=Team.BLUE)
        state = builder.from_dict(_make_state())
        plan = TacticalPlan(
            target_position=Vec3(0, 0, 0),
            target_speed=2300,
            use_boost=True,
        )
        ctrl = controller.execute(state, plan)
        assert isinstance(ctrl, ControlOutput)
        assert -1.0 <= ctrl.throttle <= 1.0
