"""Expert Strategy: deterministic decision tree for 1v1, 2v2, and 3v3.

Decision logic:
- If kickoff → handled by kickoff module
- If ball threatening our goal → DEFEND_CLEAR or DEFEND_SHADOW
- In 3v3: check if a teammate is already closer → don't double commit
- If we're much closer to ball → CHALLENGE or SHOOT
- If opponent is closer → SHADOW, ROTATE, or support
- If low boost and safe → GRAB_BOOST_SAFE
"""

from __future__ import annotations

from src.types import GameState, Intent, StrategyOutput, Vec3, CarState
from src.utils import (
    distance_2d,
    estimate_intercept_time,
    is_ball_threatening_goal,
    shot_quality,
    FIELD_HALF_LENGTH,
)
from src import config


class ExpertStrategy:
    """Hand-crafted expert strategy for 1v1/2v2/3v3."""

    def __init__(self):
        self.aggression = config.get("expert.aggression", 0.5)
        self.challenge_dist = config.get("expert.challenge_distance", 2000.0)
        self.shadow_offset = config.get("expert.shadow_offset", 800.0)
        self.clear_power = config.get("expert.clear_power_threshold", 0.4)
        self.safe_boost_threshold = config.get("expert.safe_boost_threshold", 30)
        self.shot_quality_threshold = config.get("safety.shot_quality_threshold", 0.35)

    def decide(self, state: GameState) -> StrategyOutput:
        """Main entry: produce a strategy decision."""
        # --- 3v3 role assignment ---
        role = self._get_role(state)

        # If we're the rotator (3rd man), always rotate back
        if role == "rotate":
            return self._position_defensively(state)

        # If we're support (2nd man), play conservative mid
        if role == "support":
            return self._play_support(state)

        # We're the attacker (closest to ball or only player)
        # --- Priority 1: Ball threatening our goal ---
        if is_ball_threatening_goal(state.ball, state.own_goal, threshold_time=2.0):
            return self._defend(state)

        # --- Priority 2: We have clear time advantage ---
        time_advantage = state.opponent_time_to_ball - state.time_to_ball
        if time_advantage > 0.8:
            return self._attack(state, time_advantage)

        # --- Priority 3: Opponent has advantage ---
        if time_advantage < -0.3:
            return self._position_defensively(state)

        # --- Priority 4: Contested ---
        return self._contested(state, time_advantage)

    def _get_role(self, state: GameState) -> str:
        """Determine our role in 3v3: attacker, support, or rotate."""
        alive_teammates = [tm for tm in state.teammates if not tm.is_demolished]
        if not alive_teammates:
            return "attacker"

        my_dist = state.ball_distance
        teammates_closer = 0

        for tm in alive_teammates:
            tm_dist = distance_2d(tm.position, state.ball.position)
            if tm_dist < my_dist - 300:
                teammates_closer += 1

        if teammates_closer == 0:
            return "attacker"
        elif teammates_closer == 1:
            return "support"
        else:
            return "rotate"

    def _play_support(self, state: GameState) -> StrategyOutput:
        """2nd man: stay between ball and own goal, ready to challenge."""
        mid = Vec3(
            state.ball.position.x * 0.3,
            (state.ball.position.y + state.own_goal.y) * 0.5,
            0,
        )

        if state.time_to_ball < 1.0 and state.opponent_time_to_ball > 1.5:
            return self._attack(state, state.opponent_time_to_ball - state.time_to_ball)

        if is_ball_threatening_goal(state.ball, state.own_goal, threshold_time=1.5):
            return self._defend(state)

        return StrategyOutput(
            intent=Intent.DEFEND_SHADOW,
            target=mid,
            risk=0.2,
        )

    def _defend(self, state: GameState) -> StrategyOutput:
        if state.time_to_ball < state.opponent_time_to_ball + 0.5:
            return StrategyOutput(
                intent=Intent.DEFEND_CLEAR,
                target=state.ball.position,
                risk=0.3,
            )

        ball_to_goal = (state.own_goal - state.ball.position).flat().normalized()
        shadow_pos = state.ball.position + ball_to_goal * self.shadow_offset
        return StrategyOutput(
            intent=Intent.DEFEND_SHADOW,
            target=shadow_pos,
            risk=0.2,
        )

    def _attack(self, state: GameState, time_advantage: float) -> StrategyOutput:
        quality = shot_quality(
            state.car, state.ball, state.opp_goal, state.opponents
        )

        if quality > self.shot_quality_threshold + 0.1:
            return StrategyOutput(
                intent=Intent.ATTACK_SHOOT,
                target=state.ball.position,
                risk=min(0.8, 0.3 + self.aggression * 0.5),
            )

        ball_to_opp = (state.opp_goal - state.ball.position).flat().normalized()
        approach = state.ball.position - ball_to_opp * 200
        return StrategyOutput(
            intent=Intent.ATTACK_POSSESSION,
            target=approach,
            risk=0.4,
        )

    def _position_defensively(self, state: GameState) -> StrategyOutput:
        if state.car.boost < self.safe_boost_threshold:
            ball_dist_to_goal = state.ball_to_own_goal_dist
            if ball_dist_to_goal > 3000:
                return StrategyOutput(
                    intent=Intent.GRAB_BOOST_SAFE,
                    target=state.own_goal,
                    risk=0.2,
                )

        return StrategyOutput(
            intent=Intent.ROTATE_BACK,
            target=state.own_goal,
            risk=0.1,
        )

    def _contested(self, state: GameState, time_advantage: float) -> StrategyOutput:
        ball_in_opp_half = (
            (state.team.value == 0 and state.ball.position.y > 0)
            or (state.team.value == 1 and state.ball.position.y < 0)
        )

        if ball_in_opp_half and self.aggression > 0.4:
            return StrategyOutput(
                intent=Intent.CHALLENGE,
                target=state.ball.position,
                risk=0.5 + self.aggression * 0.3,
            )

        ball_to_goal = (state.own_goal - state.ball.position).flat().normalized()
        shadow_pos = state.ball.position + ball_to_goal * self.shadow_offset
        return StrategyOutput(
            intent=Intent.DEFEND_SHADOW,
            target=shadow_pos,
            risk=0.3,
        )
