"""Safety Supervisor: enforces hard dominance rules that ML cannot override.

This is the critical safety layer. It sits between the TacticsPlanner and the
Controller, and can VETO or MODIFY any tactical plan.

Rules enforced:
1. Last Man Rule — forbid direct challenge when sole defender
2. Back Post Rule — defensive rotations go to back post
3. Shot Quality Gate — forbid low-quality shots
4. Boost Discipline — forbid corner boost detours when last man
5. Stability — catch NaN/invalid outputs, force fallback
6. No Free Goals — go conservative when one mistake = open net
"""

from __future__ import annotations

import math
from collections import deque
from typing import Optional

from src.types import (
    GameState,
    Intent,
    StrategyOutput,
    TacticalPlan,
    Vec3,
)
from src.utils import (
    back_post_position,
    distance_2d,
    is_ball_threatening_goal,
    shot_quality,
    time_to_reach,
    FIELD_HALF_LENGTH,
)
from src import config


class SafetySupervisor:
    """Vetoes or overrides tactical plans based on hard safety rules.

    Every rule returns either None (no override) or a modified TacticalPlan.
    Rules are checked in priority order; first override wins.
    """

    def __init__(self):
        # Config values (loaded once, can be reloaded)
        self._load_config()
        # Intent history for indecision detection
        self._intent_history: deque[Intent] = deque(maxlen=60)
        self._cooldown_ticks: int = 0

    def _load_config(self) -> None:
        self.shot_quality_threshold = config.get("safety.shot_quality_threshold", 0.35)
        self.last_man_ball_speed_min = config.get(
            "safety.last_man_challenge_ball_speed_min", 1500.0
        )
        self.boost_detour_max = config.get("safety.last_man_boost_detour_max_distance", 1500.0)
        self.time_margin = config.get("safety.last_man_time_margin", 1.5)
        self.indecision_window = config.get("safety.indecision_window_ticks", 15)
        self.indecision_max_switches = config.get("safety.indecision_max_switches", 3)
        self.indecision_cooldown = config.get("safety.indecision_cooldown_ticks", 30)
        self.open_net_threshold = config.get("safety.open_net_distance_threshold", 3000.0)
        self.back_post_offset = config.get("safety.back_post_offset_y", 400.0)

    def check(
        self,
        state: GameState,
        strategy: StrategyOutput,
        plan: TacticalPlan,
    ) -> TacticalPlan:
        """Run all safety checks. Returns possibly-modified plan."""
        # 1. Stability check (highest priority)
        override = self._check_stability(state, strategy, plan)
        if override:
            return override

        # 2. Indecision check
        override = self._check_indecision(state, strategy, plan)
        if override:
            return override

        # 3. No free goals
        override = self._check_no_free_goals(state, strategy, plan)
        if override:
            return override

        # 4. Last man rule
        override = self._check_last_man(state, strategy, plan)
        if override:
            return override

        # 5. Back post rule
        override = self._check_back_post(state, strategy, plan)
        if override:
            return override

        # 6. Shot quality gate
        override = self._check_shot_quality(state, strategy, plan)
        if override:
            return override

        # 7. Boost discipline
        override = self._check_boost_discipline(state, strategy, plan)
        if override:
            return override

        # Track intent history (after checks, so overrides are tracked)
        self._intent_history.append(strategy.intent)

        return plan

    # ----- Rule implementations -----

    def _check_stability(
        self, state: GameState, strategy: StrategyOutput, plan: TacticalPlan
    ) -> Optional[TacticalPlan]:
        """Catch NaN/inf or invalid intents → force safe rotation."""
        # Check for NaN in target
        t = plan.target_position
        if math.isnan(t.x) or math.isnan(t.y) or math.isnan(t.z):
            return self._safe_rotate_plan(state, "NaN in target position")
        if math.isinf(t.x) or math.isinf(t.y) or math.isinf(t.z):
            return self._safe_rotate_plan(state, "Inf in target position")

        # Check for invalid intent value
        try:
            Intent(strategy.intent)
        except ValueError:
            return self._safe_rotate_plan(state, f"Invalid intent: {strategy.intent}")

        return None

    def _check_indecision(
        self, state: GameState, strategy: StrategyOutput, plan: TacticalPlan
    ) -> Optional[TacticalPlan]:
        """Detect rapid intent switching and force cooldown."""
        if self._cooldown_ticks > 0:
            self._cooldown_ticks -= 1
            return self._safe_rotate_plan(state, "Indecision cooldown active")

        self._intent_history.append(strategy.intent)

        # Count switches in recent window
        if len(self._intent_history) >= self.indecision_window:
            recent = list(self._intent_history)[-self.indecision_window:]
            switches = sum(1 for i in range(1, len(recent)) if recent[i] != recent[i - 1])
            if switches >= self.indecision_max_switches:
                self._cooldown_ticks = self.indecision_cooldown
                return self._safe_rotate_plan(state, "Indecision detected: too many intent switches")

        return None

    def _check_no_free_goals(
        self, state: GameState, strategy: StrategyOutput, plan: TacticalPlan
    ) -> Optional[TacticalPlan]:
        """If opponent is close to our open net, force conservative play."""
        if not state.opponents:
            return None

        # Check if any opponent is dangerously close to our goal
        for opp in state.opponents:
            opp_to_goal = distance_2d(opp.position, state.own_goal)
            if opp_to_goal < self.open_net_threshold:
                # And the ball is between them and the goal
                ball_to_goal = distance_2d(state.ball.position, state.own_goal)
                if ball_to_goal < opp_to_goal + 500:
                    # And we're not already defending
                    if strategy.intent not in (Intent.DEFEND_SHADOW, Intent.DEFEND_CLEAR, Intent.ROTATE_BACK):
                        return self._safe_rotate_plan(
                            state, "No free goals: opponent near open net"
                        )

        return None

    def _check_last_man(
        self, state: GameState, strategy: StrategyOutput, plan: TacticalPlan
    ) -> Optional[TacticalPlan]:
        """Last man must not challenge unless ball is immediately threatening."""
        if not state.is_last_man:
            return None

        # Allow defensive actions
        if strategy.intent in (Intent.DEFEND_SHADOW, Intent.DEFEND_CLEAR, Intent.ROTATE_BACK):
            return None

        # Allow challenge only if ball is threatening our goal
        ball_threatening = is_ball_threatening_goal(
            state.ball, state.own_goal, threshold_time=self.time_margin
        )

        if strategy.intent == Intent.CHALLENGE and not ball_threatening:
            # Ball speed toward goal check
            to_goal = state.own_goal - state.ball.position
            ball_speed_toward = state.ball.velocity.dot(to_goal.normalized())
            if ball_speed_toward < self.last_man_ball_speed_min:
                return self._shadow_plan(state, "Last man: forbid challenge, not threatening")

        # Forbid attacking as last man unless we have a huge time advantage
        if strategy.intent in (Intent.ATTACK_SHOOT, Intent.ATTACK_POSSESSION):
            time_advantage = state.opponent_time_to_ball - state.time_to_ball
            if time_advantage < 1.0:  # Need at least 1s advantage
                return self._shadow_plan(state, "Last man: forbid attack, insufficient time advantage")

        return None

    def _check_back_post(
        self, state: GameState, strategy: StrategyOutput, plan: TacticalPlan
    ) -> Optional[TacticalPlan]:
        """Enforce back-post positioning when rotating to defend."""
        if strategy.intent not in (Intent.DEFEND_SHADOW, Intent.ROTATE_BACK):
            return None

        # Compute proper back-post position
        bp = back_post_position(state.own_goal, state.ball.position, self.back_post_offset)

        # Check if the plan's target is on the wrong post
        current_target = plan.target_position
        target_to_bp = distance_2d(current_target, bp)

        # If the plan target is far from back post, override
        if target_to_bp > 800:
            modified = TacticalPlan(
                target_position=bp,
                target_speed=plan.target_speed,
                face_target=state.ball.position,  # Always watch the ball when defending
                use_boost=plan.use_boost,
                source_intent=strategy.intent,
                was_overridden=True,
                override_reason="Back post: corrected rotation target",
            )
            return modified

        return None

    def _check_shot_quality(
        self, state: GameState, strategy: StrategyOutput, plan: TacticalPlan
    ) -> Optional[TacticalPlan]:
        """Block low-quality shots — prefer possession or safe clear."""
        if strategy.intent != Intent.ATTACK_SHOOT:
            return None

        quality = shot_quality(
            state.car, state.ball, state.opp_goal, state.opponents
        )

        if quality < self.shot_quality_threshold:
            # Convert to possession touch instead
            modified = TacticalPlan(
                target_position=plan.target_position,
                target_speed=plan.target_speed,
                face_target=None,  # Don't aim at goal
                use_boost=False,
                source_intent=Intent.ATTACK_POSSESSION,
                was_overridden=True,
                override_reason=f"Shot quality too low: {quality:.2f} < {self.shot_quality_threshold}",
            )
            return modified

        return None

    def _check_boost_discipline(
        self, state: GameState, strategy: StrategyOutput, plan: TacticalPlan
    ) -> Optional[TacticalPlan]:
        """Forbid corner boost detours when last man."""
        if not state.is_last_man:
            return None
        if strategy.intent != Intent.GRAB_BOOST_SAFE:
            return None

        # Check if the boost target is a dangerous detour
        car_to_goal = distance_2d(state.car.position, state.own_goal)
        car_to_boost = distance_2d(state.car.position, plan.target_position)
        boost_to_goal = distance_2d(plan.target_position, state.own_goal)

        # The detour distance: going via boost vs going directly to goal
        detour = (car_to_boost + boost_to_goal) - car_to_goal

        if detour > self.boost_detour_max:
            return self._safe_rotate_plan(
                state,
                f"Boost discipline: detour too large ({detour:.0f} > {self.boost_detour_max:.0f})",
            )

        # Check if we can return in time
        return_time = time_to_reach(state.car, state.own_goal, use_boost=True)
        threat_time = state.opponent_time_to_ball + 1.0  # rough estimate
        if return_time > threat_time - self.time_margin:
            return self._safe_rotate_plan(
                state, "Boost discipline: cannot return in time"
            )

        return None

    # ----- Plan generators -----

    def _safe_rotate_plan(self, state: GameState, reason: str) -> TacticalPlan:
        """Generate a safe rotation plan — go to back post."""
        bp = back_post_position(state.own_goal, state.ball.position, self.back_post_offset)
        return TacticalPlan(
            target_position=bp,
            target_speed=2300.0,
            face_target=state.ball.position,
            use_boost=state.car.boost > 20,
            source_intent=Intent.ROTATE_BACK,
            was_overridden=True,
            override_reason=reason,
        )

    def _shadow_plan(self, state: GameState, reason: str) -> TacticalPlan:
        """Generate a shadow defense plan — stay between ball and goal."""
        shadow_offset = 800.0
        ball_to_goal = (state.own_goal - state.ball.position).flat().normalized()
        shadow_pos = state.ball.position + ball_to_goal * shadow_offset
        return TacticalPlan(
            target_position=shadow_pos,
            target_speed=min(2300.0, state.ball.velocity.length() + 500),
            face_target=state.ball.position,
            use_boost=state.car.boost > 30,
            source_intent=Intent.DEFEND_SHADOW,
            was_overridden=True,
            override_reason=reason,
        )
