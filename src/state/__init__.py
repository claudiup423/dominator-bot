"""State builder: converts raw game data into a structured GameState.

This module is the **single point of entry** for game data. Everything downstream
operates on GameState, never on raw packets. This makes it easy to swap between
RLBot packets (live) and rlgym observations (training).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from src.types import (
    BallState,
    BoostPad,
    CarState,
    GameState,
    Rotator,
    Team,
    Vec3,
)
from src.utils import (
    car_forward_vector,
    distance_2d,
    time_to_reach,
    FIELD_HALF_LENGTH,
    GOAL_HALF_WIDTH,
)


# Goal center positions (Y axis is forward/back in standard RL coordinates)
BLUE_GOAL = Vec3(0, -FIELD_HALF_LENGTH, 0)
ORANGE_GOAL = Vec3(0, FIELD_HALF_LENGTH, 0)


class StateBuilder:
    """Builds a GameState from raw game data.

    Supports two modes:
    - from_rlbot_packet(): for live RLBot games
    - from_rlgym_obs(): for training in rlgym-sim
    """

    def __init__(self, car_index: int = 0, team: Team = Team.BLUE):
        self.car_index = car_index
        self.team = team
        self.tick = 0

    def from_dict(self, data: dict[str, Any]) -> GameState:
        """Build GameState from a generic dictionary.

        This is the universal entry point. Both RLBot packets and rlgym obs
        are first converted to a common dict format, then processed here.
        """
        state = GameState()
        state.tick = self.tick
        self.tick += 1

        # Ball
        ball_data = data.get("ball", {})
        state.ball = BallState(
            position=_vec3(ball_data.get("position", [0, 0, 0])),
            velocity=_vec3(ball_data.get("velocity", [0, 0, 0])),
            angular_velocity=_vec3(ball_data.get("angular_velocity", [0, 0, 0])),
        )

        # Our car
        car_data = data.get("car", {})
        state.car = _build_car(car_data, self.team, self.car_index)
        state.team = self.team

        # Opponents
        state.opponents = [
            _build_car(opp, Team.ORANGE if self.team == Team.BLUE else Team.BLUE, i)
            for i, opp in enumerate(data.get("opponents", []))
        ]

        # Teammates (empty in 1v1)
        state.teammates = [
            _build_car(tm, self.team, i)
            for i, tm in enumerate(data.get("teammates", []))
        ]

        # Boost pads
        state.boost_pads = [
            BoostPad(
                position=_vec3(bp.get("position", [0, 0, 0])),
                is_big=bp.get("is_big", False),
                is_active=bp.get("is_active", True),
                timer=bp.get("timer", 0.0),
            )
            for bp in data.get("boost_pads", [])
        ]

        # Game info
        state.time_remaining = data.get("time_remaining", 300.0)
        state.score_us = data.get("score_us", 0)
        state.score_them = data.get("score_them", 0)
        state.is_kickoff = data.get("is_kickoff", False)

        # Compute derived fields
        self._compute_derived(state)

        return state

    def _compute_derived(self, state: GameState) -> None:
        """Populate derived convenience fields on the state."""
        # Goal positions based on team
        if state.team == Team.BLUE:
            state.own_goal = BLUE_GOAL
            state.opp_goal = ORANGE_GOAL
        else:
            state.own_goal = ORANGE_GOAL
            state.opp_goal = BLUE_GOAL

        # Distances
        state.ball_distance = distance_2d(state.car.position, state.ball.position)
        diff = state.ball.position - state.car.position
        state.ball_direction = diff.normalized()
        state.ball_to_own_goal_dist = distance_2d(state.ball.position, state.own_goal)
        state.ball_to_opp_goal_dist = distance_2d(state.ball.position, state.opp_goal)

        # Time to ball estimates
        state.time_to_ball = time_to_reach(state.car, state.ball.position)

        # Fastest opponent to ball
        if state.opponents:
            state.opponent_time_to_ball = min(
                time_to_reach(opp, state.ball.position) for opp in state.opponents
            )
            state.closest_opponent_to_ball_dist = min(
                distance_2d(opp.position, state.ball.position) for opp in state.opponents
            )
        else:
            state.opponent_time_to_ball = float("inf")
            state.closest_opponent_to_ball_dist = float("inf")

        # Last man detection for any team size.
        # We are last man if no alive teammate is closer to our goal than us.
        our_dist_to_goal = distance_2d(state.car.position, state.own_goal)
        alive_teammates = [tm for tm in state.teammates if not tm.is_demolished]
        if not alive_teammates:
            state.is_last_man = True
        else:
            teammates_behind = [
                tm for tm in alive_teammates
                if distance_2d(tm.position, state.own_goal) < our_dist_to_goal
            ]
            state.is_last_man = len(teammates_behind) == 0

    def to_tensor(self, state: GameState) -> np.ndarray:
        """Convert GameState to a flat feature tensor for the ML policy.

        Feature vector layout (all floats, normalized where sensible):
        [0-2]   car position (x, y, z) / field_scale
        [3-5]   car velocity / max_speed
        [6-8]   car rotation (pitch, yaw, roll) / pi
        [9]     car boost / 100
        [10]    car speed / max_speed
        [11]    car on ground (0/1)
        [12-14] ball position / field_scale
        [15-17] ball velocity / max_speed
        [18]    ball distance / field_scale
        [19]    time to ball (clamped 0-5) / 5
        [20]    opponent time to ball / 5
        [21-23] opponent position / field_scale (first opponent)
        [24-26] opponent velocity / max_speed
        [27]    opponent boost / 100
        [28]    ball to own goal dist / field_scale
        [29]    ball to opp goal dist / field_scale
        [30]    is last man (0/1)
        [31]    is kickoff (0/1)
        [32]    score differential (us - them) / 10
        [33]    time remaining / 300
        """
        fs = FIELD_HALF_LENGTH  # field scale
        ms = 2300.0  # max speed

        features = np.zeros(34, dtype=np.float32)

        # Car
        features[0] = state.car.position.x / fs
        features[1] = state.car.position.y / fs
        features[2] = state.car.position.z / fs
        features[3] = state.car.velocity.x / ms
        features[4] = state.car.velocity.y / ms
        features[5] = state.car.velocity.z / ms
        features[6] = state.car.rotation.pitch / math.pi
        features[7] = state.car.rotation.yaw / math.pi
        features[8] = state.car.rotation.roll / math.pi
        features[9] = state.car.boost / 100.0
        features[10] = state.car.speed / ms
        features[11] = float(state.car.is_on_ground)

        # Ball
        features[12] = state.ball.position.x / fs
        features[13] = state.ball.position.y / fs
        features[14] = state.ball.position.z / fs
        features[15] = state.ball.velocity.x / ms
        features[16] = state.ball.velocity.y / ms
        features[17] = state.ball.velocity.z / ms
        features[18] = state.ball_distance / fs
        features[19] = min(5.0, state.time_to_ball) / 5.0
        features[20] = min(5.0, state.opponent_time_to_ball) / 5.0

        # First opponent (pad with zeros if none)
        if state.opponents:
            opp = state.opponents[0]
            features[21] = opp.position.x / fs
            features[22] = opp.position.y / fs
            features[23] = opp.position.z / fs
            features[24] = opp.velocity.x / ms
            features[25] = opp.velocity.y / ms
            features[26] = opp.velocity.z / ms
            features[27] = opp.boost / 100.0

        # Derived
        features[28] = state.ball_to_own_goal_dist / fs
        features[29] = state.ball_to_opp_goal_dist / fs
        features[30] = float(state.is_last_man)
        features[31] = float(state.is_kickoff)
        features[32] = (state.score_us - state.score_them) / 10.0
        features[33] = state.time_remaining / 300.0

        return features


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _vec3(data) -> Vec3:
    """Convert list/tuple/array to Vec3."""
    if isinstance(data, Vec3):
        return data
    if isinstance(data, (list, tuple, np.ndarray)):
        return Vec3(float(data[0]), float(data[1]), float(data[2]) if len(data) > 2 else 0.0)
    return Vec3()


def _build_car(data: dict, team: Team, index: int) -> CarState:
    """Build a CarState from a dict."""
    pos = _vec3(data.get("position", [0, 0, 0]))
    vel = _vec3(data.get("velocity", [0, 0, 0]))
    rot_data = data.get("rotation", [0, 0, 0])
    rot = Rotator(
        pitch=float(rot_data[0]) if len(rot_data) > 0 else 0.0,
        yaw=float(rot_data[1]) if len(rot_data) > 1 else 0.0,
        roll=float(rot_data[2]) if len(rot_data) > 2 else 0.0,
    )
    forward = car_forward_vector(rot.yaw, rot.pitch)
    speed = vel.length()

    return CarState(
        position=pos,
        velocity=vel,
        rotation=rot,
        angular_velocity=_vec3(data.get("angular_velocity", [0, 0, 0])),
        boost=float(data.get("boost", 0)),
        is_on_ground=bool(data.get("is_on_ground", True)),
        has_jumped=bool(data.get("has_jumped", False)),
        has_double_jumped=bool(data.get("has_double_jumped", False)),
        is_demolished=bool(data.get("is_demolished", False)),
        is_supersonic=speed >= 2200.0,
        team=team,
        index=index,
        forward=forward,
        speed=speed,
    )
