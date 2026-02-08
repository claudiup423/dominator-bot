"""Core data types for the bot pipeline.

All structured types that flow through the pipeline are defined here
to avoid circular imports and to serve as the single source of truth.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class Intent(enum.IntEnum):
    """High-level strategic intents the policy can output."""
    DEFEND_SHADOW = 0
    DEFEND_CLEAR = 1
    CHALLENGE = 2
    ATTACK_SHOOT = 3
    ATTACK_POSSESSION = 4
    ROTATE_BACK = 5
    GRAB_BOOST_SAFE = 6

    @classmethod
    def count(cls) -> int:
        return len(cls)


class Team(enum.IntEnum):
    BLUE = 0
    ORANGE = 1


# ---------------------------------------------------------------------------
# Geometry primitives
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class Vec3:
    """Simple 3D vector used throughout the bot."""
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

    def to_numpy(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z], dtype=np.float32)

    @classmethod
    def from_numpy(cls, arr: np.ndarray) -> Vec3:
        return cls(float(arr[0]), float(arr[1]), float(arr[2]))

    def __add__(self, other: Vec3) -> Vec3:
        return Vec3(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other: Vec3) -> Vec3:
        return Vec3(self.x - other.x, self.y - other.y, self.z - other.z)

    def __mul__(self, s: float) -> Vec3:
        return Vec3(self.x * s, self.y * s, self.z * s)

    def __rmul__(self, s: float) -> Vec3:
        return self.__mul__(s)

    def length(self) -> float:
        return float(np.sqrt(self.x**2 + self.y**2 + self.z**2))

    def length2d(self) -> float:
        return float(np.sqrt(self.x**2 + self.y**2))

    def normalized(self) -> Vec3:
        mag = self.length()
        if mag < 1e-8:
            return Vec3(0, 0, 0)
        return Vec3(self.x / mag, self.y / mag, self.z / mag)

    def dot(self, other: Vec3) -> float:
        return self.x * other.x + self.y * other.y + self.z * other.z

    def flat(self) -> Vec3:
        """Project onto XY plane."""
        return Vec3(self.x, self.y, 0.0)


@dataclass(slots=True)
class Rotator:
    """Euler angles (pitch, yaw, roll) in radians."""
    pitch: float = 0.0
    yaw: float = 0.0
    roll: float = 0.0


# ---------------------------------------------------------------------------
# Game objects
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class BallState:
    position: Vec3 = field(default_factory=Vec3)
    velocity: Vec3 = field(default_factory=Vec3)
    angular_velocity: Vec3 = field(default_factory=Vec3)


@dataclass(slots=True)
class CarState:
    position: Vec3 = field(default_factory=Vec3)
    velocity: Vec3 = field(default_factory=Vec3)
    rotation: Rotator = field(default_factory=Rotator)
    angular_velocity: Vec3 = field(default_factory=Vec3)
    boost: float = 0.0
    is_on_ground: bool = True
    has_jumped: bool = False
    has_double_jumped: bool = False
    is_demolished: bool = False
    is_supersonic: bool = False
    team: Team = Team.BLUE
    index: int = 0
    # Derived
    forward: Vec3 = field(default_factory=Vec3)
    speed: float = 0.0


@dataclass(slots=True)
class BoostPad:
    position: Vec3 = field(default_factory=Vec3)
    is_big: bool = False
    is_active: bool = True
    timer: float = 0.0


# ---------------------------------------------------------------------------
# Composite game state
# ---------------------------------------------------------------------------

@dataclass
class GameState:
    """Full structured game state built each tick."""
    ball: BallState = field(default_factory=BallState)
    car: CarState = field(default_factory=CarState)  # Our car
    opponents: list[CarState] = field(default_factory=list)
    teammates: list[CarState] = field(default_factory=list)
    boost_pads: list[BoostPad] = field(default_factory=list)
    team: Team = Team.BLUE
    tick: int = 0
    time_remaining: float = 300.0
    score_us: int = 0
    score_them: int = 0
    is_kickoff: bool = False
    # Derived fields computed by StateBuilder
    ball_distance: float = 0.0
    ball_direction: Vec3 = field(default_factory=Vec3)
    own_goal: Vec3 = field(default_factory=Vec3)
    opp_goal: Vec3 = field(default_factory=Vec3)
    is_last_man: bool = False
    ball_to_own_goal_dist: float = 0.0
    ball_to_opp_goal_dist: float = 0.0
    closest_opponent_to_ball_dist: float = 0.0
    time_to_ball: float = 0.0  # Our estimated time to reach ball
    opponent_time_to_ball: float = 0.0  # Fastest opponent's time to ball


# ---------------------------------------------------------------------------
# Pipeline outputs
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class StrategyOutput:
    """Output of the strategy layer (ML or expert)."""
    intent: Intent = Intent.ROTATE_BACK
    target: Vec3 = field(default_factory=Vec3)
    risk: float = 0.5  # 0=conservative, 1=aggressive


@dataclass(slots=True)
class TacticalPlan:
    """Concrete plan for the controller to execute."""
    target_position: Vec3 = field(default_factory=Vec3)
    target_speed: float = 2300.0
    face_target: Optional[Vec3] = None  # Where to face (e.g., ball for shots)
    use_boost: bool = False
    dodge: bool = False
    dodge_direction: Optional[Vec3] = None
    jump: bool = False
    handbrake: bool = False
    # Metadata
    source_intent: Intent = Intent.ROTATE_BACK
    was_overridden: bool = False
    override_reason: str = ""


@dataclass(slots=True)
class ControlOutput:
    """Final controller output — maps to RLBot SimpleControllerState."""
    throttle: float = 0.0
    steer: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0
    roll: float = 0.0
    jump: bool = False
    boost: bool = False
    handbrake: bool = False

    def sanitize(self) -> ControlOutput:
        """Clamp all values to valid ranges."""
        self.throttle = max(-1.0, min(1.0, self.throttle))
        self.steer = max(-1.0, min(1.0, self.steer))
        self.pitch = max(-1.0, min(1.0, self.pitch))
        self.yaw = max(-1.0, min(1.0, self.yaw))
        self.roll = max(-1.0, min(1.0, self.roll))
        return self
