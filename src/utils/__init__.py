"""Physics helpers: intercept estimation, time-to-ball, geometry.

All functions are pure (no side effects) and operate on Vec3 / numpy arrays.
These are the building blocks for both the expert and the ML pipeline.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np

from src.types import Vec3, BallState, CarState

# Constants from config defaults (avoid loading yaml in hot path)
GRAVITY = 650.0
BALL_RADIUS = 92.75
MAX_CAR_SPEED = 2300.0
BOOST_ACCEL = 991.667
FIELD_HALF_LENGTH = 5120.0
FIELD_HALF_WIDTH = 4096.0
GOAL_HALF_WIDTH = 893.0
GOAL_HEIGHT = 642.0


def distance(a: Vec3, b: Vec3) -> float:
    """Euclidean distance between two 3D points."""
    dx = a.x - b.x
    dy = a.y - b.y
    dz = a.z - b.z
    return math.sqrt(dx * dx + dy * dy + dz * dz)


def distance_2d(a: Vec3, b: Vec3) -> float:
    """Ground-plane distance (ignoring Z)."""
    dx = a.x - b.x
    dy = a.y - b.y
    return math.sqrt(dx * dx + dy * dy)


def angle_between(v1: Vec3, v2: Vec3) -> float:
    """Angle in radians between two vectors (0 to pi)."""
    d = v1.normalized().dot(v2.normalized())
    d = max(-1.0, min(1.0, d))
    return math.acos(d)


def local_coordinates(car: CarState, target: Vec3) -> tuple[float, float]:
    """Convert a world target to local (forward, right) relative to car facing.

    Returns (forward_component, right_component) — positive right means
    the target is to the right of the car's heading.
    """
    diff = target - car.position
    yaw = car.rotation.yaw
    cos_y = math.cos(-yaw)
    sin_y = math.sin(-yaw)
    local_x = diff.x * cos_y - diff.y * sin_y  # forward
    local_y = diff.x * sin_y + diff.y * cos_y  # right
    return local_x, local_y


def steer_toward(car: CarState, target: Vec3) -> float:
    """Compute a steer value in [-1, 1] to face a world target.

    Uses a simple proportional controller on the signed angle.
    """
    _, right = local_coordinates(car, target)
    dist = distance_2d(car.position, target)
    if dist < 1e-3:
        return 0.0
    # Signed angle: positive = need to turn right
    angle = math.atan2(right, max(dist, 1.0))
    # Proportional with saturation
    steer = max(-1.0, min(1.0, angle * 5.0))
    return steer


def angle_to_target(car: CarState, target: Vec3) -> float:
    """Absolute angle in radians between car's forward and direction to target."""
    to_target = (target - car.position).flat().normalized()
    forward = car.forward.flat().normalized()
    return angle_between(forward, to_target)


def time_to_reach(car: CarState, target: Vec3, use_boost: bool = True) -> float:
    """Rough estimate of time for a car to reach a ground position.

    Uses a simple model: accelerate at boost rate up to max speed.
    This is an approximation (ignores turning, doesn't brake).
    """
    dist = distance_2d(car.position, target)
    speed = car.speed
    if speed < 1e-3:
        speed = 100.0  # Avoid div-by-zero for stationary cars

    if use_boost and car.boost > 10:
        # Average acceleration with boost
        avg_speed = min(MAX_CAR_SPEED, speed + 400)
    else:
        avg_speed = min(MAX_CAR_SPEED, speed + 200)

    # Add penalty for facing wrong direction
    angle = angle_to_target(car, target)
    turn_penalty = angle * 0.4  # ~0.4s per radian of turning

    return (dist / avg_speed) + turn_penalty


def predict_ball_position(ball: BallState, dt: float) -> Vec3:
    """Predict ball position after dt seconds (simple ballistic, no bounces).

    Good enough for short-term predictions (<1s). For longer predictions,
    use the full simulation.
    """
    return Vec3(
        ball.position.x + ball.velocity.x * dt,
        ball.position.y + ball.velocity.y * dt,
        max(BALL_RADIUS, ball.position.z + ball.velocity.z * dt - 0.5 * GRAVITY * dt * dt),
    )


def estimate_intercept_time(
    car: CarState, ball: BallState, max_time: float = 4.0, steps: int = 20
) -> tuple[float, Vec3]:
    """Find the earliest time at which the car can reach the ball's predicted position.

    Iterates over future timesteps and finds where car-arrival-time ≤ timestep.
    Returns (time, intercept_position).
    """
    best_time = max_time
    best_pos = ball.position
    for i in range(1, steps + 1):
        t = (i / steps) * max_time
        ball_pos = predict_ball_position(ball, t)
        car_time = time_to_reach(car, ball_pos)
        if car_time <= t:
            best_time = t
            best_pos = ball_pos
            break
    return best_time, best_pos


def shot_quality(
    car: CarState,
    ball: BallState,
    opp_goal: Vec3,
    opponents: list[CarState],
) -> float:
    """Compute shot quality ∈ [0, 1].

    Factors: angle to goal, distance to goal, nearest opponent proximity,
    ball speed, net openness.
    Higher = better shot opportunity.
    """
    ball_to_goal = opp_goal - ball.position
    ball_to_goal_dist = ball_to_goal.length2d()

    # Angle factor: how aligned is car→ball→goal?
    car_to_ball = (ball.position - car.position).flat().normalized()
    ball_to_goal_dir = ball_to_goal.flat().normalized()
    alignment = car_to_ball.dot(ball_to_goal_dir)
    angle_score = max(0.0, (alignment + 1.0) / 2.0)  # [0, 1]

    # Distance factor: closer to goal is better (diminishing returns)
    dist_score = max(0.0, 1.0 - ball_to_goal_dist / (FIELD_HALF_LENGTH * 2))

    # Opponent proximity factor: closer opponent = worse shot
    min_opp_dist = float("inf")
    for opp in opponents:
        d = distance_2d(opp.position, ball.position)
        min_opp_dist = min(min_opp_dist, d)
    opp_score = min(1.0, min_opp_dist / 2000.0)

    # Ball speed factor: some speed is needed for a good shot
    speed_score = min(1.0, ball.velocity.length() / 1500.0) * 0.3 + 0.7

    # Combined
    quality = angle_score * 0.4 + dist_score * 0.2 + opp_score * 0.25 + speed_score * 0.15
    return max(0.0, min(1.0, quality))


def is_ball_threatening_goal(ball: BallState, own_goal: Vec3, threshold_time: float = 2.0) -> bool:
    """Check if the ball is heading toward our goal and will arrive soon."""
    # Direction check: ball moving toward our goal
    to_goal = own_goal - ball.position
    if ball.velocity.dot(to_goal) < 0:
        return False  # Ball moving away from our goal

    # Time check
    dist = distance_2d(ball.position, own_goal)
    ball_speed_toward = max(1.0, abs(ball.velocity.dot(to_goal.normalized())))
    eta = dist / ball_speed_toward
    return eta < threshold_time


def back_post_position(own_goal: Vec3, ball: Vec3, offset: float = 400.0) -> Vec3:
    """Compute back-post defensive position.

    Back post is the post *opposite* to the side the ball is on.
    This gives the best coverage angle.
    """
    # Determine which side the ball is on
    if ball.x > 0:
        # Ball is on the right → back post is left
        post_x = -GOAL_HALF_WIDTH + offset
    else:
        post_x = GOAL_HALF_WIDTH - offset

    return Vec3(post_x, own_goal.y, 0.0)


def clamp_to_field(pos: Vec3, margin: float = 200.0) -> Vec3:
    """Clamp a position to stay within field boundaries."""
    return Vec3(
        max(-FIELD_HALF_WIDTH + margin, min(FIELD_HALF_WIDTH - margin, pos.x)),
        max(-FIELD_HALF_LENGTH + margin, min(FIELD_HALF_LENGTH - margin, pos.y)),
        max(0.0, pos.z),
    )


def car_forward_vector(yaw: float, pitch: float = 0.0) -> Vec3:
    """Compute the forward unit vector from yaw and pitch."""
    cp = math.cos(pitch)
    return Vec3(
        cp * math.cos(yaw),
        cp * math.sin(yaw),
        math.sin(pitch),
    )
