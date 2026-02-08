"""Unit tests for core types and physics utilities."""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.types import Vec3, Intent, ControlOutput, CarState, BallState, Rotator
from src.utils import (
    distance,
    distance_2d,
    angle_between,
    steer_toward,
    time_to_reach,
    predict_ball_position,
    estimate_intercept_time,
    shot_quality,
    is_ball_threatening_goal,
    back_post_position,
    clamp_to_field,
    car_forward_vector,
    local_coordinates,
    FIELD_HALF_LENGTH,
)


class TestVec3:
    def test_basic_ops(self):
        a = Vec3(1, 2, 3)
        b = Vec3(4, 5, 6)
        c = a + b
        assert c.x == 5 and c.y == 7 and c.z == 9

    def test_subtraction(self):
        a = Vec3(5, 5, 5)
        b = Vec3(1, 2, 3)
        c = a - b
        assert c.x == 4 and c.y == 3 and c.z == 2

    def test_length(self):
        v = Vec3(3, 4, 0)
        assert abs(v.length() - 5.0) < 1e-6

    def test_normalized(self):
        v = Vec3(0, 0, 10)
        n = v.normalized()
        assert abs(n.z - 1.0) < 1e-6
        assert abs(n.length() - 1.0) < 1e-6

    def test_zero_normalized(self):
        v = Vec3(0, 0, 0)
        n = v.normalized()
        assert n.length() < 1e-6

    def test_dot(self):
        a = Vec3(1, 0, 0)
        b = Vec3(0, 1, 0)
        assert abs(a.dot(b)) < 1e-6  # Perpendicular

    def test_to_numpy(self):
        v = Vec3(1.5, 2.5, 3.5)
        arr = v.to_numpy()
        assert arr.shape == (3,)
        assert abs(arr[0] - 1.5) < 1e-6

    def test_flat(self):
        v = Vec3(1, 2, 100)
        f = v.flat()
        assert f.z == 0.0


class TestIntent:
    def test_count(self):
        assert Intent.count() == 7

    def test_values(self):
        assert Intent.DEFEND_SHADOW == 0
        assert Intent.GRAB_BOOST_SAFE == 6


class TestControlOutput:
    def test_sanitize(self):
        ctrl = ControlOutput(throttle=2.0, steer=-5.0, pitch=0.5)
        ctrl.sanitize()
        assert ctrl.throttle == 1.0
        assert ctrl.steer == -1.0
        assert ctrl.pitch == 0.5


class TestPhysics:
    def test_distance(self):
        a = Vec3(0, 0, 0)
        b = Vec3(3, 4, 0)
        assert abs(distance(a, b) - 5.0) < 1e-6

    def test_distance_2d(self):
        a = Vec3(0, 0, 100)
        b = Vec3(3, 4, 200)
        assert abs(distance_2d(a, b) - 5.0) < 1e-6

    def test_angle_between(self):
        a = Vec3(1, 0, 0)
        b = Vec3(0, 1, 0)
        angle = angle_between(a, b)
        assert abs(angle - math.pi / 2) < 1e-6

    def test_predict_ball_position(self):
        ball = BallState(
            position=Vec3(0, 0, 200),
            velocity=Vec3(1000, 0, 0),
        )
        pos = predict_ball_position(ball, 1.0)
        assert abs(pos.x - 1000.0) < 1e-3

    def test_car_forward_vector(self):
        fwd = car_forward_vector(0.0, 0.0)
        assert abs(fwd.x - 1.0) < 1e-6
        assert abs(fwd.y) < 1e-6

    def test_clamp_to_field(self):
        pos = Vec3(99999, 99999, -50)
        clamped = clamp_to_field(pos)
        assert clamped.x < 5000
        assert clamped.y < 6000
        assert clamped.z >= 0

    def test_back_post_position(self):
        own_goal = Vec3(0, -FIELD_HALF_LENGTH, 0)
        ball = Vec3(1000, 0, 0)  # Ball on right
        bp = back_post_position(own_goal, ball)
        # Back post should be on left (negative x)
        assert bp.x < 0

    def test_is_ball_threatening_goal(self):
        own_goal = Vec3(0, -5120, 0)
        # Ball moving toward our goal
        ball = BallState(
            position=Vec3(0, -2000, 100),
            velocity=Vec3(0, -2000, 0),
        )
        assert is_ball_threatening_goal(ball, own_goal, threshold_time=3.0)

        # Ball moving away
        ball_away = BallState(
            position=Vec3(0, -2000, 100),
            velocity=Vec3(0, 2000, 0),
        )
        assert not is_ball_threatening_goal(ball_away, own_goal)

    def test_shot_quality_ranges(self):
        car = CarState(
            position=Vec3(0, 0, 17),
            forward=Vec3(0, 1, 0),
            speed=1000,
        )
        ball = BallState(position=Vec3(0, 3000, 100), velocity=Vec3(0, 500, 0))
        opp_goal = Vec3(0, 5120, 0)
        opponents = [
            CarState(position=Vec3(2000, 4000, 17))
        ]
        quality = shot_quality(car, ball, opp_goal, opponents)
        assert 0.0 <= quality <= 1.0


class TestSteerAndNav:
    def test_steer_toward_straight_ahead(self):
        car = CarState(
            position=Vec3(0, 0, 0),
            rotation=Rotator(0, 0, 0),  # Facing +X
        )
        target = Vec3(1000, 0, 0)  # Straight ahead
        steer = steer_toward(car, target)
        assert abs(steer) < 0.1

    def test_local_coordinates(self):
        car = CarState(
            position=Vec3(0, 0, 0),
            rotation=Rotator(0, 0, 0),  # Facing +X
        )
        target = Vec3(100, 50, 0)  # Ahead and to the right
        fwd, right = local_coordinates(car, target)
        assert fwd > 0  # Target is in front
