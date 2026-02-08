"""
DominanceBot PPO Training v2 — Aggressive reward shaping for GC-level play.

Auto-resumes from latest checkpoint if available.

Usage:
    python train_ppo_rlgym.py
"""

import numpy as np
from rlgym.api import RLGym, RewardFunction
from rlgym.rocket_league.action_parsers import LookupTableAction, RepeatAction
from rlgym.rocket_league.done_conditions import (
    GoalCondition,
    NoTouchTimeoutCondition,
    TimeoutCondition,
    AnyCondition,
)
from rlgym.rocket_league.obs_builders import DefaultObs
from rlgym.rocket_league.reward_functions import CombinedReward, GoalReward
from rlgym.rocket_league.sim import RocketSimEngine
from rlgym.rocket_league.state_mutators import (
    MutatorSequence,
    FixedTeamSizeMutator,
    KickoffMutator,
)
from rlgym.rocket_league import common_values
from rlgym_ppo.util import RLGymV2GymWrapper


# ===========================================================================
# REWARD FUNCTIONS
# ===========================================================================

class VelocityPlayerToBallReward(RewardFunction):
    def reset(self, agents, initial_state, shared_info):
        pass

    def get_rewards(self, agents, state, is_terminated, is_truncated, shared_info):
        rewards = {}
        for agent in agents:
            car = state.cars[agent]
            ball = state.ball
            car_to_ball = ball.position - car.physics.position
            dist = np.linalg.norm(car_to_ball)
            car_to_ball_norm = car_to_ball / (dist + 1e-8)
            vel = car.physics.linear_velocity
            vel_toward_ball = np.dot(vel, car_to_ball_norm)
            rewards[agent] = vel_toward_ball / 2300.0
        return rewards


class VelocityBallToGoalReward(RewardFunction):
    def reset(self, agents, initial_state, shared_info):
        pass

    def get_rewards(self, agents, state, is_terminated, is_truncated, shared_info):
        rewards = {}
        for agent in agents:
            car = state.cars[agent]
            ball = state.ball
            if car.team_num == common_values.BLUE_TEAM:
                opp_goal = np.array(common_values.ORANGE_GOAL_CENTER)
            else:
                opp_goal = np.array(common_values.BLUE_GOAL_CENTER)
            ball_to_goal = opp_goal - ball.position
            ball_to_goal_norm = ball_to_goal / (np.linalg.norm(ball_to_goal) + 1e-8)
            ball_vel = ball.linear_velocity
            vel_toward_goal = np.dot(ball_vel, ball_to_goal_norm)
            rewards[agent] = vel_toward_goal / 6000.0
        return rewards


class TouchBallWithSpeedReward(RewardFunction):
    """Only reward touches that move the ball fast."""
    def reset(self, agents, initial_state, shared_info):
        pass

    def get_rewards(self, agents, state, is_terminated, is_truncated, shared_info):
        rewards = {}
        for agent in agents:
            car = state.cars[agent]
            if car.ball_touches > 0:
                ball_speed = np.linalg.norm(state.ball.linear_velocity)
                rewards[agent] = min(ball_speed / 4600.0, 1.0)
            else:
                rewards[agent] = 0.0
        return rewards


class OffensivePositioningReward(RewardFunction):
    """Reward being ahead of ball toward opponent goal."""
    def reset(self, agents, initial_state, shared_info):
        pass

    def get_rewards(self, agents, state, is_terminated, is_truncated, shared_info):
        rewards = {}
        for agent in agents:
            car = state.cars[agent]
            ball = state.ball
            if car.team_num == common_values.BLUE_TEAM:
                opp_goal_y = common_values.ORANGE_GOAL_CENTER[1]
                car_ahead = (car.physics.position[1] - ball.position[1]) / abs(opp_goal_y)
            else:
                opp_goal_y = common_values.BLUE_GOAL_CENTER[1]
                car_ahead = -(car.physics.position[1] - ball.position[1]) / abs(opp_goal_y)
            rewards[agent] = np.clip(car_ahead * 0.5, -0.5, 0.5)
        return rewards


class DefensivePositioningReward(RewardFunction):
    """Reward being between ball and own goal when ball is on our side."""
    def reset(self, agents, initial_state, shared_info):
        pass

    def get_rewards(self, agents, state, is_terminated, is_truncated, shared_info):
        rewards = {}
        for agent in agents:
            car = state.cars[agent]
            ball = state.ball
            if car.team_num == common_values.BLUE_TEAM:
                own_goal = np.array(common_values.BLUE_GOAL_CENTER)
                ball_on_our_side = ball.position[1] < 0
            else:
                own_goal = np.array(common_values.ORANGE_GOAL_CENTER)
                ball_on_our_side = ball.position[1] > 0

            if ball_on_our_side:
                goal_to_ball = ball.position - own_goal
                goal_to_car = car.physics.position - own_goal
                dist_ball = np.linalg.norm(goal_to_ball[:2])
                dist_car = np.linalg.norm(goal_to_car[:2])
                if dist_ball > 500:
                    dot = np.dot(goal_to_ball[:2], goal_to_car[:2])
                    alignment = dot / (dist_ball * dist_car + 1e-8)
                    between = 1.0 if dist_car < dist_ball else 0.0
                    rewards[agent] = alignment * between * 0.5
                else:
                    rewards[agent] = 0.0
            else:
                rewards[agent] = 0.0
        return rewards


class BoostUsageReward(RewardFunction):
    """Reward maintaining ~40 boost. Punish empty and hoarding."""
    def reset(self, agents, initial_state, shared_info):
        pass

    def get_rewards(self, agents, state, is_terminated, is_truncated, shared_info):
        rewards = {}
        for agent in agents:
            boost = state.cars[agent].boost_amount
            if boost < 40:
                rewards[agent] = (boost / 40.0) * 0.3 - 0.1
            else:
                rewards[agent] = 0.3 - (boost - 40) / 200.0
        return rewards


class AerialTouchReward(RewardFunction):
    """Reward touching ball while in the air."""
    def reset(self, agents, initial_state, shared_info):
        pass

    def get_rewards(self, agents, state, is_terminated, is_truncated, shared_info):
        rewards = {}
        for agent in agents:
            car = state.cars[agent]
            if car.ball_touches > 0 and not car.on_ground:
                ball_height = state.ball.position[2]
                rewards[agent] = min(ball_height / 1000.0, 2.0)
            else:
                rewards[agent] = 0.0
        return rewards


class SpeedReward(RewardFunction):
    """Small reward for maintaining high speed."""
    def reset(self, agents, initial_state, shared_info):
        pass

    def get_rewards(self, agents, state, is_terminated, is_truncated, shared_info):
        rewards = {}
        for agent in agents:
            speed = np.linalg.norm(state.cars[agent].physics.linear_velocity)
            rewards[agent] = speed / 2300.0
        return rewards


class DontBallChaseReward(RewardFunction):
    """Punish being close to a teammate who is also close to ball."""
    def reset(self, agents, initial_state, shared_info):
        pass

    def get_rewards(self, agents, state, is_terminated, is_truncated, shared_info):
        rewards = {}
        for agent in agents:
            car = state.cars[agent]
            ball_pos = state.ball.position
            my_dist = np.linalg.norm(car.physics.position - ball_pos)
            penalty = 0.0
            for other_agent in agents:
                if other_agent == agent:
                    continue
                other_car = state.cars[other_agent]
                if other_car.team_num != car.team_num:
                    continue
                other_dist = np.linalg.norm(other_car.physics.position - ball_pos)
                if my_dist < 1500 and other_dist < 1500:
                    if my_dist > other_dist:
                        penalty = -0.5 * (1.0 - my_dist / 1500.0)
                    else:
                        penalty = -0.1
            rewards[agent] = penalty
        return rewards


class DribbleReward(RewardFunction):
    """Reward having ball on top of car with control."""
    def reset(self, agents, initial_state, shared_info):
        pass

    def get_rewards(self, agents, state, is_terminated, is_truncated, shared_info):
        rewards = {}
        for agent in agents:
            car = state.cars[agent]
            ball = state.ball
            car_pos = car.physics.position
            ball_pos = ball.position
            dx = ball_pos[0] - car_pos[0]
            dy = ball_pos[1] - car_pos[1]
            horiz_dist = np.sqrt(dx * dx + dy * dy)
            height_diff = ball_pos[2] - car_pos[2]
            rel_vel = np.linalg.norm(ball.linear_velocity - car.physics.linear_velocity)

            if not car.on_ground:
                rewards[agent] = 0.0
                continue

            horiz_ok = horiz_dist < 250
            height_ok = 100 < height_diff < 220
            speed_ok = rel_vel < 400

            if horiz_ok and height_ok and speed_ok:
                control_quality = 1.0 - (horiz_dist / 250.0) * 0.5 - (rel_vel / 400.0) * 0.5
                car_speed = np.linalg.norm(car.physics.linear_velocity)
                if car.team_num == common_values.BLUE_TEAM:
                    forward_progress = car.physics.linear_velocity[1] / (car_speed + 1e-8)
                else:
                    forward_progress = -car.physics.linear_velocity[1] / (car_speed + 1e-8)
                rewards[agent] = control_quality + max(0, forward_progress) * 0.5
            else:
                rewards[agent] = 0.0
        return rewards


class BallControlReward(RewardFunction):
    """Reward keeping ball close with matched velocity."""
    def reset(self, agents, initial_state, shared_info):
        pass

    def get_rewards(self, agents, state, is_terminated, is_truncated, shared_info):
        rewards = {}
        for agent in agents:
            car = state.cars[agent]
            ball = state.ball
            dist = np.linalg.norm(ball.position - car.physics.position)
            rel_vel = np.linalg.norm(ball.linear_velocity - car.physics.linear_velocity)
            if dist < 500 and car.on_ground:
                closeness = 1.0 - dist / 500.0
                vel_match = max(0, 1.0 - rel_vel / 800.0)
                rewards[agent] = closeness * vel_match * 0.5
            else:
                rewards[agent] = 0.0
        return rewards


class FlickReward(RewardFunction):
    """Reward launching ball upward off the car with speed."""
    def reset(self, agents, initial_state, shared_info):
        self._prev_ball_z_vel = {}

    def get_rewards(self, agents, state, is_terminated, is_truncated, shared_info):
        rewards = {}
        ball_z_vel = state.ball.linear_velocity[2]
        for agent in agents:
            car = state.cars[agent]
            prev_z = self._prev_ball_z_vel.get(agent, 0.0)
            if car.ball_touches > 0:
                z_impulse = ball_z_vel - prev_z
                ball_speed = np.linalg.norm(state.ball.linear_velocity)
                ball_height = state.ball.position[2]
                if z_impulse > 500 and ball_height < 400 and ball_speed > 1500:
                    rewards[agent] = min(z_impulse / 1500.0, 2.0)
                else:
                    rewards[agent] = 0.0
            else:
                rewards[agent] = 0.0
            self._prev_ball_z_vel[agent] = ball_z_vel
        return rewards


# ===========================================================================
# Environment Builder
# ===========================================================================

def build_rlgym_v2_env():
    spawn_opponents = True
    team_size = 3
    blue_team_size = team_size
    orange_team_size = team_size if spawn_opponents else 0

    tick_skip = 8
    no_touch_timeout_seconds = 30
    game_timeout_seconds = 300

    action_parser = RepeatAction(LookupTableAction(), repeats=tick_skip)
    termination_condition = GoalCondition()
    truncation_condition = AnyCondition(
        NoTouchTimeoutCondition(timeout_seconds=no_touch_timeout_seconds),
        TimeoutCondition(timeout_seconds=game_timeout_seconds),
    )

    state_mutator = MutatorSequence(
        FixedTeamSizeMutator(blue_size=blue_team_size, orange_size=orange_team_size),
        KickoffMutator(),
    )

    obs_builder = DefaultObs(zero_padding=None)

    reward_fn = CombinedReward(
        # Core game objectives
        (GoalReward(), 10.0),
        (VelocityBallToGoalReward(), 3.0),

        # Smart touches
        (TouchBallWithSpeedReward(), 1.5),
        (AerialTouchReward(), 0.5),

        # Positioning
        (OffensivePositioningReward(), 0.3),
        (DefensivePositioningReward(), 0.4),
        (DontBallChaseReward(), 0.8),

        # Possession
        (DribbleReward(), 1.2),
        (BallControlReward(), 0.3),
        (FlickReward(), 0.8),

        # Movement
        (VelocityPlayerToBallReward(), 0.05),
        (SpeedReward(), 0.03),

        # Boost
        (BoostUsageReward(), 0.02),
    )

    engine = RocketSimEngine()

    env = RLGym(
        state_mutator=state_mutator,
        obs_builder=obs_builder,
        action_parser=action_parser,
        reward_fn=reward_fn,
        transition_engine=engine,
        termination_cond=termination_condition,
        truncation_cond=truncation_condition,
    )

    return RLGymV2GymWrapper(env)


# ===========================================================================
# Training
# ===========================================================================

if __name__ == "__main__":
    import os
    from rlgym_ppo import Learner

    # Auto-find latest checkpoint
    checkpoint_load = None
    for dirname in os.listdir("data/checkpoints") if os.path.exists("data/checkpoints") else []:
        full = os.path.join("data/checkpoints", dirname)
        if os.path.isdir(full):
            subdirs = [d for d in os.listdir(full) if d.isdigit()]
            if subdirs:
                latest = max(subdirs, key=lambda d: int(d))
                candidate = os.path.join(full, latest)
                if checkpoint_load is None or int(latest) > int(os.path.basename(checkpoint_load)):
                    checkpoint_load = candidate

    if checkpoint_load:
        print(f"RESUMING FROM CHECKPOINT: {checkpoint_load}")
    else:
        print("STARTING FRESH — no checkpoint found")

    learner = Learner(
        env_create_function=build_rlgym_v2_env,

        policy_layer_sizes=[256, 256],
        critic_layer_sizes=[256, 256],

        ppo_batch_size=100_000,
        ts_per_iteration=100_000,
        exp_buffer_size=300_000,
        ppo_minibatch_size=50_000,
        ppo_ent_coef=0.008,
        ppo_epochs=3,
        policy_lr=2e-4,
        critic_lr=2e-4,

        n_proc=12,

        standardize_returns=True,
        standardize_obs=False,

        save_every_ts=2_000_000,
        timestep_limit=10_000_000_000,
        log_to_wandb=False,
        checkpoint_load_folder=checkpoint_load,

        render=False,
        render_delay=0.05,

        add_unix_timestamp=False,
    )

    learner.learn()
