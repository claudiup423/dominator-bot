"""Fast 1v1 training — should show basic ball-chasing in 2-3 hours."""

import numpy as np
from rlgym.api import RLGym, RewardFunction
from rlgym.rocket_league.action_parsers import LookupTableAction, RepeatAction
from rlgym.rocket_league.done_conditions import GoalCondition, NoTouchTimeoutCondition, AnyCondition
from rlgym.rocket_league.obs_builders import DefaultObs
from rlgym.rocket_league.reward_functions import CombinedReward, GoalReward, TouchReward
from rlgym.rocket_league.sim import RocketSimEngine
from rlgym.rocket_league.state_mutators import MutatorSequence, FixedTeamSizeMutator, KickoffMutator
from rlgym.rocket_league import common_values
from rlgym_ppo.util import RLGymV2GymWrapper


class VelocityPlayerToBallReward(RewardFunction):
    def reset(self, agents, initial_state, shared_info): pass
    def get_rewards(self, agents, state, is_terminated, is_truncated, shared_info):
        rewards = {}
        for agent in agents:
            car = state.cars[agent]
            ball = state.ball
            car_to_ball = ball.position - car.physics.position
            dist = np.linalg.norm(car_to_ball)
            direction = car_to_ball / (dist + 1e-8)
            vel_toward = np.dot(car.physics.linear_velocity, direction)
            rewards[agent] = vel_toward / 2300.0
        return rewards


class VelocityBallToGoalReward(RewardFunction):
    def reset(self, agents, initial_state, shared_info): pass
    def get_rewards(self, agents, state, is_terminated, is_truncated, shared_info):
        rewards = {}
        for agent in agents:
            car = state.cars[agent]
            if car.team_num == common_values.BLUE_TEAM:
                goal = np.array(common_values.ORANGE_GOAL_CENTER)
            else:
                goal = np.array(common_values.BLUE_GOAL_CENTER)
            ball_to_goal = goal - state.ball.position
            direction = ball_to_goal / (np.linalg.norm(ball_to_goal) + 1e-8)
            vel_toward = np.dot(state.ball.linear_velocity, direction)
            rewards[agent] = vel_toward / 6000.0
        return rewards


class SpeedReward(RewardFunction):
    def reset(self, agents, initial_state, shared_info): pass
    def get_rewards(self, agents, state, is_terminated, is_truncated, shared_info):
        rewards = {}
        for agent in agents:
            speed = np.linalg.norm(state.cars[agent].physics.linear_velocity)
            rewards[agent] = speed / 2300.0
        return rewards


def build_env():
    tick_skip = 8
    action_parser = RepeatAction(LookupTableAction(), repeats=tick_skip)
    termination_condition = GoalCondition()
    truncation_condition = AnyCondition(
        NoTouchTimeoutCondition(timeout_seconds=15),
    )
    state_mutator = MutatorSequence(
        FixedTeamSizeMutator(blue_size=1, orange_size=1),
        KickoffMutator(),
    )
    obs_builder = DefaultObs(zero_padding=None)

    reward_fn = CombinedReward(
        (GoalReward(), 10.0),
        (VelocityBallToGoalReward(), 5.0),
        (VelocityPlayerToBallReward(), 1.0),
        (TouchReward(), 3.0),
        (SpeedReward(), 0.1),
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


if __name__ == "__main__":
    from rlgym_ppo import Learner

    learner = Learner(
        env_create_function=build_env,
        policy_layer_sizes=[256, 256],
        critic_layer_sizes=[256, 256],
        ppo_batch_size=50_000,
        ts_per_iteration=50_000,
        exp_buffer_size=150_000,
        ppo_minibatch_size=25_000,
        ppo_ent_coef=0.01,
        ppo_epochs=3,
        policy_lr=5e-4,
        critic_lr=5e-4,
        n_proc=16,
        standardize_returns=True,
        standardize_obs=False,
        save_every_ts=5_000_000,
        timestep_limit=10_000_000_000,
        log_to_wandb=False,
        render=False,
        add_unix_timestamp=True,
    )
    learner.learn()
