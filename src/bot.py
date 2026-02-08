"""RLBot v5 entrypoint for DominanceBot.

This is the file that RLBot v5 executes. It subclasses rlbot.managers.Bot
and translates v5 GamePacket data into our internal format, then delegates
all decision-making to DominanceBotBrain.

Supports 1v1, 2v2, and 3v3.
"""

from __future__ import annotations

import sys
import logging
from pathlib import Path

# Ensure project root is on path so our src modules are importable
_PROJECT_ROOT = Path(__file__).parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from rlbot.flat import ControllerState, GamePacket, FieldInfo, Vector3 as FlatVec3
from rlbot.managers import Bot

from src.bot_brain import DominanceBotBrain
from src.types import Team, ControlOutput

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("DominanceBot")

# Boost pad positions on standard map (DFH Stadium).
# Index matches RLBot's boost pad ordering. We cache these on first field_info.
_BOOST_PAD_CACHE: list[dict] | None = None


def _cache_boost_pads(field_info: FieldInfo) -> list[dict]:
    """Cache boost pad locations from field info (doesn't change during match)."""
    global _BOOST_PAD_CACHE
    if _BOOST_PAD_CACHE is not None:
        return _BOOST_PAD_CACHE

    pads = []
    for pad in field_info.boost_pads:
        loc = pad.location
        pads.append({
            "position": [loc.x, loc.y, loc.z],
            "is_big": pad.is_full_boost,
            "is_active": True,
            "timer": 0.0,
        })
    _BOOST_PAD_CACHE = pads
    return _BOOST_PAD_CACHE


def _packet_to_dict(packet: GamePacket, index: int, team: int, field_info: FieldInfo) -> dict:
    """Convert RLBot v5 GamePacket to our internal dict format.

    Handles any number of players (1v1, 2v2, 3v3).
    """
    # Ball (use first ball)
    ball = packet.balls[0].physics if packet.balls else None
    ball_data = {
        "position": [ball.location.x, ball.location.y, ball.location.z] if ball else [0, 0, 93],
        "velocity": [ball.velocity.x, ball.velocity.y, ball.velocity.z] if ball else [0, 0, 0],
        "angular_velocity": [
            ball.angular_velocity.x, ball.angular_velocity.y, ball.angular_velocity.z
        ] if ball else [0, 0, 0],
    }

    # Our car
    my_car = packet.players[index]
    my_phys = my_car.physics
    car_data = {
        "position": [my_phys.location.x, my_phys.location.y, my_phys.location.z],
        "velocity": [my_phys.velocity.x, my_phys.velocity.y, my_phys.velocity.z],
        "rotation": [my_phys.rotation.pitch, my_phys.rotation.yaw, my_phys.rotation.roll],
        "angular_velocity": [
            my_phys.angular_velocity.x, my_phys.angular_velocity.y, my_phys.angular_velocity.z
        ],
        "boost": my_car.boost,
        "is_on_ground": my_car.has_wheel_contact,
        "has_jumped": my_car.air_state != 0,  # 0 = on ground
        "has_double_jumped": my_car.air_state == 3 if hasattr(my_car, 'air_state') else False,
        "is_demolished": my_car.is_demolished,
    }

    # Opponents and teammates
    opponents = []
    teammates = []
    for i, player in enumerate(packet.players):
        if i == index:
            continue
        phys = player.physics
        p_data = {
            "position": [phys.location.x, phys.location.y, phys.location.z],
            "velocity": [phys.velocity.x, phys.velocity.y, phys.velocity.z],
            "rotation": [phys.rotation.pitch, phys.rotation.yaw, phys.rotation.roll],
            "angular_velocity": [
                phys.angular_velocity.x, phys.angular_velocity.y, phys.angular_velocity.z
            ],
            "boost": player.boost,
            "is_on_ground": player.has_wheel_contact,
            "is_demolished": player.is_demolished,
        }
        if player.team == team:
            teammates.append(p_data)
        else:
            opponents.append(p_data)

    # Boost pads — merge cached locations with live active state
    boost_pads = _cache_boost_pads(field_info)
    live_pads = []
    for i, cached in enumerate(boost_pads):
        pad = dict(cached)  # copy
        if i < len(packet.boost_pad_states):
            state = packet.boost_pad_states[i]
            pad["is_active"] = state.is_active
            pad["timer"] = state.timer
        live_pads.append(pad)

    # Game info
    is_kickoff = packet.game_status == 3  # Kickoff status in v5

    # Score from team info
    score_us = 0
    score_them = 0
    for t in packet.teams:
        if t.team_index == team:
            score_us = t.score
        else:
            score_them = t.score

    return {
        "ball": ball_data,
        "car": car_data,
        "opponents": opponents,
        "teammates": teammates,
        "boost_pads": live_pads,
        "time_remaining": packet.match_info.game_time_remaining if packet.match_info else 300.0,
        "score_us": score_us,
        "score_them": score_them,
        "is_kickoff": is_kickoff,
    }


class DominanceBot(Bot):
    """RLBot v5 Bot implementation.

    All decision-making is delegated to DominanceBotBrain.
    This class only handles the RLBot v5 API translation.
    """

    def initialize(self):
        """Called once when the match is loaded and ready."""
        team_enum = Team.BLUE if self.team == 0 else Team.ORANGE
        model_path = _PROJECT_ROOT / "models" / "stable" / "strategy_model.pt"

        self.brain = DominanceBotBrain(
            car_index=self.index,
            team=team_enum,
            model_path=str(model_path) if model_path.exists() else None,
        )
        logger.info(
            f"DominanceBot initialized: index={self.index}, team={self.team}, "
            f"mode={self.brain.mode.value}"
        )

    def get_output(self, packet: GamePacket) -> ControllerState:
        """Called every tick (120Hz). Must return ControllerState."""
        try:
            data = _packet_to_dict(packet, self.index, self.team, self.field_info)
            ctrl = self.brain.tick(data)
            return _to_controller_state(ctrl)
        except Exception as e:
            logger.error(f"get_output failed: {e}", exc_info=True)
            return ControllerState()


def _to_controller_state(ctrl: ControlOutput) -> ControllerState:
    """Convert our ControlOutput to RLBot v5 ControllerState."""
    cs = ControllerState()
    cs.throttle = ctrl.throttle
    cs.steer = ctrl.steer
    cs.pitch = ctrl.pitch
    cs.yaw = ctrl.yaw
    cs.roll = ctrl.roll
    cs.jump = ctrl.jump
    cs.boost = ctrl.boost
    cs.handbrake = ctrl.handbrake
    return cs


if __name__ == "__main__":
    DominanceBot.run()
