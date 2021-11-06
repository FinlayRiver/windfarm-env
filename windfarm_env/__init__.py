import logging
from gym.envs.registration import register

logger = logging.getLogger(__name__)

register(
    id='Windfarm-v0',
    entry_point='windfarm_env.envs:FarmEnv'
)