import gym
import gym_super_mario_bros
from gym_super_mario_bros.actions import RIGHT_ONLY
from nes_py.wrappers import JoypadSpace
from ray.rllib.env.atari_wrappers import MonitorEnv, NoopResetEnv, MaxAndSkipEnv, FireResetEnv, WarpFrame, FrameStack
from ray.tune import register_env


class EpisodicLifeEnv(gym.Wrapper):
    """
    This class is overridden from the ray rLlib because the super mario bros env. does not have access to 'ale'.
    Instead uses custom env. to find number of lives.
    """
    def __init__(self, env):
        """Make end-of-life == end-of-episode, but only reset on true game over.
        Done by DeepMind for the DQN and co. since it helps value estimation.
        """
        gym.Wrapper.__init__(self, env)
        self.lives = 0
        self.was_real_done = True

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.was_real_done = done
        # check current lives, make loss of life terminal,
        # then update lives to handle bonus lives
        lives = self.env.unwrapped._life  # THIS IS THE IMPORTANT CHANGE FOR SUPER MARIO BROS ENV ****

        if self.lives > lives > 0:
            # for Qbert sometimes we stay in lives == 0 condtion for a few fr
            # so its important to keep lives > 0, so that we only reset once
            # the environment advertises done.
            done = True
        self.lives = lives
        return obs, reward, done, info

    def reset(self, **kwargs):
        """Reset only when lives are exhausted.
        This way all states are still reachable even though lives are episodic,
        and the learner need not know about any of this behind-the-scenes.
        """
        if self.was_real_done:
            obs = self.env.reset(**kwargs)
        else:
            # no-op step to advance from terminal/lost life state
            obs, _, _, _ = self.env.step(0)
        self.lives = self.env.unwrapped._life  # THIS IS THE IMPORTANT CHANGE FOR SUPER MARIO BROS ENV ****
        return obs


def wrap_deepmind(env, dim=84, framestack=True):
    """Configure environment for DeepMind-style Atari.
    Note that we assume reward clipping is done outside the wrapper.
    Args:
        dim (int): Dimension to resize observations to (dim x dim).
        framestack (bool): Whether to framestack observations.
    """
    env = MonitorEnv(env)
    env = NoopResetEnv(env, noop_max=30)
    if "NoFrameskip" in env.spec.id:
        env = MaxAndSkipEnv(env, skip=4)
    env = EpisodicLifeEnv(env)
    if "FIRE" in env.unwrapped.get_action_meanings():
        env = FireResetEnv(env)
    env = WarpFrame(env, dim)
    # env = ScaledFloatFrame(env)  # TODO: use for dqn?
    # env = ClipRewardEnv(env)  # reward clipping is handled by policy eval
    if framestack:
        env = FrameStack(env, 4)
    return env


def env_creator(env_config):
    env = gym_super_mario_bros.make('SuperMarioBros-v0')
    env = JoypadSpace(env, RIGHT_ONLY)
    env = wrap_deepmind(env)
    return env


def register_super_mario_env():
    register_env("super_mario", env_creator)


def test_env_obs():
    env = env_creator("asdf")
    env.reset()
    import numpy as np

    for i in range(1000):
        obs, reward, done, info = env.step(0)

        if not np.isfinite(obs).all() or not np.isfinite(reward):
            print("oops")
