from typing import Optional, Dict, Any, Union, Type

import gym
import gym_super_mario_bros
from gym_super_mario_bros.actions import RIGHT_ONLY
from nes_py.wrappers import JoypadSpace
from stable_baselines3.common.atari_wrappers import NoopResetEnv, MaxAndSkipEnv, FireResetEnv, WarpFrame, ClipRewardEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecEnv, DummyVecEnv, SubprocVecEnv


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


class SuperMarioAtariWrapper(gym.Wrapper):
    def __init__(
            self,
            env: gym.Env,
            noop_max: int = 30,
            frame_skip: int = 4,
            screen_size: int = 84,
            terminal_on_life_loss: bool = True,
            clip_reward: bool = True,
    ):
        env = NoopResetEnv(env, noop_max=noop_max)
        env = MaxAndSkipEnv(env, skip=frame_skip)
        if terminal_on_life_loss:
            env = EpisodicLifeEnv(env)
        if "FIRE" in env.unwrapped.get_action_meanings():
            env = FireResetEnv(env)
        env = WarpFrame(env, width=screen_size, height=screen_size)
        if clip_reward:
            env = ClipRewardEnv(env)
        env = JoypadSpace(env, RIGHT_ONLY)

        super(SuperMarioAtariWrapper, self).__init__(env)


def make_super_mario_env(
        env_id: Union[str, Type[gym.Env]],
        n_envs: int = 1,
        seed: Optional[int] = None,
        start_index: int = 0,
        monitor_dir: Optional[str] = None,
        wrapper_kwargs: Optional[Dict[str, Any]] = None,
        env_kwargs: Optional[Dict[str, Any]] = None,
        vec_env_cls: Optional[Union[DummyVecEnv, SubprocVecEnv]] = None,
        vec_env_kwargs: Optional[Dict[str, Any]] = None,
        monitor_kwargs: Optional[Dict[str, Any]] = None,
) -> VecEnv:
    if wrapper_kwargs is None:
        wrapper_kwargs = {}

    def mario_wrapper(env: gym.Env) -> gym.Env:
        env = SuperMarioAtariWrapper(env, **wrapper_kwargs)
        return env

    return make_vec_env(
        env_id,
        n_envs=n_envs,
        seed=seed,
        start_index=start_index,
        monitor_dir=monitor_dir,
        wrapper_class=mario_wrapper,
        env_kwargs=env_kwargs,
        vec_env_cls=vec_env_cls,
        vec_env_kwargs=vec_env_kwargs,
        monitor_kwargs=monitor_kwargs,
    )


def env_creator(num_envs):
    _ = gym_super_mario_bros.make('SuperMarioBros-v0')  # register just in case
    env = make_super_mario_env('SuperMarioBros-v0', n_envs=num_envs, wrapper_kwargs={"clip_reward": False})
    return env
