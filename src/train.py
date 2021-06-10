
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecFrameStack, VecNormalize

from src.env_wrappers import env_creator

env = env_creator(num_envs=8)
env = VecFrameStack(env, n_stack=4)
env = VecNormalize(env)

model = PPO('CnnPolicy', env, verbose=1, gae_lambda=0.95, n_steps=128, batch_size=2048, ent_coef=0.01)
model.learn(total_timesteps=200000)
model.save("../outputs/saved_models/super_mario_ppo")

obs = env.reset()
for i in range(1000):
    action, _state = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    env.render()
    if done.all():
        obs = env.reset()
