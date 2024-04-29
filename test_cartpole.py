from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from cartpole_env import CustomCartPoleEnv

# Load the trained model
model = PPO.load("cartpole_model")

# Create the CartPole environment
env = make_vec_env(CustomCartPoleEnv, n_envs=4)

# Evaluate the agent
mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=10) #_ is used to capture additional information returned by the evaluate_policy

print(f"Cumulative reward: {mean_reward:.2f}")
