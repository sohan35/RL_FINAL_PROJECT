from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from cartpole_env import CustomCartPoleEnv

# Create the CartPole environment
env = make_vec_env(CustomCartPoleEnv, n_envs=1)

# Define and train the PPO agent
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10000)

# Save the trained model
model.save("cartpole_model")
print("DONE")

#python train_cartpole.py
#python test_cartpole.py
