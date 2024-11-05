import gymnasium as gym
from stable_baselines3 import SAC
import torch
import CarRacingObstacles.obstacle_ver
from CarRacingObstacles.wrappers import wrap_CarRacingObst
from stable_baselines3.common.callbacks import EvalCallback
from datetime import datetime


if __name__ == "__main__":
    env_name = "CarRacing"
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    print(f"Using device: {device}")

    # Create CarRacing environment
    env = gym.make(env_name, max_episode_steps=1500)
    env = wrap_CarRacingObst(env)

    # Initialize SAC
    model = SAC(policy="CnnPolicy",
                env=env,
                verbose=1,
                device=device)

    date = datetime.now()
    save_path = "./logs/" + env_name + f"{date.month}{date.day}"
    eval_callback = EvalCallback(env,
                                 best_model_save_path=save_path,
                                 log_path=save_path,
                                 eval_freq=50000,  # Evaluate every 10,000 steps
                                 n_eval_episodes=5,  # Evaluate over 5 episodes
                                 deterministic=True,
                                 render=False)

    # Train the model
    model.learn(total_timesteps=1000000, callback=eval_callback)

    # Save the model
    model.save("sac_car_racing")