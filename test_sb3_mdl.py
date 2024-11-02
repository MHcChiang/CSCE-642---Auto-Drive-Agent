import gymnasium as gym
from stable_baselines3 import SAC
import CarRacingObstacles.obstacle_ver
import CarRacingObstacles.obstacle_obj
from CarRacingObstacles.wrappers import wrap_CarRacingObst


if __name__ == "__main__":
    env_name = "CarRacing-obstaclesV2" #'CarRacing-v2'
    # Create CarRacing environment
    env = gym.make(env_name, render_mode='human', max_episode_steps=1000)
    env = wrap_CarRacingObst(env)

    # Initialize SAC
    model = SAC.load("logs/CarRacing_test/best_model.zip")

    obs, _ = env.reset(seed=1)
    done = False
    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        # print(reward)
        env.render()
        done = truncated or terminated
