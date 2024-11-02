import obstacle_ver
import cv2
import gymnasium as gym

if __name__ == "__main__":

    env = gym.make("CarRacing-obstacles", continuous=False, render_mode='rgb_array', max_episode_steps=300)
    obs, _ = env.reset()
    while True:
        obs = cv2.cvtColor(obs, cv2.COLOR_BGR2RGB)
        cv2.imshow('Test', obs)
        cv2.waitKey()
        obs, reward, terminated, truncated, _ = env.step(3)
        if terminated or truncated:
            obs, _ = env.reset()