import gymnasium as gym
from stable_baselines3 import SAC
import CarRacingObstacles.obstacle_ver
import CarRacingObstacles.obstacle_obj
from CarRacingObstacles.wrappers import wrap_CarRacingObst
import cv2
import os


if __name__ == "__main__":
    mdl_name = "CarRacing-obstaclesV2_115_collideP"
    save_VOD = False
    env_name = "CarRacing-obstaclesV2" #'CarRacing-v2'


    # Initialize SAC
    mdl_path = "logs/" + mdl_name + "/best_model.zip"
    model = SAC.load(mdl_path)
    done = False

    if save_VOD:
        # Create CarRacing environment
        env = gym.make(env_name, render_mode='rgb_array', max_episode_steps=1500)
        env = wrap_CarRacingObst(env)
        # model.set_env(env)  # This can use the same wrappers of training
        obs, _ = env.reset(seed=1)

        # prepare recording
        vod_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../VOD')
        vod_name = vod_path + "/" + mdl_name + ".avi"
        frame = env.render()
        frameSize = (frame.shape[1], frame.shape[0])
        recorder = cv2.VideoWriter(vod_name, cv2.VideoWriter_fourcc('m','j','p','g'), 30, frameSize)

        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)

            frame = env.render()
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # Convert to BGR for OpenCV
            cv2.imshow('display', frame)
            cv2.waitKey(10)
            done = truncated or terminated
            recorder.write(frame)
        recorder.release()

    else:
        # Create CarRacing environment
        env = gym.make(env_name, render_mode='human', max_episode_steps=1500)
        env = wrap_CarRacingObst(env)
        obs, _ = env.reset(seed=1)

        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            # print(reward)
            env.render()
            done = truncated or terminated
