import gymnasium as gym
from stable_baselines3 import SAC
import CarRacingObstacles.obstacle_obj
from CarRacingObstacles.wrappers import wrap_CarRacingObst
from CarRacingObstacles.utils import wrap_eval_env
import cv2
import os


if __name__ == "__main__":
    mdl_name = "CarRacing-obstaclesV2_1129_Ts0ApSteer2"
    check = None #40 * 10000 # None
    best = 33 * 10000
    env_name = "CarRacing-obstaclesV2"  # 'CarRacing-v2'
    save_VOD = False
    seed = 1

    # Initialize SAC
    if check:
        mdl_path = "logs/" + mdl_name + f"/Checkpoint_{check}_steps.zip"
    elif best:
        mdl_path = "logs/" + mdl_name + f"/best_model_{best}.zip"
    else:
        mdl_path = "logs/" + mdl_name + f"/best_model.zip"

    model = SAC.load(mdl_path)
    done = False
    print(model.policy.critic)

    if save_VOD:
        # Create CarRacing environment
        env = gym.make(env_name, render_mode='rgb_array', max_episode_steps=1500)
        env = wrap_CarRacingObst(env)
        # model.set_env(env)  # This can use the same wrappers of training
        obs, _ = env.reset(seed=seed)

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
        obs, _ = env.reset(seed=seed)

        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = truncated or terminated
            # print(action)
            env.render()
