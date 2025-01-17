import gymnasium as gym
from stable_baselines3 import SAC
import CarRacingObstacles.obstacle_obj
from CarRacingObstacles.wrappers import wrap_CarRacingObst
from CarRacingObstacles.utils import wrap_eval_env
import cv2
import os

def build_parser():
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--mdl', type=str, default=None)
    parser.add_argument('--env_name', type=str, default="CarRacing-obstaclesV2")
    parser.add_argument('--save_VOD', action='store_true')
    parser.add_argument('--seed', type=int, default=1)

    args = parser.parse_args()
    params = vars(args)
    return params


if __name__ == "__main__":
    params = build_parser()

    mdl_path = params['mdl']
    mdl_name = mdl_path.split('/')[-1]

    # Initialize SAC
    model = SAC.load(mdl_path)
    done = False

    if params['save_VOD']:
        # Create CarRacing environment
        env = gym.make(params['env_name'], render_mode='rgb_array', max_episode_steps=1000)
        env = wrap_CarRacingObst(env)
        # model.set_env(env)  # This can use the same wrappers of training
        obs, _ = env.reset(seed=params['seed'])

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
        print(f"Success to save to: {vod_name}")

    else:
        # Create CarRacing environment
        env = gym.make(params['env_name'], render_mode='human', max_episode_steps=1000)
        env = wrap_CarRacingObst(env)
        env = wrap_eval_env(env)
        obs = env.reset()

        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            env.render()
