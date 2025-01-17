import cv2
import gymnasium as gym
import gymnasium.spaces
import numpy as np
import collections
from gymnasium import spaces
from gymnasium.spaces import Dict, Box
import CarRacingObstacles.obstacle_obj


def wrap_CarRacingObst(env):
    env = MergeGasBrake(env)

    # env = MaxAndSkipEnv(env)
    # env = SkipZoom(env)
    env = ResizeFrame(env, 48)
    env = AddMeasurementObs(env)
    return env


class AddMeasurementObs(gym.ObservationWrapper):
    def __init__(self, env=None):
        super(AddMeasurementObs, self).__init__(env, )

        self.observation_space = Dict({
            "image": env.observation_space,  # 圖像空間來自原始環境
            "data": Box(low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32)  # 新增數值觀察空間
        })

    def observation(self, obs):
        measure = np.array([self.env.true_speed,
                            self.env.car.wheels[0].omega,
                            self.env.car.wheels[1].omega,
                            self.env.car.wheels[2].omega,
                            self.env.car.wheels[3].omega,
                            self.env.car.wheels[0].joint.angle,
                            self.env.car.hull.angularVelocity],
                           dtype=np.float32)

        return {
            "image": obs,
            "data": measure
        }


class MergeGasBrake(gym.Wrapper):
    def __init__(self, env=None):
        super(MergeGasBrake, self).__init__(env)
        # most recent raw observations (for max pooling across time steps)
        if self.continuous:
            self.action_space = spaces.Box(
                np.array([-1, -1]).astype(np.float32),
                np.array([+1, +1]).astype(np.float32),
            )  # steer, gas+brake
        print("Using Merge Brake and Gas Wrapper")

    def step(self, action):
        # merge brake and gas
        if action[1] >=0:
            gas = action[1]
            brake = 0
        else:
            gas = 0
            brake = -action[1]
        action = np.array([action[0], gas, brake])
        return self.env.step(action)


class SkipZoom(gym.Wrapper):
    def __init__(self, env=None):
        super(SkipZoom, self).__init__(env)
        print("wrap with skip zoom")

    def reset(self, seed=None, options=None):
        self.env.reset(seed=seed)
        for _ in range(50):
            obs, _, done, _, _ = self.env.step(np.zeros(self.action_space.shape[0]))
        return obs, {}


class MaxAndSkipEnv(gym.Wrapper):
    def __init__(self, env=None, skip=4):
        """Return only every `skip`-th frame"""
        super(MaxAndSkipEnv, self).__init__(env)
        # most recent raw observations (for max pooling across time steps)
        self._obs_buffer = collections.deque(maxlen=2)
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        terminated, truncated = False, False
        for _ in range(self._skip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            self._obs_buffer.append(obs)
            total_reward += reward
            if terminated or truncated:
                break
        # 堆疊蒐集的畫面，並取同位置上像素的最大值作為觀察
        max_frame = np.max(np.stack(self._obs_buffer), axis=0)
        return max_frame, total_reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        """Clear past frame buffer and init. to first obs. from inner env."""
        self._obs_buffer.clear()
        obs, _ = self.env.reset(seed=seed)
        self._obs_buffer.append(obs)
        return obs, {}


class ResizeFrame(gym.ObservationWrapper):
    def __init__(self, env=None, size=48):
        super(ResizeFrame, self).__init__(env, )
        self.size = size
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(self.size, self.size, 3), dtype=np.uint8)

    def observation(self, obs):
        return ResizeFrame.process(obs, self.size)

    @staticmethod
    def process(frame, size):
        img = np.reshape(frame, [96, 96, 3]).astype(np.float32)
        img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
        img = np.reshape(img, [size, size, 3])
        return img.astype(np.uint8)



if __name__ == "__main__":
    import pygame

    # 建立環境，最大單一回合長度設定為600 steps，視情況自己加長
    continuous = True
    render_mode = 'rgb_array'  #'human' #
    env = gym.make("CarRacing-obstaclesV2", continuous=continuous, render_mode=render_mode, max_episode_steps=1000)
    env = wrap_CarRacingObst(env)
    print(f"action space: {env.action_space.low}~{env.action_space.high}, type: {type(env.action_space.high)}")
    # 設定pygame以接收鍵盤輸入
    pygame.init()
    obs, _ = env.reset(seed=0)
    clock = pygame.time.Clock()
    if continuous:
        steer, gas = 0, 0.3

    step=0
    while True:
        img = obs['image']
        # img = np.moveaxis(img, 0, 2)
        cv2.imshow('obs_visualize', img)
        cv2.waitKey()
        step += 1
        # print(f"step: {step}")
        # 檢查否有按鍵輸入
        keys = pygame.key.get_pressed()
        action = 0
        if continuous:
            steer, brake = 0.1, 0
            if keys[pygame.K_UP]:
                gas += 0.01  #
            elif keys[pygame.K_DOWN]:
                gas -= 0.01
            if keys[pygame.K_LEFT]:
                steer = -0.2
            elif keys[pygame.K_RIGHT]:
                steer = 0.2
            if keys[pygame.K_SPACE]:
                brake = 1.0
            gas = np.clip(gas, 0, 1.0)
            action = [steer, gas, brake]
        else:
            if keys[pygame.K_UP]:
                action = 3  #
            elif keys[pygame.K_DOWN]:
                action = 4
            elif keys[pygame.K_LEFT]:
                action = 2
            elif keys[pygame.K_RIGHT]:
                action = 1
        obs, reward, terminated, truncated, _ = env.step(action)
        # clock.tick(30)
        if terminated or truncated:
            obs, _ = env.reset()

