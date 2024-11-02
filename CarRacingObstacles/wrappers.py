import cv2
import gymnasium as gym
import gymnasium.spaces
import numpy as np
import collections
import CarRacingObstacles.obstacle_ver
import CarRacingObstacles.obstacle_obj


def wrap_CarRacingObst(env):
    # env = MaxAndSkipEnv(env)
    env = SkipZoom(env)

    # env = ProcessFrame(env)
    env = ResizeFrame(env)

    # env = ImageToPyTorch(env)
    # env = BufferWrapper(env, 4)
    # env = ScaledFloatFrame(env)
    return env


class SkipZoom(gym.Wrapper):
    def __init__(self, env=None):
        """For environments where the user need to press FIRE for the game to start."""
        super(SkipZoom, self).__init__(env)

    def reset(self, seed=None, options=None):
        self.env.reset(seed=seed)
        for _ in range(10):
            obs, _, done, _, _ = self.env.step([0,0,0])
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


class ProcessFrame(gym.ObservationWrapper):
    def __init__(self, env=None):
        super(ProcessFrame, self).__init__(env)
        self.size = 72
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(self.size, self.size, 1), dtype=np.uint8)


    def observation(self, obs):
        return ProcessFrame.process(obs, self.size)

    @staticmethod
    def process(frame, size):
        # 轉換色彩通道
        img = np.reshape(frame, [96, 96, 3]).astype(np.float32)
        # 轉成灰階
        img = img[:, :, 0] * 0.299 + img[:, :, 1] * 0.587 + img[:, :, 2] * 0.114
        img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
        img = np.reshape(img, [size, size, 1])
        return img.astype(np.uint8)


class ImageToPyTorch(gym.ObservationWrapper):
    def __init__(self, env):
        super(ImageToPyTorch, self).__init__(env)
        old_shape = self.observation_space.shape
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(old_shape[-1], old_shape[0], old_shape[1]),
                                                dtype=np.float32)

    def observation(self, observation):
        return np.moveaxis(observation, 2, 0)


class ScaledFloatFrame(gym.ObservationWrapper):
    def observation(self, obs):
        return np.array(obs).astype(np.float32) / 255.0


class BufferWrapper(gym.ObservationWrapper):
    def __init__(self, env, n_steps, dtype=np.float32):
        super(BufferWrapper, self).__init__(env)
        self.dtype = dtype
        old_space = env.observation_space
        self.observation_space = gym.spaces.Box(old_space.low.repeat(n_steps, axis=0),
                                                old_space.high.repeat(n_steps, axis=0), dtype=dtype)

    def reset(self, seed=None, options=None):
        self.buffer = np.zeros_like(self.observation_space.low, dtype=self.dtype)
        return self.observation(self.env.reset(seed=seed)[0]), {}

    def observation(self, observation):
        self.buffer[:-1] = self.buffer[1:]
        self.buffer[-1] = observation
        return self.buffer


class ResizeFrame(gym.ObservationWrapper):
    def __init__(self, env=None, size=48):
        super(ResizeFrame, self).__init__(env, )
        self.size = size
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(self.size, self.size, 3), dtype=np.uint8)

    def observation(self, obs):
        return ResizeFrame.process(obs, self.size)

    @staticmethod
    def process(frame, size):
        # 轉換色彩通道
        img = np.reshape(frame, [96, 96, 3]).astype(np.float32)
        img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
        img = np.reshape(img, [size, size, 3])
        return img.astype(np.uint8)


if __name__ == "__main__":
    import pygame

    # 建立環境，最大單一回合長度設定為600 steps，視情況自己加長
    continuous = True
    render_mode = 'rgb_array'  #'human' #
    env = gym.make("CarRacing-obstaclesV2", continuous=continuous, render_mode=render_mode, max_episode_steps=1200)
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
        # img = np.moveaxis(obs, 0, 2)
        img = obs
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