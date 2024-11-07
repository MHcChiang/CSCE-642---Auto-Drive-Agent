import numpy as np
from typing import Optional, Union
import gymnasium as gym
from gymnasium.envs.box2d.car_racing import CarRacing, FrictionDetector
from gymnasium.envs.box2d.car_dynamics import Car
import Box2D
import pygame
from gymnasium.envs.registration import register
from CarRacingObstacles.utils import polygons_intersect
from gymnasium.utils import seeding
from Box2D.b2 import fixtureDef, polygonShape, revoluteJointDef


STATE_W = 96  # less than Atari 160x192
STATE_H = 96
VIDEO_W = 600
VIDEO_H = 400
WINDOW_W = 1000
WINDOW_H = 800

SCALE = 6.0  # Track scale
ZOOM = 2.7  # Camera zoom
ZOOM_FOLLOW = True  # Set to False for fixed view (don't use zoom)
TRACK_WIDTH = 40 / SCALE

OBSTACLE_CATEGORY = 0x0040  # 障礙物範疇
CAR_CATEGORY = 0x0060  # 汽車範疇


def CarRacingObst(**kwargs):
    env = CarRacingObstacles_v2(**kwargs)
    return env

register(
    id="CarRacing-obstaclesV2",
    entry_point=CarRacingObst,
    max_episode_steps=3000
)


class CarRacingObstacles_v2(CarRacing):
    """
    This class adds obstacles to the CarRacing-v2 environment
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.obstacles = []
        self.obstacles_size = []
        self.contactListener_keepref = FrictionDetectorObstacles(self, self.lap_complete_percent)
        self.world = Box2D.b2World((0, 0), contactListener=self.contactListener_keepref)
        self.contacting = set()
        self.fd_tile = fixtureDef(
            shape=polygonShape(vertices=[(0, 0), (1, 0), (1, -1), (0, -1)]),
            maskBits=0x0060,
        )
        self.collideObst = False

    def step(self, action):
        obs, reward, terminated, truncated, _ = super().step(action)

        # reward -= 0.05 * np.min([self.car.wheels[0].brake, self.car.wheels[0].gas])  # self.car.wheels[0].brake * 0.05
        # reward -= self.car.wheels[0].brake * 0.2

        # early stop
        if len(self.contacting) == 0:  # if get off road
            reward -= 0.1 # if v =/= 0, less
            self.off += 1
            if self.off > 200:
                terminated = True
        else:
            self.off = 0

        if self.collideObst:
            reward -= 0.5
            # print(f"hit, reward: {reward}")

        return obs, reward, terminated, truncated, {}

    def reset(
            self,
            *,
            seed: Optional[int] = None,
            options: Optional[dict] = None,
    ):
        """
            加入了obstacles建構函式
        """
        if seed is not None:
            self._np_random, seed = seeding.np_random(seed)
        self._destroy()
        self.world.contactListener_bug_workaround = FrictionDetectorObstacles(
            self, self.lap_complete_percent
        )
        self.world.contactListener = self.world.contactListener_bug_workaround
        self.reward = 0.0
        self.prev_reward = 0.0
        self.tile_visited_count = 0
        self.t = 0.0
        self.new_lap = False
        self.road_poly = []
        self.obstacles_size = []

        if self.domain_randomize:
            randomize = True
            if isinstance(options, dict):
                if "randomize" in options:
                    randomize = options["randomize"]

            self._reinit_colors(randomize)

        while True:
            success = self._create_track()
            if success:
                break
            if self.verbose:
                print(
                    "retry to generate track (normal if there are not many"
                    "instances of this message)"
                )
        self.car = Car(self.world, *self.track[0][1:4])

        if self.render_mode == "human":
            self.render()

        self._create_obstacles()
        self.contacting.clear()
        self.collideObst = False
        self.off = 0

        return self.step(None)[0], {}

    def render(self):
        if self.render_mode is None:
            assert self.spec is not None
            gym.logger.warn(
                "You are calling render method without specifying any render mode. "
                "You can specify the render_mode at initialization, "
                f'e.g. gym.make("{self.spec.id}", render_mode="rgb_array")'
            )
            return
        else:
            return self._render(self.render_mode)

    def _create_obstacles(self):
        """
        以一定間距在跑道上生成 15 個障礙物
        :return:
        """
        # 以一定間距生成障礙物
        num_obstacles = 15
        max_ind_step = len(self.track) / num_obstacles
        min_ind_step = 10
        if max_ind_step < min_ind_step:
            raise ValueError('Too many obstacles, please reduce num of obst')
        obstables_loc_ind = []
        index = 0
        for _ in range(num_obstacles):
            index += self.np_random.integers(min_ind_step, max_ind_step)
            obstables_loc_ind.append(index)


        for i, track_ind in enumerate(obstables_loc_ind):
            alpha1, beta1, x1, y1 = self.track[track_ind]
            o_width = TRACK_WIDTH / (6 - self.np_random.uniform(0, 2))

            obstacle = self.world.CreateDynamicBody(position=(x1, y1))
            obs_fixture = obstacle.CreateFixture(
                shape=Box2D.b2PolygonShape(box=(o_width/2, o_width/2)),
                density=5.0,
                friction=0.5,
                restitution=0.5,
                )
            obs_fixture.filterData.categoryBits = OBSTACLE_CATEGORY  # 障礙物類別
            obs_fixture.filterData.maskBits = CAR_CATEGORY
            obstacle.mass = 100.0

            obstacle.userData = obstacle
            obstacle.color = np.array([255, 0, 0])
            obstacle.obs_id = i

            self.obstacles_size.append(o_width)
            self.obstacles.append(obstacle)

        for fixture in self.car.hull.fixtures:
            fixture.filterData.categoryBits = CAR_CATEGORY
            fixture.filterData.maskBits = OBSTACLE_CATEGORY | 0x001
            fixture.restitution = 0.1

    def _render(self, mode: str):
        assert mode in self.metadata["render_modes"]

        pygame.font.init()
        if self.screen is None and mode == "human":
            pygame.init()
            pygame.display.init()
            self.screen = pygame.display.set_mode((WINDOW_W, WINDOW_H))
        if self.clock is None:
            self.clock = pygame.time.Clock()

        if "t" not in self.__dict__:
            return  # reset() not called yet

        self.surf = pygame.Surface((WINDOW_W, WINDOW_H))

        assert self.car is not None
        # computing transformations
        angle = -self.car.hull.angle
        # Animating first second zoom.
        zoom = 0.1 * SCALE * max(1 - self.t, 0) + ZOOM * SCALE * min(self.t, 1)
        scroll_x = -(self.car.hull.position[0]) * zoom
        scroll_y = -(self.car.hull.position[1]) * zoom
        trans = pygame.math.Vector2((scroll_x, scroll_y)).rotate_rad(angle)
        trans = (WINDOW_W / 2 + trans[0], WINDOW_H / 4 + trans[1])

        self._render_road(zoom, trans, angle)
        self._render_obstacles(zoom, trans, angle)
        self.car.draw(
            self.surf,
            zoom,
            trans,
            angle,
            mode not in ["state_pixels_list", "state_pixels"],
        )

        self.surf = pygame.transform.flip(self.surf, False, True)

        # showing stats
        self._render_indicators(WINDOW_W, WINDOW_H)

        font = pygame.font.Font(pygame.font.get_default_font(), 42)
        text = font.render("%04i" % self.reward, True, (255, 255, 255), (0, 0, 0))
        text_rect = text.get_rect()
        text_rect.center = (60, WINDOW_H - WINDOW_H * 2.5 / 40.0)
        self.surf.blit(text, text_rect)

        if mode == "human":
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            assert self.screen is not None
            self.screen.fill(0)
            self.screen.blit(self.surf, (0, 0))
            pygame.display.flip()
        elif mode == "rgb_array":
            return self._create_image_array(self.surf, (VIDEO_W, VIDEO_H))
        elif mode == "state_pixels":
            return self._create_image_array(self.surf, (STATE_W, STATE_H))
        else:
            return self.isopen

    def _render_obstacles(self, zoom, translation, angle):
        """
        :param zoom: same
        :param translation: same
        :param angle: same
        :return: None
        """
        color = (255, 0, 0)
        for i, obstacle in enumerate(self.obstacles):
            position = obstacle.position
            x, y = position.x, position.y
            o_width = self.obstacles_size[i]
            # print(type(x), type(o_width))
            obst_poly = [(x + o_width / 2, y + o_width / 2),
                         (x + o_width / 2, y - o_width / 2),
                         (x - o_width / 2, y - o_width / 2),
                         (x - o_width / 2, y + o_width / 2)]
            self._draw_colored_polygon(self.surf, obst_poly, color, zoom, translation, angle)

    def _destroy(self):
        if not self.road:
            return
        for t in self.road:
            self.world.DestroyBody(t)
        for obst in self.obstacles:
            self.world.DestroyBody(obst)
        self.road = []
        self.obstacles = []
        assert self.car is not None
        self.car.destroy()


class FrictionDetectorObstacles(FrictionDetector):
    def __init__(self, env, lap_complete_percent):
        super().__init__(env, lap_complete_percent)

    def BeginContact(self, contact):
        self._contact(contact, True)

    def EndContact(self, contact):
        self._contact(contact, False)

    def _contact(self, contact, begin):
        tile = None
        obj = None
        obst = None
        u1 = contact.fixtureA.body.userData
        u2 = contact.fixtureB.body.userData
        if u1 and "road_friction" in u1.__dict__:
            tile = u1
            obj = u2
        if u2 and "road_friction" in u2.__dict__:
            tile = u2
            obj = u1
        if u1 and "obs_id" in u1.__dict__:
            obst = u1
        if u2 and "obs_id" in u2.__dict__:
            obst = u2

        if not tile and not obst:
            return

        if tile:
            # inherit tile color from env
            tile.color[:] = self.env.road_color
            if not obj or "tiles" not in obj.__dict__:
                return

            if begin:
                obj.tiles.add(tile)
                self.env.current_road_id = tile.idx
                self.env.contacting.add(tile.idx)
                if not tile.road_visited:
                    tile.road_visited = True
                    self.env.reward += 1000.0 / len(self.env.track)
                    self.env.tile_visited_count += 1

                    # Lap is considered completed if enough % of the track was covered
                    if (
                        tile.idx == 0
                        and self.env.tile_visited_count / len(self.env.track)
                        > self.lap_complete_percent
                    ):
                        self.env.new_lap = True
            else:
                obj.tiles.remove(tile)
                self.env.contacting.discard(tile.idx)

        if obst:
            if begin:
                self.env.collideObst = True

            else:
                self.env.collideObst = False



if __name__ == "__main__":

    # 建立環境，最大單一回合長度設定為600 steps，視情況自己加長
    continuous = True
    render_mode = 'human'  #'rgb_array' #
    env = gym.make("CarRacing-obstaclesV2", continuous=continuous, render_mode=render_mode, max_episode_steps=500)

    # 設定pygame以接收鍵盤輸入
    pygame.init()
    obs, _ = env.reset(seed=0)
    clock = pygame.time.Clock()
    if continuous:
        steer, gas = 0, 0
    while True:
        # cv2.imshow('obs', obs)
        # cv2.waitKey()
        # 檢查否有按鍵輸入
        keys = pygame.key.get_pressed()
        # 針對上下左右鍵分配action
        # discrete actions: do nothing, steer left, steer right, gas, brake.
        action = 0
        if continuous:
            steer, brake = 0, 0
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
        clock.tick(30)
        # if reward > 0:
        #     print(reward)
        if terminated or truncated:
            obs, _ = env.reset(seed=0)


