"""
Classic cart-pole system implemented by Rich Sutton et al.
Copied from http://incompleteideas.net/sutton/book/code/pole.c
permalink: https://perma.cc/C9ZM-652R
"""
import math
from typing import Optional, Tuple, Union
import numpy as np
import gymnasium as gym
from gymnasium import logger, spaces
from gymnasium.envs.classic_control import utils
from gymnasium.error import DependencyNotInstalled
from gymnasium.vector import VectorEnv
from gymnasium.vector.utils import batch_space




class myCartPoleEnvF(gym.Env[np.ndarray, Union[int, np.ndarray]]):

# import math
# import gym
# from gym import spaces, logger
# from gym.utils import seeding
# import numpy as np

# class myCartPoleEnvF(gym.Env):
    """
    Description:
        A pole is attached by an un-actuated joint to a cart, which moves along a frictionless track. The pendulum starts upright, and the goal is to prevent it from falling over by increasing and reducing the cart's velocity.

    Source:
        This environment corresponds to the version of the cart-pole problem described by Barto, Sutton, and Anderson

    Observation: 
        Type: Box(4)
        Num	Observation                 Min                         Max
        0	Cart Position             -4.8                          4.8
        1	Cart Velocity             -Inf                          Inf
        2	Pole Angle                -24 deg (-0.418 rad)          24 deg (0.418 rad)
        3	Pole Velocity At Tip      -Inf                          Inf
        
    Actions:
        Type: Box(1)
        Num	Observation                 Min         Max
        0	Voltage to motor            -10          10     

        #so, input from -1 to 1, then multiplied by max_volt
        
        Note: The amount the velocity that is reduced or increased is not fixed; 
        it depends on the angle the pole is pointing. This is because the center of gravity of the pole increases the amount of energy needed to move the cart underneath it

    Reward:
        Reward is 1 for every step taken, including the termination step

    Starting State:
        All observations are assigned a uniform random value in [-0.05..0.05]

    Episode Termination:
        Pole Angle is more than 90 degrees #12
        Cart Position is more than 2.4 (center of the cart reaches the edge of the display)
        Episode length is greater than 200
        Solved Requirements
        Considered solved when the average reward is greater than or equal to 195.0 over 100 consecutive trials.
    """
    
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 50,
    }

    def __init__(self, render_mode: Optional[str] = None):
        self.fps = 100 # Nikki changed from 50 to 100 
        
        self.gravity = 9.81
        self.masscart = 0.57+0.37
        self.masspole = 0.230
        self.total_mass = (self.masspole + self.masscart)
        self.length = 0.3302  # actually half the pole's length
        self.polemass_length = (self.masspole * self.length)
        self.force_mag = 10.0 # should be 8 for our case?
        self.tau = 1/self.fps  # seconds between state updates
        # self.tau = 0.002
        # self.tau = 0.02
        self.metadata = {
            "render_modes": ["human", "rgb_array"],
            "render_fps": int(np.round(1.0 / self.tau)),
        }

        # self.kinematics_integrator = "semi-implicit-euler" # newly added from native CartPole
        self.kinematics_integrator = "RK4"
        
        # self.min_action = -1.0 # min normalized voltage
        # self.max_action = 1.0 # max normalized voltage
        
        # from Emi's thesis
        self.r_mp = 6.35e-3 # motor pinion radius
        self.Jm = 3.90e-7 # rotor moment of inertia
        self.Kg = 3.71 # planetary gearbox gear ratio
        self.Rm = 2.6 # motor armature resistance
        self.Kt = 0.00767 # motor torque constant
        self.Km = 0.00767 # Back-ElectroMotive-Force (EMF) Constant V.s/RAD
        # both of these are in N.m.s/RAD, not degrees
        self.Beq = 5.4 #  equivalent viscous damping coecient as seen at the motor pinion
        self.Bp = 0.0024 # viscous damping doecient, as seen at the pendulum axis
        
        """

        # self.fps = 50
        gravity = 9.81
        masscart = 0.57+0.37
        masspole = 0.230
        total_mass = masspole + masscart
        length = 0.3302  # actually half the pole's length
        # polemass_length = (self.masspole * self.length)
        force_mag = 10.0 # should be 8 for our case?
        tau = 0.02  # seconds between state updates
        
        # from Emi's thesis
    
        r_mp = 6.35e-3 # motor pinion radius
        Jm = 3.90e-7 # rotor moment of inertia
        Kg = 3.71 # planetary gearbox gear ratio
        Rm = 2.6 # motor armature resistance
        # both of these are in N.m.s/RAD, not degrees
        Beq = 5.4 #  equivalent viscous damping coecient as seen at the motor pinion
        Bp = 0.0024 # viscous damping doecient, as seen at the pendulum axis
        
        # dynamics for actuator
        Kt = 0.00767 # motor torque constant
        Km = 0.00767 # Back-ElectroMotive-Force (EMF) Constant V.s/RAD

        """
        
        

        # Angle at which to fail the episode
        # self.theta_threshold_radians = 12  * math.pi / 180 # according to Dylan's thesis
        self.theta_threshold_radians = 0.2 # approximatedly 12 deg
        # self.theta_threshold_radians = 36  * math.pi / 180 # this might be too wide
        # self.x_threshold = 2.4 # this value is 2.4 in native gym cartpole env but Dylan had 0.4
        self.x_threshold = 0.2 # lab result triggers watchdog, Nikki changed from 0.25 to 0.2

        # Angle limit set to 2 * theta_threshold_radians so failing observation
        # is still within bounds
        # recall state = x, x_dot, theta, theta_dot !!!
        high = np.array(
                [
                    self.x_threshold * 2,
                    np.finfo(np.float32).max,
                    self.theta_threshold_radians * 2,
                    np.finfo(np.float32).max
                ],
                dtype = np.float32)
        # normalized volage
        volt = 1;  

        self.action_space = spaces.Box(low=-volt, high=volt, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)
        
        #from new native CartPole
        self.render_mode = render_mode

        self.screen_width = 600
        self.screen_height = 400
        self.screen = None
        self.clock = None
        self.isopen = True
        self.state = None

        self.steps_beyond_terminated = None
        
        # seems to be outdated
        """
        self.seed()
        self.viewer = None
        self.state = None

        self.steps_beyond_done = None

        def seed(self, seed=None):
            self.np_random, seed = seeding.np_random(seed)
        return [seed]
        """

        """
        ## In native CartPole beginning part - possibly already has integrated seeding
        >>> env.reset(seed=123, options={"low": 0, "high": 1})
        (array([0.6823519 , 0.05382102, 0.22035988, 0.18437181], dtype=float32), {})
        """

    def stepPhysics(self, force):
        # assert self.action_space.contains(
        #     action
        # ), f"{action!r} ({type(action)}) invalid"

        assert self.state is not None, "Call reset before using step method."
        # x, theta, x_dot, theta_dot = self.state
        # to be consistent with native CartPole state = x, x_dot, theta, theta_dot
        x, x_dot, theta, theta_dot = self.state
        
        costheta = math.cos(theta)
        sintheta = math.sin(theta)

        # denominator used in a bunch of stuff
        d = 4 * self.masscart * self.r_mp**2 + self.masspole * self.r_mp**2 + 4 * self.Jm * self.Kg**2         

        
        xacc = ((-4 * (self.Rm * self.r_mp**2 * self.Beq + self.Kg**2 * self.Kt * self.Km)) / (self.Rm *(d + 3 * self.r_mp**2 * self.masspole * sintheta**2))) * x_dot + ((-3 * self.Bp * self.r_mp**2 * costheta) / (self.length * (d + 3 * self.r_mp**2 * self.masspole * sintheta**2))) * theta_dot + ((-4 * self.masspole * self.length * self.r_mp**2 * sintheta) / (d + 3 * self.r_mp**2 * self.masspole * sintheta**2)) * theta_dot**2 + ((3 * self.masspole * self.gravity * self.r_mp**2 * costheta * sintheta) / (d + 3 * self.r_mp**2 * self.masspole * sintheta**2)) + (4 * self.r_mp * self.Kg * self.Kt) / (self.Rm * (d + 3 * self.r_mp**2 * self.masspole * sintheta**2)) * force

        thetaacc = ((-3 * (self.Rm * self.r_mp**2 * self.Beq + self.Kg**2 * self.Kt * self.Km) * costheta) / (self.length * self.Rm * (d + 3 * self.r_mp**2 * self.masspole * sintheta**2))) * x_dot + ((-3 * (self.masscart * self.r_mp**2 + self.masspole * self.r_mp**2 + self.Jm * self.Kg**2) * self.Bp) / (self.masspole * self.length**2 * (d + 3 * self.r_mp**2 * self.masspole * sintheta**2))) * theta_dot + ((-3 * self.masspole * self.r_mp**2 * sintheta * costheta) / (d + 3 * self.r_mp**2 * self.masspole * sintheta**2)) * theta_dot**2 + ((3 * (self.masscart * self.r_mp**2 + self.masspole * self.r_mp**2 + self.Jm * self.Kg**2) * self.gravity * sintheta) / (self.length * (d + 3 * self.r_mp**2 * self.masspole * sintheta**2))) + (3 * self.r_mp * self.Kg * self.Kt * costheta) / (self.length * self.Rm * (d + 3 * self.r_mp**2 * self.masspole * sintheta**2)) * force

        if self.kinematics_integrator == "euler":
            x = x + self.tau * x_dot
            x_dot = x_dot + self.tau * xacc
            theta = theta + self.tau * theta_dot
            theta_dot = theta_dot + self.tau * thetaacc
        else:  # semi-implicit euler
            x_dot = x_dot + self.tau * xacc
            x = x + self.tau * x_dot
            theta_dot = theta_dot + self.tau * thetaacc
            theta = theta + self.tau * theta_dot

            """
            # RK4
            k1 = RHS(y, t = t[i], Vm = force)
            k2 = RHS(y + h * k1 / 2, t = t[i] + h / 2, Vm = Vm)
            k3 = RHS(y + h * k2 / 2, t = t[i] + h / 2, Vm = Vm)
            k4 = RHS(y + h * k3, t = t[i] + h, Vm = Vm)
            y = y + h/6 * (k1 + 2 * k2 + 2 * k3 + k4)

            """
        # self.state = (x, x_dot, theta, theta_dot)
        return np.array( (x, x_dot, theta, theta_dot), dtype = np.float32)

    def step(self, action):
        # Cast action to float to strip np trappings
        force = self.force_mag * float(action)
        # force = float(np.clip( self.force_mag * action[0], -self.force_mag, self.force_mag))
        self.state = self.stepPhysics(force) # generate state and then step?
        
        # x, theta, x_dot, theta_dot = self.state
        x, x_dot, theta, theta_dot = self.state
        
        terminated = bool(
                (np.abs(theta) > self.theta_threshold_radians)
                or (np.abs(x) > self.x_threshold)
                ) #NX Changed to the above from
        """
            x < -self.x_threshold
            or x > self.x_threshold
            or theta < -self.theta_threshold_radians
            or theta > self.theta_threshold_radians
        """

        
        if not terminated:
            reward = 1.0
        elif self.steps_beyond_terminated is None:
            # Pole just fell!
            self.steps_beyond_terminated = 0
            reward = 1.0

        else:
            if self.steps_beyond_terminated == 0:
                logger.warn(
                    "You are calling 'step()' even though this "
                    "environment has already returned terminated = True. You "
                    "should always call 'reset()' once you receive 'terminated = "
                    "True' -- any further steps are undefined behavior."
                )
            self.steps_beyond_terminated += 1
            reward = 0.0
        """
        #NX changed from above to below
        reward = int(not terminated)
        info = {"reward_survive": reward}
        """

        if self.render_mode == "human":
            self.render()
        return np.array(self.state, dtype=np.float32), reward, terminated, False, {}
                 # quad_reward, terminated, False, {}



    ## This def is copied from native cartpole
    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):
        super().reset(seed=seed)
        # Note that if you use custom reset bounds, it may lead to out-of-bound
        # state/observations.
        low, high = utils.maybe_parse_reset_bounds(
            # options, -0.05, 0.05  # default low
            options, -0.08, 0.08 #NX changed from above
        )  # default high
        self.state = self.np_random.uniform(low=low, high=high, size=(4,))
        # self.state = np.array(
        #         [ self.np_random.uniform(low = -0.08, high = 0.08), 
        #           0 , 
        #           self.np_random.uniform(low = -0.05, high = 0.05), 
        #           0 ]
        #         )
        self.steps_beyond_terminated = None

        if self.render_mode == "human":
            self.render()
        return np.array(self.state, dtype=np.float32), {}

    
    """
    # this def was from Dylan

    def render(self, mode='human'):
        screen_width = 600
        screen_height = 400

        world_width = self.x_threshold * 2
        scale = screen_width /world_width #ie pixels per m
        carty = 100  # TOP OF CART
        polewidth = 10.0
        polelen = scale * self.length #should be *2 again, but I want it to fit on screen
        cartwidth = 50.0
        cartheight = 30.0

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)
            l, r, t, b = -cartwidth / 2, cartwidth / 2, cartheight / 2, -cartheight / 2
            axleoffset = cartheight / 4.0
            cart = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            self.carttrans = rendering.Transform()
            cart.add_attr(self.carttrans)
            self.viewer.add_geom(cart)
            l, r, t, b = -polewidth / 2, polewidth / 2, polelen-polewidth / 2, -polewidth / 2
            pole = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            pole.set_color(.8, .6, .4)
            self.poletrans = rendering.Transform(translation=(0, axleoffset))
            pole.add_attr(self.poletrans)
            pole.add_attr(self.carttrans)
            self.viewer.add_geom(pole)
            self.axle = rendering.make_circle(polewidth / 2)
            self.axle.add_attr(self.poletrans)
            self.axle.add_attr(self.carttrans)
            self.axle.set_color(.5, .5, .8)
            self.viewer.add_geom(self.axle)
            self.track = rendering.Line((0, carty), (screen_width, carty))
            self.track.set_color(0, 0, 0)
            self.viewer.add_geom(self.track)

        if self.state is None:
            return None

        x = self.state
        cartx = x[0] * scale + screen_width / 2.0  # MIDDLE OF CART
        self.carttrans.set_translation(cartx, carty)
        self.poletrans.set_rotation(x[1])

        return self.viewer.render(return_rgb_array=(mode == 'rgb_array'))

    def close(self):
        if self.viewer:
            self.viewer.close()
    """

    ## this render def is copied from CartPole

    def render(self):
        if self.render_mode is None:
            assert self.spec is not None
            gym.logger.warn(
                "You are calling render method without specifying any render mode. "
                "You can specify the render_mode at initialization, "
                f'e.g. gym.make("{self.spec.id}", render_mode="rgb_array")'
            )
            return

        try:
            import pygame
            from pygame import gfxdraw
        except ImportError as e:
            raise DependencyNotInstalled(
                "pygame is not installed, run `pip install gymnasium[classic-control]`"
            ) from e

        if self.screen is None:
            pygame.init()
            if self.render_mode == "human":
                pygame.display.init()
                self.screen = pygame.display.set_mode(
                    (self.screen_width, self.screen_height)
                )
            else:  # mode == "rgb_array"
                self.screen = pygame.Surface((self.screen_width, self.screen_height))
        if self.clock is None:
            self.clock = pygame.time.Clock()

        world_width = self.x_threshold * 2
        scale = self.screen_width / world_width
        polewidth = 10.0
        polelen = scale * (2 * self.length)
        cartwidth = 50.0
        cartheight = 30.0

        if self.state is None:
            return None

        x = self.state

        self.surf = pygame.Surface((self.screen_width, self.screen_height))
        self.surf.fill((255, 255, 255))

        l, r, t, b = -cartwidth / 2, cartwidth / 2, cartheight / 2, -cartheight / 2
        axleoffset = cartheight / 4.0
        cartx = x[0] * scale + self.screen_width / 2.0  # MIDDLE OF CART
        carty = 100  # TOP OF CART
        cart_coords = [(l, b), (l, t), (r, t), (r, b)]
        cart_coords = [(c[0] + cartx, c[1] + carty) for c in cart_coords]
        gfxdraw.aapolygon(self.surf, cart_coords, (0, 0, 0))
        gfxdraw.filled_polygon(self.surf, cart_coords, (0, 0, 0))

        l, r, t, b = (
            -polewidth / 2,
            polewidth / 2,
            polelen - polewidth / 2,
            -polewidth / 2,
        )

        pole_coords = []
        for coord in [(l, b), (l, t), (r, t), (r, b)]:
            coord = pygame.math.Vector2(coord).rotate_rad(-x[2])
            coord = (coord[0] + cartx, coord[1] + carty + axleoffset)
            pole_coords.append(coord)
        gfxdraw.aapolygon(self.surf, pole_coords, (202, 152, 101))
        gfxdraw.filled_polygon(self.surf, pole_coords, (202, 152, 101))

        gfxdraw.aacircle(
            self.surf,
            int(cartx),
            int(carty + axleoffset),
            int(polewidth / 2),
            (129, 132, 203),
        )
        gfxdraw.filled_circle(
            self.surf,
            int(cartx),
            int(carty + axleoffset),
            int(polewidth / 2),
            (129, 132, 203),
        )

        gfxdraw.hline(self.surf, 0, self.screen_width, carty, (0, 0, 0))

        self.surf = pygame.transform.flip(self.surf, False, True)
        self.screen.blit(self.surf, (0, 0))
        if self.render_mode == "human":
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            pygame.display.flip()

        elif self.render_mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
            )

    def close(self):
        if self.screen is not None:
            import pygame

            pygame.display.quit()
            pygame.quit()
            self.isopen = False

    

# below is copied without modification from native cartpole

class myCartPoleFVectorEnv(VectorEnv):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 50,
    }

    def __init__(
        self,
        num_envs: int = 2,
        max_episode_steps: int = 500,
        render_mode: Optional[str] = None,
    ):
        self.num_envs = num_envs
        self.max_episode_steps = max_episode_steps
        self.render_mode = render_mode

        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = self.masspole + self.masscart
        self.length = 0.5  # actually half the pole's length
        self.polemass_length = self.masspole * self.length
        self.force_mag = 10.0
        self.tau = 0.02  # seconds between state updates
        self.kinematics_integrator = "euler"

        self.steps = np.zeros(num_envs, dtype=np.int32)

        # Angle at which to fail the episode
        self.theta_threshold_radians = 12 * 2 * math.pi / 360
        self.x_threshold = 2.4

        # Angle limit set to 2 * theta_threshold_radians so failing observation
        # is still within bounds.
        high = np.array(
            [
                self.x_threshold * 2,
                np.finfo(np.float32).max,
                self.theta_threshold_radians * 2,
                np.finfo(np.float32).max,
            ],
            dtype=np.float32,
        )

        self.low = -0.05
        self.high = 0.05

        self.single_action_space = spaces.Discrete(2)
        self.action_space = batch_space(self.single_action_space, num_envs)
        self.single_observation_space = spaces.Box(-high, high, dtype=np.float32)
        self.observation_space = batch_space(self.single_observation_space, num_envs)

        self.screen_width = 600
        self.screen_height = 400
        self.screens = None
        self.clocks = None
        self.isopen = True
        self.state = None

        self.steps_beyond_terminated = None

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
        assert self.action_space.contains(
            action
        ), f"{action!r} ({type(action)}) invalid"
        assert self.state is not None, "Call reset before using step method."
        ##Here is the hyperparameter of state each step, for four dimensions
        x, x_dot, theta, theta_dot = self.state
        force = np.sign(action - 0.5) * self.force_mag
        costheta = np.cos(theta)
        sintheta = np.sin(theta)

        # For the interested reader:
        # https://coneural.org/florian/papers/05_cart_pole.pdf
        temp = (
            force + self.polemass_length * theta_dot**2 * sintheta
        ) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (
            self.length * (4.0 / 3.0 - self.masspole * costheta**2 / self.total_mass)
        )
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass

        if self.kinematics_integrator == "euler":
            x = x + self.tau * x_dot
            x_dot = x_dot + self.tau * xacc
            theta = theta + self.tau * theta_dot
            theta_dot = theta_dot + self.tau * thetaacc
        else:  # semi-implicit euler
            x_dot = x_dot + self.tau * xacc
            x = x + self.tau * x_dot
            theta_dot = theta_dot + self.tau * thetaacc
            theta = theta + self.tau * theta_dot

        self.state = np.stack((x, x_dot, theta, theta_dot))

        terminated: np.ndarray = (
            (x < -self.x_threshold)
            | (x > self.x_threshold)
            | (theta < -self.theta_threshold_radians)
            | (theta > self.theta_threshold_radians)
        )

        self.steps += 1

        truncated = self.steps >= self.max_episode_steps

        done = terminated | truncated

        if any(done):
            # This code was generated by copilot, need to check if it works
            self.state[:, done] = self.np_random.uniform(
                low=self.low, high=self.high, size=(4, done.sum())
            ).astype(np.float32)
            self.steps[done] = 0

        reward = np.ones_like(terminated, dtype=np.float32)

        if self.render_mode == "human":
            self.render()

        return self.state.T, reward, terminated, truncated, {}

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):
        super().reset(seed=seed)
        # Note that if you use custom reset bounds, it may lead to out-of-bound
        # state/observations.
        self.low, self.high = utils.maybe_parse_reset_bounds(
            options, -0.05, 0.05  # default low
        )  # default high
        self.state = self.np_random.uniform(
            low=self.low, high=self.high, size=(4, self.num_envs)
        ).astype(np.float32)
        self.steps_beyond_terminated = None

        if self.render_mode == "human":
            self.render()
        return self.state.T, {}

    def render(self):
        if self.render_mode is None:
            assert self.spec is not None
            gym.logger.warn(
                "You are calling render method without specifying any render mode. "
                "You can specify the render_mode at initialization, "
                f'e.g. gym("{self.spec.id}", render_mode="rgb_array")'
            )
            return

        try:
            import pygame
            from pygame import gfxdraw
        except ImportError:
            raise DependencyNotInstalled(
                "pygame is not installed, run `pip install gymnasium[classic_control]`"
            )

        if self.screens is None:
            pygame.init()
            if self.render_mode == "human":
                pygame.display.init()
                self.screens = [
                    pygame.display.set_mode((self.screen_width, self.screen_height))
                    for _ in range(self.num_envs)
                ]
            else:  # mode == "rgb_array"
                self.screens = [
                    pygame.Surface((self.screen_width, self.screen_height))
                    for _ in range(self.num_envs)
                ]
        if self.clocks is None:
            self.clock = [pygame.time.Clock() for _ in range(self.num_envs)]

        world_width = self.x_threshold * 2
        scale = self.screen_width / world_width
        polewidth = 10.0
        polelen = scale * (2 * self.length)
        cartwidth = 50.0
        cartheight = 30.0

        if self.state is None:
            return None

        for state, screen, clock in zip(self.state, self.screens, self.clocks):
            x = self.state.T

            self.surf = pygame.Surface((self.screen_width, self.screen_height))
            self.surf.fill((255, 255, 255))

            l, r, t, b = -cartwidth / 2, cartwidth / 2, cartheight / 2, -cartheight / 2
            axleoffset = cartheight / 4.0
            cartx = x[0] * scale + self.screen_width / 2.0  # MIDDLE OF CART
            carty = 100  # TOP OF CART
            cart_coords = [(l, b), (l, t), (r, t), (r, b)]
            cart_coords = [(c[0] + cartx, c[1] + carty) for c in cart_coords]
            gfxdraw.aapolygon(self.surf, cart_coords, (0, 0, 0))
            gfxdraw.filled_polygon(self.surf, cart_coords, (0, 0, 0))

            l, r, t, b = (
                -polewidth / 2,
                polewidth / 2,
                polelen - polewidth / 2,
                -polewidth / 2,
            )

            pole_coords = []
            for coord in [(l, b), (l, t), (r, t), (r, b)]:
                coord = pygame.math.Vector2(coord).rotate_rad(-x[2])
                coord = (coord[0] + cartx, coord[1] + carty + axleoffset)
                pole_coords.append(coord)
            gfxdraw.aapolygon(self.surf, pole_coords, (202, 152, 101))
            gfxdraw.filled_polygon(self.surf, pole_coords, (202, 152, 101))

            gfxdraw.aacircle(
                self.surf,
                int(cartx),
                int(carty + axleoffset),
                int(polewidth / 2),
                (129, 132, 203),
            )
            gfxdraw.filled_circle(
                self.surf,
                int(cartx),
                int(carty + axleoffset),
                int(polewidth / 2),
                (129, 132, 203),
            )

            gfxdraw.hline(self.surf, 0, self.screen_width, carty, (0, 0, 0))

            self.surf = pygame.transform.flip(self.surf, False, True)
            screen.blit(self.surf, (0, 0))

        if self.render_mode == "human":
            pygame.event.pump()
            [clock.tick(self.metadata["render_fps"]) for clock in self.clocks]
            pygame.display.flip()

        elif self.render_mode == "rgb_array":
            return [
                np.transpose(
                    np.array(pygame.surfarray.pixels3d(screen)), axes=(1, 0, 2)
                )
                for screen in self.screens
            ]

    def close(self):
        if self.screens is not None:
            import pygame

            pygame.display.quit()
            pygame.quit()
            self.isopen = False
