# import required library
import os
import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO

# stable baseline3 is allow vectorize environment in which agent learn on more than one environment at the same time
from stable_baselines3.common.vec_env import DummyVecEnv
# for test the model performance
from stable_baselines3.common.evaluation import evaluate_policy

import warnings
warnings.filterwarnings("ignore")


#############LOAD ENVIRONMENT#############

# specify total duration of balance task and time interval of integration
balance_time = 20; h_in = 1/100;

# register new gym env
CartPoleLab = gym.register(id='CartPoleLab',
                             entry_point='myCartpoleF:myCartPoleEnvF',
                             reward_threshold=balance_time / h_in * 0.95,
                             max_episode_steps=int( balance_time / h_in ) )

# make the gym env
env = gym.make('CartPoleLab')

# inspect the env specs to make sure it's working properly
gym.spec('CartPoleLab')

# check the env
episodes = 5
for episode in range(1, episodes + 1):
    state = env.reset()  # initial set of observation
    done = False
    score = 0 
    while not done:
        env.render()    # display the environment
        action = env.action_space.sample()
        step_result = env.step(action)
        
        # Unpack the first four elements, ignore the rest
        n_state, reward, done, info, _ = step_result
        score += reward
    
    print('Episode:{} Score:{}'.format(episode, score))
env.close()

## Environment Functions
   # env.reset(): reset the env and obtain initial observations
   # env.render(): visualise the environment
   # env.step(): apply an action to the environment
   # env.close(): close down the render frame

# initial set of observation
env.reset()

# Action space is '0' or '1'
env.action_space

# 0-push cart to left, 1-push cart to the right
env.action_space.sample()

env.observation_space

# [cart position[-4.8,4.8], cart velocity[-Inf,Inf], pole angle[-24,24], pole angular velocity[-Inf,Inf]]
env.observation_space.sample()

env.step(1)


#############ADDING A CALLBACK#############
from collections import deque
from stable_baselines3.common.callbacks import BaseCallback

class StableWindow(BaseCallback):
    """
    Monitor scalar signals and determine whether they are stable within an episode
    """
    def __init__(self, signal_fn, window_size=500, tol=0.5, name="signal", verbose=0):
        super().__init__(verbose)
        self.signal_fn  = signal_fn
        self.window_size = window_size
        self.tol        = tol
        self.name       = name
        self.buffer     = deque(maxlen=window_size)
        self._stable    = False

    def is_stable(self) -> bool:
        return self._stable

    def _on_step(self) -> bool:
        #  If an episode ends, clear the window immediately
        done_flag = False
        if 'dones' in self.locals:                      # Vectorized environment
            done_flag = bool(np.any(self.locals['dones']))
        elif 'done' in self.locals:                     # Non-Vectorized environment
            done_flag = bool(self.locals['done'])
        if done_flag:
            self.buffer.clear()         # Ensure that the window only stores the data of "the same episode"
            self._stable = False

        # put the latest signal in the window 
        val = float(self.signal_fn(self.locals))
        self.buffer.append(val)

        # if the window is not full, continue training
        if len(self.buffer) < self.window_size:
            return True

        # determine whether the amplitude within the window is within the tolerance range
        r_min, r_max = min(self.buffer), max(self.buffer)
        self._stable = (r_max - r_min) <= self.tol

        if self.verbose >= 2:
            print(f"[{self.name}] window min={r_min:.4f} max={r_max:.4f} "
                  f"Δ={r_max - r_min:.4f} (tol={self.tol})  stable={self._stable}")

        return True   # Only update the status, not directly suspend training
    
class AndCallback(BaseCallback):
    """
    Aggregate multiple StableWindows; Stop training early when all are in place
    """
    def __init__(self, callbacks, verbose=0):
        super().__init__(verbose)
        self.callbacks = callbacks

    # 1. Let the child callbacks also obtain handles such as model/logger/training_env
    def init_callback(self, model):
        super().init_callback(model)           
        for cb in self.callbacks:              
            cb.init_callback(model)

    # 2. Hooks at the beginning of training (optional, but it's more secure to keep the levels aligned
    def _on_training_start(self):
        for cb in self.callbacks:
            cb._on_training_start()

    # 3. Pass locals/globals to the child callback at each step and execute on_step
    def _on_step(self) -> bool:
        for cb in self.callbacks:
            cb.locals  = self.locals
            cb.globals = self.globals
            cb.on_step()                       

        all_stable = all(cb.is_stable() for cb in self.callbacks)
        if all_stable:
            if self.verbose:
                print("All signals are stable - stop training in advance")
            return False                       
        return True

    # 4. It is also recursively closed at the end of the training (optional)
    def _on_training_end(self):
        for cb in self.callbacks:
            cb._on_training_end()

from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
import os

save_path = os.path.join('Training', 'Saved Models')
log_path = os.path.join('Training', 'Logs')

env = gym.make('CartPoleLab')
env = DummyVecEnv([lambda: env])

from stable_baselines3.common.callbacks import CallbackList, EvalCallback

# 1️. Encapsulate 4  observations of CartPole into signal_fn
def cartpole_signal(idx):
    """
    idx: 0 = x, 1 = x_dot, 2 = theta, 3 = theta_dot
    Take the latest observations directly from locals_['new_obs']
    """
    return lambda locals_: float(locals_["new_obs"][0][idx])

# 2️. Configure each StableWindow (continuous WINDOW step amplitude ≤ tol)
WINDOW = 400      
cb_x       = StableWindow(cartpole_signal(0), WINDOW, 0.2,  name="x",          verbose=1)
cb_xdot    = StableWindow(cartpole_signal(1), WINDOW, 0.2,  name="x_dot")
cb_theta   = StableWindow(cartpole_signal(2), WINDOW, 0.2, name="theta")
cb_thetad  = StableWindow(cartpole_signal(3), WINDOW, 0.2, name="theta_dot")

# 3️. As long as 4  Windows are stable ⇒ early suspension of training
stop_all = AndCallback([cb_x, cb_xdot, cb_theta, cb_thetad], verbose=1)

eval_cb   = EvalCallback(
    env,
    best_model_save_path="models",
    eval_freq=10000,
    n_eval_episodes=5,
    verbose=0,
)

# 4. Summarize all callbacks
callbacks = CallbackList([eval_cb, stop_all])

# 5.Create a training environment & model
train_env = gym.make('CartPoleLab')
train_env = DummyVecEnv([lambda: train_env])
model = PPO('MlpPolicy',
            env, 
            n_steps=4096,
            batch_size=256,
            n_epochs=10,
            clip_range=0.15,
            learning_rate=lambda f: 3e-3 * f,
            verbose = 1,
            tensorboard_log=log_path)

# 6. Start training - It will be automatically interrupted if the early stop condition is met
model.learn(total_timesteps=1000000, callback=callbacks)

# 7. Clear
train_env.close()
env.close()

model_path = os.path.join('Training', 'Saved Models', 'best_model')
model.save(model_path)

del model
model = PPO.load(model_path, env=env)

evaluate_policy(model, env, n_eval_episodes=10, render=True)

env.close()