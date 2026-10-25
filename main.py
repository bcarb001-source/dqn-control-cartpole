import gymnasium as gym
import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import keyboard
import numpy as np
from gymnasium import spaces
from pathlib import Path
from agent import Agent
import pygame

# Keep saved weights and plots next to this script, wherever it's run from
SCRIPT_DIR = Path(__file__).resolve().parent
path = SCRIPT_DIR / 'Cartpole-State' / 'cartpole.pth'

class ControlledCart(gym.Wrapper):
    ARROW_COLOR = (40, 110, 200)
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)
        self.target_vel = 0
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(5,), dtype=self.observation_space.dtype)
        
    def reset(self, seed=None, options=None):
        self.target_vel = 0
        state, info = self.env.reset(seed=seed, options=options)
        state = np.append(state, self.target_vel)
        return state, info
    def step(self, action, tv):
        self.target_vel = tv
        ob, reward, terminated, truncated, info = self.env.step(action)

        u = self.env.unwrapped
        if terminated and abs(ob[2]) <= u.theta_threshold_radians:
            # Terminated only because the cart left the track (pole is fine):
            # wrap the cart to the other side and keep going
            x = (ob[0] + u.x_threshold) % (2 * u.x_threshold) - u.x_threshold
            u.state = np.array((x, ob[1], ob[2], ob[3]), dtype=np.float64)
            u.steps_beyond_terminated = None   # stops the env treating the episode as over
            ob[0] = x
            terminated = False
        self._draw_target_arrow()
        ob = np.append(ob, self.target_vel)
        return ob, reward, terminated, truncated, info
    
    def _draw_target_arrow(self):
        u = self.env.unwrapped
        if u.render_mode != "human" or u.screen is None or self.target_vel == 0:
            return
        d = 1 if self.target_vel > 0 else -1
        cx, y = u.screen_width // 2, 40
        tip = cx + d * 45
        pygame.draw.line(u.screen, self.ARROW_COLOR, (cx - d * 45, y), (tip - d * 18, y), 8)
        pygame.draw.polygon(u.screen, self.ARROW_COLOR,
                            [(tip, y), (tip - d * 22, y - 16), (tip - d * 22, y + 16)])
        pygame.display.flip()    


env = gym.make("CartPole-v1", max_episode_steps=2000)
env = ControlledCart(env=env)


# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

# if GPU is to be used
device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"

)

# Uncomment to use seeds, ensures reproducibility during training.

#seed = 42
#random.seed(seed)
#torch.manual_seed(seed)
#env.reset(seed=seed)
#env.action_space.seed(seed)
#env.observation_space.seed(seed)
#if torch.cuda.is_available():
    #torch.cuda.manual_seed(seed)


def answer(ans):
    if ans in ('Y', 'y'):
        return True
    elif ans in ('N','n'):
        return False
    else:
        return 'invalid answer.'        

# BATCH_SIZE is the number of transitions sampled from the replay buffer
# GAMMA is the discount factor as mentioned in the previous section
# EPS_START is the starting value of epsilon
# EPS_END is the final value of epsilon
# EPS_DECAY controls the rate of exponential decay of epsilon, higher means a slower decay
# TAU is the update rate of the target network
# LR is the learning rate of the ``AdamW`` optimizer

EPS_START = 0.9
EPS_END = 0.01
TAU = 0.005
EPS_DECAY = 25000


# Get the number of actions from gym action space
n_actions = env.action_space.n
# Get the number of state observations
state, info = env.reset()
n_observations = len(state)

agent = Agent(n_observations, n_actions, device)

# Check if there's saved weights available, and if there is, prompt the user.
load_weights = False
ans = input('Load saved weights? (y/n):')
load_weights = answer(ans)
do_train = True
if not type(load_weights) is bool:
    print(load_weights, ' continuing without loading weights')
elif load_weights:
    if not agent.load_weights(path):
        print('No file was found at ', path)
        print('Starting fresh...')
    else:
        print('save loaded successfully')
        ans = input('Train? (y/n):')
        do_train = answer(ans)
        if not type(do_train) is bool:
            print(do_train, ' training is disabled. skipping to inference')
            do_train = False
        else:
            print('training...')
else:
    print('Starting fresh weights.')




def select_action(state, _agent):
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * _agent.Steps_done() / EPS_DECAY)
    _agent.step()
    if sample > eps_threshold:
        return _agent.greedy_action(state)
    else:
        return torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)

  
episode_durations = []
episode_cumulative_reward = []
def rolling_mean(values, window=100):
    # Mean of the previous `window` episodes, starting once that many exist
    if len(values) < window:
        return [], []
    means = np.convolve(np.asarray(values, dtype=float), np.ones(window) / window, mode='valid')
    return np.arange(window, len(values) + 1), means

def plot_durations(show_result=False):
    fig = plt.figure(1, figsize=(8, 6))
    fig.clf()
    ax_len, ax_rew = fig.subplots(2, 1, sharex=True)
    fig.suptitle('Result' if show_result else 'Training...')

    episodes = np.arange(1, len(episode_durations) + 1)
    # Cumulative reward grows with episode length, so divide it out to isolate tracking quality
    reward_per_step = np.asarray(episode_cumulative_reward) / np.maximum(episode_durations, 1)

    # Top: how long the agent kept the pole up
    ax_len.plot(episodes, episode_durations, alpha=0.35, label='Episode')
    ax_len.plot(*rolling_mean(episode_durations), label='100-episode mean')
    ax_len.set_ylabel('Episode length (steps)')
    ax_len.set_ylim(bottom=0)
    ax_len.legend(loc='upper left')

    # Bottom: how closely it followed the velocity command (1 = perfect)
    ax_rew.plot(episodes, reward_per_step, alpha=0.35, label='Episode')
    ax_rew.plot(*rolling_mean(reward_per_step), label='100-episode mean')
    ax_rew.set_ylabel('Tracking reward per step')
    ax_rew.set_ylim(0, 1.05)
    ax_rew.set_xlabel('Episode')
    ax_rew.legend(loc='lower right')

    fig.tight_layout()
    plt.pause(0.001) # pause a bit so that plots are updated
    if is_ipython:
        if not show_result:
            display.display(fig)
            display.clear_output(wait=True)
        else:
            display.display(fig)
    return fig




def train(_agent, _env, n_episodes):
    episode_durations.clear()

    episode_cumulative_reward.clear()
    for i_episode in range(n_episodes):
        # Initialize the environment and get its state
        state, info = _env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        action_trigger = 1
        target_vel = 0
        endtrain = False
        cumulative_reward = 0
        ep_loss = []
        for t in count():
            #state[0, 4] = target_vel
            action = select_action(state, _agent)
            observation, reward, terminated, truncated, _ = _env.step(action.item(), target_vel)
            if keyboard.is_pressed('esc'):
                endtrain = True
            #
            # observation index 1 is Cart velocity. use this to calculate the target velocity error
            #
            actual_vel = observation[1]


            vel_err = abs(target_vel - actual_vel)
            reward = 1.0 - 0.5 * min(vel_err, 2.0)
            # for tracking total episode rewards
            cumulative_reward += reward

            reward = torch.tensor([reward], device=device)
            done = terminated or truncated
             # update the target action when triggered
            if t > action_trigger:
                target_vel = random.randint(-1, 1)
                action_trigger += random.randint(50, 300)
            if terminated:
                next_state = None
                reward -= 10.0
            else:
                next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
                next_state[0, 4] = target_vel
            # Store the transition in memory
            _agent.store_transition(state, action, next_state, reward)

            # Move to the next state
            state = next_state

            # Perform one step of the optimization (on the policy network)
            loss = _agent.optimize_model()
            if loss is not None:
                ep_loss.append(loss)

            _agent.soft_update(TAU)

            if done:
                episode_durations.append(t+1)
                episode_cumulative_reward.append(cumulative_reward)
                plot_durations()
                break
        if endtrain:
            break
    print('Complete')
    fig = plot_durations(show_result=True)
    fig.savefig(SCRIPT_DIR / 'training_curves.png', dpi=150)
    plt.ioff()
    plt.show()
    return _agent

# Fixed command sequence for evaluate(): (target velocity, number of steps)
EVAL_SCHEDULE = [(0, 150), (1, 300), (-1, 300), (0, 250)]
# The cart's velocity jitters by about ±0.2 m/s from step to step, so settling is judged on a
# 0.2 s moving average: settled once that average stays within SETTLE_BAND m/s of the target
SMOOTH_STEPS = 10
SETTLE_BAND = 0.2

def evaluate(_agent, _env, schedule=EVAL_SCHEDULE, seed=0):
    # Run the greedy policy through a scripted command sequence, print tracking
    # error, and save a target vs. actual velocity plot
    u = _env.unwrapped
    targets = np.concatenate([np.full(steps, tv, dtype=float) for tv, steps in schedule])
    observation, info = _env.reset(seed=seed)
    velocities, angles = [], []
    for target in targets:
        observation[4] = target
        state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
        action = _agent.greedy_action(state)
        observation, _, terminated, truncated, _ = _env.step(action.item(), target)
        velocities.append(observation[1])
        angles.append(np.degrees(observation[2]))
        if terminated or truncated:
            break

    n = len(velocities)
    completed = n == len(targets)
    targets = targets[:n]
    velocities = np.array(velocities)
    angles = np.array(angles)
    t = np.arange(n) * u.tau  # u.tau is the simulation time step (0.02 s)
    err = np.abs(targets - velocities)
    # Trailing moving average (shorter window for the first few steps)
    idx = np.arange(1, n + 1)
    lo = np.maximum(idx - SMOOTH_STEPS, 0)
    csum = np.concatenate(([0.0], np.cumsum(velocities)))
    smoothed = (csum[idx] - csum[lo]) / (idx - lo)
    smoothed_err = np.abs(targets - smoothed)

    print('Evaluation:', 'completed' if completed else f'pole fell at {t[-1]:.1f} s')
    print(f'  Mean |velocity error|: {err.mean():.3f} m/s')
    print(f'  Max |pole angle|: {np.abs(angles).max():.1f} deg')
    start = 0
    for tv, steps in schedule:
        seg = err[start:start + steps]
        if len(seg) == 0:
            break
        outside = np.nonzero(smoothed_err[start:start + steps] > SETTLE_BAND)[0]
        settle = outside[-1] + 1 if len(outside) else 0
        if settle >= len(seg):
            print(f'  Target {tv:+d} m/s: never settled within {SETTLE_BAND} m/s')
        else:
            print(f'  Target {tv:+d} m/s: settled in {settle * u.tau:.2f} s, '
                  f'then mean error {seg[settle:].mean():.3f} m/s')
        start += steps

    fig = plt.figure(2, figsize=(8, 5.5))
    fig.clf()
    ax_vel, ax_ang = fig.subplots(2, 1, sharex=True)
    fig.suptitle('Scripted evaluation')

    ax_vel.step(t, targets, where='post', color='black', linewidth=1.2, label='Target')
    ax_vel.plot(t, velocities, color='tab:blue', linewidth=0.8, alpha=0.35, label='Actual')
    ax_vel.plot(t, smoothed, color='tab:blue', linewidth=1.5, label='Actual (0.2 s average)')
    ax_vel.set_ylabel('Cart velocity (m/s)')
    ax_vel.legend(loc='upper right')

    limit = np.degrees(u.theta_threshold_radians)
    ax_ang.plot(t, angles, linewidth=1)
    ax_ang.axhline(limit, color='gray', linestyle='--', linewidth=0.8, label='Fall limit')
    ax_ang.axhline(-limit, color='gray', linestyle='--', linewidth=0.8)
    ax_ang.set_ylabel('Pole angle (deg)')
    ax_ang.set_xlabel('Time (s)')
    ax_ang.legend(loc='upper right')

    fig.tight_layout()
    fig.savefig(SCRIPT_DIR / 'velocity_tracking.png', dpi=150)
    return fig

def run(_agent, _env, n_episodes):
    for i_episode in range(n_episodes):
        # Initialize the environment and get its state
        state, info = _env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        target_vel = 0

        for t in count():
            if keyboard.is_pressed('right arrow'):
                target_vel = 1
            elif keyboard.is_pressed('left arrow'):
                target_vel = -1
            else:
                target_vel = 0

            action = _agent.greedy_action(state)
            observation, reward, terminated, truncated, _ = _env.step(action.item(), target_vel)
            
            #
            # observation index 1 is Cart velocity. use this to calculate the target velocity error
            #
            actual_vel = observation[1]          
            control_reward = -abs(target_vel-actual_vel)
            reward+=control_reward
            reward = torch.tensor([reward], device=device)
            done = terminated or truncated
            print('Target Vel: ', target_vel, 'Actual Vel: ', actual_vel, 'Reward :', control_reward)

            if terminated:
                next_state = None
            else:
                next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
                next_state[0, 4] = target_vel

            state = next_state
            if done:
                break


if do_train:
    agent = train(agent, env, 800)
    path.parent.mkdir(parents=True, exist_ok=True)
    agent.save(path)

# Scripted test run on the headless env: prints tracking error, saves velocity_tracking.png
evaluate(agent, env)
print('Close the plot window to start the demo.')
plt.ioff()
plt.show()

# set the render mode to 'human' for the real time demo
env = gym.make("CartPole-v1", render_mode='human', max_episode_steps=2000)
env = ControlledCart(env=env)
run(agent, env, 100)




