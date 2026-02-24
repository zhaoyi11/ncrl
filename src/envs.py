from collections import OrderedDict, deque, namedtuple
from typing import Any, NamedTuple
import os
import abc
import re

import numpy as np
import gymnasium as gym

MAX_EPISODE_LENGTH_DMC = 1000
MAX_EPISODE_LENGTH_MW = 200

# 50 meta world tasks in total, 1 embodiment
_MetaWorld_TASK_SET = (
    'assembly', 'basketball', 'bin-picking', 'box-close', 'button-press',
    'button-press-topdown', 'button-press-topdown-wall', 'button-press-wall', 'coffee-button', 'coffee-pull',
    'coffee-push', 'dial-turn', 'disassemble', 'door-close', 'door-lock',
    'door-open', 'door-unlock', 'drawer-close', 'drawer-open', 'faucet-open',
    'faucet-close', 'hammer', 'hand-insert', 'handle-press-side', 'handle-press',
    'handle-pull-side', 'handle-pull', 'lever-pull', 'peg-insert-side', 'peg-unplug-side',
    'pick-out-of-hole', 'pick-place-wall', 'pick-place', 'plate-slide-back-side', 'plate-slide-back',
    'plate-slide-side', 'plate-slide', 'push-back', 'push-wall', 'push',
    'reach', 'reach-wall', 'shelf-place', 'soccer', 'stick-push',
    'stick-pull', 'sweep-into', 'sweep', 'window-close', 'window-open',
)
MetaWorld_TASK_SET = tuple([f'mw-{name}' for name in _MetaWorld_TASK_SET])

# 22 dm control tasks in total, 5 embodiments
_DMControl_TASK_SET = (
    'cartpole-balance', # 1
    'acrobot-swingup', 'acrobot-swingup-sparse', 'acrobot-swingup-hard', # 3
    'cheetah-run', 'cheetah-run-backwards', 'cheetah-jump', 'cheetah-run-back', 'cheetah-run-front', # 5
    'walker-stand', 'walker-walk', 'walker-walk-hard', 'walker-run', 'walker-walk-backwards', 'walker-run-backwards', 'walker-backflip', # 7
    'quadruped-stand', 'quadruped-walk', 'quadruped-run', 'quadruped-jump', 'quadruped-roll', 'quadruped-roll-fast', # 6
)
DMControl_TASK_SET = tuple([f'dmc-{name}' for name in _DMControl_TASK_SET])

TASK_SET = (*DMControl_TASK_SET, *MetaWorld_TASK_SET)
TASK_DICT = {k: i for i, k in enumerate(TASK_SET)}

# action dim
_DOMAIN_ACT_DIM = {'dmc-cartpole': 1, 'dmc-acrobot': 1, 'dmc-cheetah': 6, 'dmc-walker': 6, 'dmc-quadruped': 12, 'mw-': 4}
# get action dim for each task
TASK_ACT_DIM = {task: _DOMAIN_ACT_DIM.get(re.match(r'^([\w]+-[\w]+)', task).group(), 4) for task in TASK_SET} # the default action dim is 4 (for metaworld tasks)
MAX_ACTION_DIM = 21 # TODO: check if this is 12 or 21??

observation_space = namedtuple('observation_space', ['image', 'state'])
specs = namedtuple('specs', ['shape', 'dtype', 'name'])
Timestep = namedtuple('Timestep', ['observation', 'state', 'reward', 'action',
                                    'done', 'discount', 'is_first', 'is_last', 'info']) # done: whether an episode ends, is_last: whether an episode is truncated, discount: whether an episode is terminated (1. if not terminated, 0. if terminated)

class Env(abc.ABC):
    @abc.abstractmethod
    def reset():
        pass

    @abc.abstractmethod
    def step():
        pass

    @abc.abstractmethod
    def close():
        pass

    @property
    @abc.abstractmethod
    def obs_space():
        pass

    @property
    @abc.abstractmethod
    def action_space():
        pass

    @property
    @abc.abstractmethod
    def episode_length():
        pass


class MetaWorld(Env):
    def __init__(self, task_name, seed, obs_type='pixels',
                    width=64, height=64,
                    camera_name='corner2', render_mode='rgb_array'):
        import metaworld, copy
        from metaworld.envs import ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE

        env_id = task_name.split('-', 1)[-1]
        task = f'{env_id}-v2-goal-observable'
        if not task_name.startswith('mw-') or task not in ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE:
            raise ValueError('Unknown task:', task_name)

        env_cls = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE[task]
        self._env = env_cls(seed=seed, width=width, height=height, camera_name=camera_name, render_mode=render_mode)
        self._env._freeze_rand_vec = False

        self._obs_type = obs_type
        self._camera_name = camera_name

    def reset(self):
        if self._camera_name == 'corner2':
            # from https://github.com/XuGW-Kevin/DrM/blob/main/metaworld_env.py
            self._env.model.cam_pos[2] = [0.75, 0.075, 0.7] # bring the camera closer to the object

        state, info = self._env.reset()
        state = state.astype(np.float32)
        # reset viewer https://github.com/Farama-Foundation/Gymnasium/issues/736
        if self._env.mujoco_renderer.viewer is not None:
            self._env.mujoco_renderer.viewer.make_context_current()

        if self._obs_type == 'pixels':
            image = self._env.render()[::-1].copy() # flip the image, due to the difference in coordinate system of opengl and opencv
        else:
            image = None
        return Timestep(observation=image, state=state, reward=0.0,
                        action=np.zeros_like(self._env.action_space.sample()),
                         done=False, is_first=True, is_last=False, discount=1.0, info=info)

    def step(self, action):
        assert np.isfinite(action).all(), action
        state, reward, terminated, truncated, info = self._env.step(action)
        state = state.astype(np.float32)
        done = terminated or truncated
        if self._obs_type == 'pixels':
            image = self._env.render()[::-1].copy()
        else:
            image = None

        return Timestep(observation=image, state=state, reward=reward, action=action,
                        done=done, is_first=False, is_last=truncated, discount=float(not terminated), info=info)

    def close(self):
        self._env.close()

    @property
    def obs_space(self):
        return observation_space(image=specs(shape=(self._env.height, self._env.width, 3), dtype=np.uint8, name='image'),
                                state=specs(shape=(self._env.observation_space.shape[0],), dtype=np.float32, name='state'))

    @property
    def action_space(self):
        return specs(shape=(self._env.action_space.shape[0],), dtype=np.float32, name='action')

    @property
    def episode_length(self): 
        return self._env.episode_length


class DMControl(Env):
    def __init__(self, task_name, seed, obs_type='pixels',
                    width=64, height=64,
                ):
        from dm_control import suite
        from dm_control.suite.wrappers import pixels
        import custom_dmc_tasks as cdmc


        domain, task = task_name.split('-', 1)[-1].split('-', 1)
        domain = dict(cup='ball_in_cup', point='point_mass').get(domain, domain)
        task = task.replace('-', '_')

        env = suite.load(
            domain, task,
            task_kwargs=dict(random=seed),
            environment_kwargs=dict(flat_observation=True),
            visualize_reward=False,
        )

        # pixel observation
        if obs_type == 'pixels':
            camera_id = dict(quadruped=2).get(domain, 0)
            render_kwargs = dict(width=width, height=height, camera_id=camera_id)
            env = pixels.Wrapper(env,
                                pixels_only=False,
                                render_kwargs=render_kwargs)

        self._env = env
        self._obs_type = obs_type
        env._width = width
        env._height = height

    def reset(self):
        timestep = self._env.reset()
        if self._obs_type == 'pixels':
            image = timestep.observation['pixels']
        else:
            image = None
        return Timestep(observation=image, state=timestep.observation['observations'], reward=0.,
                        action=np.zeros(self._env.action_spec().shape, dtype=np.float32),
                        done=False, is_first=timestep.first(), is_last=timestep.last(), discount=1.0, info=None)

    def step(self, action):
        timestep = self._env.step(action)

        if self._obs_type == 'pixels':
            image = timestep.observation['pixels']
        else:
            image = None
        return Timestep(observation=image, state=timestep.observation['observations'], reward=timestep.reward, action=action,
                        done=timestep.last(), is_first=timestep.first(), is_last=timestep.last(), discount=timestep.discount, info=None)

    def close(self):
        self._env.close()

    @property
    def obs_space(self):
        return observation_space(image=specs(shape=self._env.observation_spec()['pixels'].shape, dtype=np.uint8, name='image'),
                                state=specs(shape=self._env.observation_spec()['observations'].shape, dtype=np.float32, name='state'))

    @property
    def action_space(self):
        return specs(shape=(self._env.action_spec().shape[0],), dtype=np.float32, name='action')

    @property
    def episode_length(self):
        return self._env._max_episode_steps


# Env Wrappers
class TimeLimitWrapper(Env):
    """ Terminate an episode after a fixed number of steps. """
    def __init__(self, env, max_episode_steps):
        self._env = env
        self._max_episode_steps = max_episode_steps
        self.curr_steps = 0

    def reset(self):
        self.curr_steps = 0
        return self._env.reset()

    def step(self, action):
        timestep = self._env.step(action)
        self.curr_steps += 1
        truncated = (self.curr_steps >= self._max_episode_steps)
        terminated = (timestep.discount == 0.0)
        done = truncated or terminated
        return timestep._replace(done=done, is_last=truncated, discount=timestep.discount)

    def close(self):
        self._env.close()

    @property
    def obs_space(self):
        return self._env.obs_space

    @property
    def action_space(self):
        return self._env.action_space

    @property
    def episode_length(self):
        return self._max_episode_steps


class ActionRepeatWrapper(Env):
    """ Repeat the same action for a fixed number of times. """
    def __init__(self, env, repeat):
        self._env = env
        self._repeat = repeat
        assert hasattr(self._env, 'curr_steps'), 'The environment must have max_episode_steps attribute (Try to call TimeLimitWrapper first).'

    def reset(self):
        return self._env.reset()

    def step(self, action):
        reward = 0.
        done = False
        for _ in range(self._repeat):
            timestep = self._env.step(action)
            reward += (timestep.reward or 0.0) * float(not done)
            done = done or timestep.done
            if done:
                break
        return timestep._replace(reward=reward, done=done)

    def close(self):
        self._env.close()

    @property
    def obs_space(self):
        return self._env.obs_space

    @property
    def action_space(self):
        return self._env.action_space

    @property
    def episode_length(self):
        return self._env.episode_length


class FrameStackWrapper(Env):
    """ Stack consecutive frames as a single observation. """
    def __init__(self, env, num_stack=1):
        self._env = env
        self._num_stack = num_stack
        self._frames = deque(maxlen=num_stack)

    def reset(self):
        timestep = self._env.reset()
        if timestep.observation is not None:
            for _ in range(self._num_stack):
                self._frames.append(timestep.observation)
            return timestep._replace(observation=np.concatenate(list(self._frames), axis=-1))
        return timestep

    def step(self, action):
        timestep = self._env.step(action)
        if timestep.observation is not None:
            self._frames.append(timestep.observation)
            return timestep._replace(observation=np.concatenate(list(self._frames), axis=-1))
        return timestep

    def close(self):
        self._env.close()

    @property
    def obs_space(self):
        image_shape = self._env.obs_space.image.shape
        return observation_space(image=specs(shape=(image_shape[0], image_shape[1], image_shape[2] * self._num_stack), dtype=np.uint8, name='image'),
                                state=self._env.obs_space.state)

    @property
    def action_space(self):
        return self._env.action_space

    @property
    def episode_length(self):
        return self._env.episode_length


class HardTaskWrapper(Env):
    """ Penalize actions and use sparse rewards for hard tasks. """
    def __init__(self, env, reward_threshold=0.0, action_penalty=0.0):
        self._env = env
        self._reward_threshold = reward_threshold
        self._action_penalty = action_penalty

    def reset(self):
        return self._env.reset()

    def step(self, action):
        timestep = self._env.step(action)
        reward = timestep.reward
        if reward < self._reward_threshold:
            reward = 0.
        reward = reward - self._action_penalty * np.linalg.norm(action)
        return timestep._replace(reward=reward)

    def close(self):
        self._env.close()

    @property
    def obs_space(self):
        return self._env.obs_space

    @property
    def action_space(self):
        return self._env.action_space

    @property
    def episode_length(self):
        return self._env.episode_length


class FormatOutputWrapper(Env):
    """ Covert output to float32, and permute image to CHW. """
    def __init__(self, env):
        self._env = env

    def _format_obs(self, timestep):
        if timestep.observation is not None:
            image = np.transpose(timestep.observation, (2, 0, 1)) # uint8, HWC -> CHW
        else:
            image = None
        if timestep.state is not None:
            state = timestep.state.astype(np.float32)
        else:
            state = None
        return timestep._replace(observation=image, state=state)

    def reset(self):
        timestep = self._env.reset()
        return self._format_obs(timestep)

    def step(self, action):
        timestep = self._env.step(action)
        return self._format_obs(timestep)

    def close(self):
        self._env.close()

    @property
    def obs_space(self):
        obs_space = self._env.obs_space
        # permute image to CHW
        return obs_space._replace(image=specs(shape=(obs_space.image.shape[2], obs_space.image.shape[0], obs_space.image.shape[1]),
                                 dtype=np.uint8, name='image'))

    @property
    def action_space(self):
        return self._env.action_space

    @property
    def episode_length(self):
        return self._env.episode_length

def make(task_name, seed, obs_type="pixels", action_repeat=1, frame_stack=1,  width=64, height=64, reward_threshold=0.0, action_penalty=0.0):
    """ Make a single environment. """
    _hard_task = False
    if task_name.endswith('-hard'):
        _task_name = task_name[:-5]
        _hard_task = True
    else:
        _task_name = task_name
        _hard_task = False

    if task_name.startswith('mw-'):
        env = MetaWorld(_task_name, seed, obs_type=obs_type, width=width, height=height)
        max_episode_frames = MAX_EPISODE_LENGTH_MW
    elif task_name.startswith('dmc-'):
        env = DMControl(_task_name, seed, obs_type=obs_type, width=width, height=height)
        max_episode_frames = MAX_EPISODE_LENGTH_DMC
    else:
        raise ValueError('Unknown task:', task_name)

    # wrap the environment
    if _hard_task:
        env = HardTaskWrapper(env, reward_threshold=reward_threshold, action_penalty=action_penalty)

    env = TimeLimitWrapper(env, max_episode_frames)
    env = ActionRepeatWrapper(env, action_repeat)
    env = FrameStackWrapper(env, frame_stack)
    env = FormatOutputWrapper(env)

    return env


def make_env(cfg, seed):
    """ Make an environment. """
    assert cfg.obs_type in ('pixels', 'state')

    env = make(cfg.task, seed, obs_type=cfg.obs_type, action_repeat=cfg.action_repeat, frame_stack=cfg.frame_stack,
                    width=cfg.img_size, height=cfg.img_size, reward_threshold=cfg.reward_threshold, action_penalty=cfg.action_penalty)
    return env

