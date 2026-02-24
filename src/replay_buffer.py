# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import datetime
import io
import random
import uuid
import traceback
from collections import defaultdict
import pathlib
from typing import NamedTuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import IterableDataset


def episode_len(episode):
    # subtract -1 because the dummy first transition
    return next(iter(episode.values())).shape[0] - 1

def save_episode(episode, fn):
    with io.BytesIO() as bs:
        np.savez_compressed(bs, **episode)
        bs.seek(0)
        with fn.open('wb') as f:
            f.write(bs.read())

def load_episode(fn):
    with fn.open('rb') as f:
        episode = np.load(f)
        episode = {k: episode[k] for k in episode.keys()}
        return episode

def convert(value):
  value = np.array(value)
  if np.issubdtype(value.dtype, np.floating):
    return value.astype(np.float32)
  elif np.issubdtype(value.dtype, np.signedinteger):
    return value.astype(np.int32)
  elif np.issubdtype(value.dtype, np.uint8):
    return value.astype(np.uint8)
  return value


class ReplayBufferStorage:
    """ Replay buffer storage, where the episodes are stored in the replay directory as npz files.
        This buffer storage is modified to support multiple tasks. It saves the episodes of each task in replay_dir.
    """
    def __init__(self, replay_dir, data_specs):
        # create the base folder for storing the episodes
        self._replay_dir = pathlib.Path(replay_dir).expanduser()
        self._replay_dir.mkdir(parents=True, exist_ok=True)

        self._data_specs = data_specs
        self._current_episode = defaultdict(list)
        self._preload()

    def __len__(self):
        return self._num_transitions

    def add(self, time_step):
        """ Add one transition to the replay buffer. Save the episode when the episode is done."""
        time_step = time_step._asdict()

        for spec in self._data_specs:
            value = time_step[spec.name]
            if np.isscalar(value):
                assert spec.shape is not None
                value = np.full(spec.shape, value, spec.dtype)
            if isinstance(value, bool):
                value = np.array(value, dtype=np.float32)
            assert spec.dtype == value.dtype, f"Data type mismatch, expected {spec.dtype}, got {value.dtype}."
            if spec.shape is not None: 
                assert spec.shape == value.shape, f"Shape mismatch, expected {spec.shape}, got {value.shape}."

            self._current_episode[spec.name].append(value)
        
        if time_step['done']:
            episode = dict()
            for spec in self._data_specs:
                value = self._current_episode[spec.name]
                episode[spec.name] = np.array(value, spec.dtype)

            self._current_episode = defaultdict(list) # reset the current episode
            self._store_episode(episode)

    def copy_episode(self, episode):
        """ Copy an episode to the replay buffer. """
        assert isinstance(episode, dict), "Episode should be a dictionary."
        self._store_episode(episode)

    def _preload(self):
        self._num_episodes = 0
        self._num_transitions = 0
        for fn in self._replay_dir.rglob('*.npz'):
            _, _, _, eps_len = fn.stem.split('-')
            self._num_episodes += 1
            self._num_transitions += int(eps_len)

    def _store_episode(self, episode):
        # save the episode to the replay directory
        _path = self._replay_dir
        _path.mkdir(parents=True, exist_ok=True)
        
        eps_idx = self._num_episodes
        eps_len = episode_len(episode)
        self._num_episodes += 1
        self._num_transitions += eps_len
        ts = datetime.datetime.now().strftime('%Y%m%dT%H%M%S')
        identifier = str(uuid.uuid4().hex)
        eps_fn = f'{ts}-{identifier}-{eps_idx}-{eps_len}.npz'
        save_episode(episode, _path / eps_fn)


class ReplayBuffer(IterableDataset):
    """ Replay buffer iterable dataset, where the episodes are stored in the replay directory as npz files.
        It keeps fetching episodes from the replay directory and when the buffer is full, it removes the oldest episodes.
    """
    def __init__(self, replay_dir, data_specs, max_size, num_workers, nstep, pad_action,
                 fetch_every, save_snapshot, max_act_dim):
        self._replay_dir = replay_dir
        self._data_name = [spec.name for spec in data_specs]

        self._size = 0
        self._max_size = max_size
        self._num_workers = max(1, num_workers)
        self._episode_fns = []
        self._episodes = dict() # episode file path -> episode data
        self._max_act_dim = max_act_dim # maximum action dimension among all tasks
        self._pad_action = pad_action # if True, pad action with zeros to max_act_dim
        self._nstep = nstep # chunk length of sampled data
        self._fetch_every = fetch_every
        self._samples_since_last_fetch = fetch_every
        self._save_snapshot = save_snapshot

    def _sample_episode(self):
        eps_fn = random.choice(self._episode_fns)
        return eps_fn, self._episodes[eps_fn]

    def _store_episode(self, eps_fn):
        try:
            episode = load_episode(eps_fn)
        except:
            return False
        eps_len = episode_len(episode)
        while eps_len + self._size > self._max_size:
            early_eps_fn = self._episode_fns.pop(0)
            early_eps = self._episodes.pop(early_eps_fn)
            self._size -= episode_len(early_eps)
            early_eps_fn.unlink(missing_ok=True)
        self._episode_fns.append(eps_fn)
        self._episode_fns.sort()
        self._episodes[eps_fn] = episode
        self._size += eps_len

        if not self._save_snapshot:
            eps_fn.unlink(missing_ok=True)
        return True

    def _try_fetch(self):
        if self._samples_since_last_fetch < self._fetch_every:
            return
        self._samples_since_last_fetch = 0
        try:
            worker_id = torch.utils.data.get_worker_info().id
        except:
            worker_id = 0
        eps_fns = sorted(self._replay_dir.rglob('*.npz'), reverse=True, key=lambda x: int(x.stem.split('-')[-2])) # sort according to episode index
        fetched_size = 0
        for eps_fn in eps_fns:
            eps_idx, eps_len = [int(x) for x in eps_fn.stem.split('-')[2:]]

            if eps_idx % self._num_workers != worker_id:
                continue
            if eps_fn in self._episodes.keys():
                break
            if fetched_size + int(eps_len) > self._max_size:
                break
            fetched_size += int(eps_len)
            if not self._store_episode(eps_fn):
                break

    def _sample(self):
        try:
            self._try_fetch() # fetch new episodes if needed
        except:
            traceback.print_exc()
        self._samples_since_last_fetch += 1
        
        ep_name, episode = self._sample_episode()

        length = len(episode['action'])
        assert length >= self._nstep, f'Episode length {length} is too short for sampling chunk of length {self._nstep}.'
        upper = length - self._nstep + 1

        idx = np.random.randint(0, upper)
        seq = {k: np.copy(convert(v[idx:idx+self._nstep])) for k, v in episode.items() if k in self._data_name}
        # padding action with zeros: (L, act_dim) -> (L, max_act_dim)
        if self._pad_action:
            seq['action_mask'] = np.concatenate([np.ones_like(seq['action']), 
                    np.zeros((self._nstep, self._max_act_dim - seq['action'].shape[-1]), dtype=seq['action'].dtype)], axis=1)
            seq['action'] = np.concatenate([seq['action'], 
                    np.zeros((self._nstep, self._max_act_dim - seq['action'].shape[-1]), dtype=seq['action'].dtype)], axis=1)
        else:
            seq['action_mask'] = np.ones_like(seq['action'])
        seq['is_first'] = np.zeros(len(seq['action']), bool)
        seq['is_first'][0] = True
        seq['is_terminal'] = (seq['discount'] == 0)
        return seq
        
    def __iter__(self):
        while True:
            yield self._sample()


def _worker_init_fn(worker_id):
    seed = np.random.get_state()[1][0] + worker_id
    np.random.seed(int(seed))
    random.seed(int(seed))


def make_replay_loader(replay_dir, data_specs, max_size, batch_size, num_workers,
                       nstep, max_act_dim, pad_action, save_buffer=True):
    max_size_per_worker = max_size // max(1, num_workers)

    iterable = ReplayBuffer(replay_dir,
                            data_specs,
                            max_size_per_worker,
                            num_workers,
                            nstep,
                            pad_action=pad_action,
                            fetch_every=100,
                            save_snapshot=save_buffer,
                            max_act_dim=max_act_dim) # TODO: add max act dim, and pad action


    loader = torch.utils.data.DataLoader(iterable,
                                         batch_size=batch_size,
                                         num_workers=num_workers,
                                         pin_memory=True,
                                         worker_init_fn=_worker_init_fn)
    return loader