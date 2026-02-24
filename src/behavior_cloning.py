import warnings
from time import time

import os
os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'

import time
import random
from pathlib import Path
import hydra
from tqdm import tqdm
import numpy as np
import torch
import wandb
from collections import namedtuple
import omegaconf
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
from einops import rearrange

import envs
import utils
from replay_buffer import ReplayBufferStorage, make_replay_loader
from dreamer.dreamer import DreamerAgent
import dreamer.nets as nets

warnings.filterwarnings('ignore', category=DeprecationWarning)

torch.backends.cudnn.benchmark = True


class BCAgent(nn.Module):
    def __init__(self, obs_shape, act_dim, **kwargs):
        super().__init__()
        self._obs_horizon = kwargs.get('obs_horizon', 3)
        self._encoder = nets.Encoder(obs_shape)
        # Computing embed dim
        with torch.no_grad():
            dummy_input = torch.zeros((1,) + obs_shape)
            outs = self._encoder(dummy_input)
            embed_dim = outs.shape[-1]
        self._actor = nn.Sequential(nn.Linear(embed_dim, 512), nn.LayerNorm(512), nn.ELU(),
                                    nn.Linear(512, 512), nn.LayerNorm(512), nn.ELU(),
                                    nn.Linear(512, act_dim))

        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        self.aug = utils.RandomShiftsAug(3)

    @property
    def device(self):
        return next(self.parameters()).device

    def act(self, obs):
        obs = obs/255. - 0.5
        action = self._actor(self._encoder(obs[None]))
        return action
    
    def preprocess_data(self, data):
        #|o|o|o|     observations: 3
        #| | |p|     actions predicted: 8

        ret = {}
        obs = data['observation'][:, :self._obs_horizon] / 255. - 0.5
        obs = obs.to(device=self.device, dtype=torch.float32)
        b, t, c, h, w = obs.shape
        # reshape and augment observations
        aug_obs = self.aug(rearrange(obs, 'b t c h w -> b (t c) h w'))

        ret['observation'] = aug_obs
        ret['action'] = data['action'][:, self._obs_horizon:self._obs_horizon+1].to(device=self.device, dtype=torch.float32) 
        ret['mask'] = data['action_mask'][:, self._obs_horizon:self._obs_horizon+1].to(device=self.device, dtype=torch.float32)
        # remove the time dim of action
        ret['action'] = ret['action'].squeeze(-2)
        ret['mask'] = ret['mask'].squeeze(-2)
        return ret

    def update(self, data):
        data = self.preprocess_data(data)
        action_pred = self._actor(self._encoder(data['observation']))
        losses = F.mse_loss(action_pred, data['action'], reduction='none')
        
        loss = (losses * data['mask']).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {'bc_loss': loss.item()}

    def save_model(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        state_dict = torch.load(path)
        self.load_state_dict(state_dict)
