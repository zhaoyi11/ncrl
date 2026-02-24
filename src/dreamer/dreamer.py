import torch.nn as nn
import torch
import torch.nn.functional as F
from functools import partial
import collections
import random

import utils
import dreamer.dreamer_utils as common
from collections import OrderedDict
import numpy as np
from omegaconf import open_dict

from dreamer.wm import WorldModel
from dreamer.actor_critic import ActorCritic
import dreamer.nets as nets
from envs import TASK_DICT, TASK_ACT_DIM


def stop_gradient(x):
  return x.detach()

Module = nn.Module

class DreamerAgent(Module):

  def __init__(self, cfg, obs_shape, act_dim, padded_act_dim, **kwargs):
    super().__init__()
    # add kwargs to cfg
    with open_dict(cfg):
      cfg.update(**kwargs)
    self.cfg = cfg
    self._use_amp = (cfg.precision == 16)
    self.device = device = self.cfg.device
    self.padded_act_dim = padded_act_dim
    self.wm = WorldModel(cfg, obs_shape, padded_act_dim) 
    self.two_hot = common.SymTwoHot(255, -20, 20)
    
    if cfg.mode in ['scratch', 'finetune']:
      # actor critic
      self._task_behavior = ActorCritic(cfg, act_dim, padded_act_dim, use_aux_critic=True) # TODO: what is use_aux_critic?

    self.to(device)
    self.requires_grad_(requires_grad=False)

  def preprocess_img(self, img):
    assert img.dtype in [np.uint8, torch.uint8], 'Image must be uint8.'
    return img / 255.0 - 0.5

  @torch.no_grad()  
  def act(self, time_step, step, eval_mode, state, **kwargs):
    time_step = time_step._asdict()
    time_step['observation'] = self.preprocess_img(time_step['observation'])
    time_step = {k : torch.as_tensor(np.copy(v), device=self.device).unsqueeze(0).float() for k, v in time_step.items() if k != 'info'}
    B = 1 # ! batch size is set to 1, only allow one environment at a time
    if state is None:
      latent = self.wm.rssm.initial(B)
      action = torch.zeros((B,) + (self.padded_act_dim,), device=self.device)
    else:
      latent, action = state
    
    embed = self.wm.encoder(time_step['observation'])

    should_sample = (not eval_mode) or (not self.cfg.eval_state_mean)
    latent, _ = self.wm.rssm.obs_step(latent, action, embed, time_step['is_first'], should_sample)

    if eval_mode:
      action = self._task_behavior.actor(latent['deter'], latent['stoch']).mean
    else:
      action = self._task_behavior.actor(latent['deter'], latent['stoch']).sample

    return action, latent

  def update_wm(self, data, step):
    metrics = {}
    state, outputs, mets = self.wm.update(data, state=None)
    outputs['is_terminal'] = data['is_terminal']
    metrics.update(mets)
    return state, outputs, metrics

  def update(self, online_data, offline_data, step):
    if (not online_data) and offline_data:
      offline_data = {k: torch.as_tensor(np.copy(v), device=self.device) for k, v in offline_data.items()}
      data = offline_data
    elif online_data and (not offline_data):
      online_data = {k: torch.as_tensor(np.copy(v), device=self.device) for k, v in online_data.items()}
      data = online_data
    else:
      # merge data
      data = {k: torch.cat([v, offline_data[k]], dim=0) for k, v in online_data.items()}
      # put data on device
      data = {k: torch.as_tensor(np.copy(v), device=self.device) for k, v in data.items()}
      online_data = {k: torch.as_tensor(np.copy(v), device=self.device) for k, v in online_data.items()}
      
    data['observation'] = self.preprocess_img(data['observation'])

    # update world model
    state, outputs, metrics = self.update_wm(data, step)

    if self.cfg.mode in ['scratch', 'finetune']: # actor critic are updated during scratch and finetune stage
      # update reward fn with online data
      num_online_data = online_data['reward'].shape[0]
      metrics.update(self.wm.update_reward(
        outputs['post']['deter'][:num_online_data].detach(),
        outputs['post']['stoch'][:num_online_data].detach(),
        online_data['reward']))
      
      # update actor critic
      start = outputs['post']
      start = {k: stop_gradient(v) for k,v in start.items()}
      
      def reward_fn(seq):
        assert self.wm.reward_model.num_nets == 1, "Only one reward network is supported."
        return self.two_hot.decode(self.wm.reward_model(seq['deter'], seq['stoch']))

      metrics.update(self._task_behavior.update(
          self.wm, start, data['is_terminal'], partial(reward_fn)))

    return state, metrics

  @torch.no_grad()
  def report(self, data):
    report = {}
    data['observation'] = self.preprocess_img(data['observation'])
    report[f'open_loop_pred'] = self.wm.video_pred(data)
    return report

  def get_meta_specs(self):
    return tuple()

  def init_meta(self):
    return OrderedDict()

  def update_meta(self, meta, global_step, time_step, finetune=False):
    return meta

  def save_model(self, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    weight_dict = {
      'wm':{
        'encoder': self.wm.encoder.state_dict(),
        'rssm': self.wm.rssm.state_dict(),
        'decoder': self.wm.decoder.state_dict(),
        
      }}
    if self.cfg.mode in ['scratch', 'finetune']:
      weight_dict.update(
        {'reward': self.wm.reward_model.state_dict(),
        'actor': self._task_behavior.actor.state_dict(),
        'critic': self._task_behavior.critic.state_dict(),
        'target_critic': self._task_behavior._target_critic.state_dict(),}
      )

    torch.save(weight_dict, path)

  def load(self, path, load_model_dict=None):
    """ Load models' parameters from a checkpoint."""
    params_dict = torch.load(path)
    
    if load_model_dict is None or sum(load_model_dict.values()) == 0:
      print("No model to load.")

    if load_model_dict["wm"]:
      # copy parameters over
      print(f"Copying the pretrained world model")
      self.wm.rssm.load_state_dict(params_dict['wm']['rssm'])
      self.wm.encoder.load_state_dict(params_dict['wm']['encoder'])
      self.wm.decoder.load_state_dict(params_dict['wm']['decoder'])

    if load_model_dict["actor"]:
      print(f"Copying the pretrained actor")
      self._task_behavior.actor.load_state_dict(params_dict['actor'])
    
    if load_model_dict["critic"]:
      print(f"Copying the pretrained critic")
      self._task_behavior.critic.load_state_dict(params_dict['critic'])
      if self.cfg.slow_target:
        self._task_behavior._target_critic.load_state_dict(params_dict['target_critic'])
