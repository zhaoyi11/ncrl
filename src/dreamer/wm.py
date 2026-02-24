from math import ceil
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import numpy as np
from omegaconf import open_dict

import utils
import dreamer.dreamer_utils as common
import dreamer.rssm as rssm
import dreamer.nets as nets

def stop_gradient(x):
  return x.detach()

Module = nn.Module

class WorldModel(Module):
  def __init__(self, config, obs_shape, act_dim): # action dim is a pre-defined value, which corresponds to the humanoid domain.
    super().__init__()
    self.cfg = config
    self.device = config.device

    # encoder model
    self.encoder = nets.Encoder(obs_shape, **config.encoder)
    
    # Computing embed dim
    with torch.no_grad():
      dummy_input = torch.zeros((1,) + obs_shape)
      outs = self.encoder(dummy_input)
      embed_dim = outs.shape[-1]
    self.embed_dim = embed_dim

    self.rssm = rssm.RSSM(embed_dim=embed_dim, action_dim=act_dim, **config.rssm)
    self._use_amp = (config.precision == 16)

    inp_size = config.rssm.deter + config.rssm.stoch * config.rssm.classes
    # decoder model
    self.decoder = nets.Decoder(obs_shape, config.rssm.deter, config.rssm.stoch*config.rssm.classes, **config.decoder)

    if config.mode != 'pretrain':
      # learn reward function if not pretraining
      self.reward_model = nets.MLPs(config.rssm.deter, config.rssm.stoch * config.rssm.classes, 255, **config.reward)
    
    self.two_hot = common.SymTwoHot(255, -20, 20)
    # optim
    self.model_opt = common.Optimizer('model', self.parameters(), **config.model_opt, use_amp=self._use_amp)


  def update(self, data, state=None):
    with common.RequiresGrad(self):
      with torch.amp.autocast(device_type='cuda', enabled=self._use_amp):
        model_loss, state, outputs, metrics = self.loss(data, state)
      metrics.update(self.model_opt(model_loss, self.parameters())) 
    return state, outputs, metrics

  def update_reward(self, deter, stoch, rew_target):
    assert hasattr(self, 'reward_model'), 'No reward model found.'

    with common.RequiresGrad(self):
      with torch.amp.autocast(device_type='cuda', enabled=self._use_amp):
        logits = self.reward_model(deter, stoch)
        loss = self.two_hot.loss(logits, rew_target)
      metrics = self.model_opt(loss, self.parameters())
    return metrics

  def loss(self, data, state=None):
    embed = self.encoder(data['observation'])
    post, prior = self.rssm.observe(
        embed, data['action'], data['is_first'], state)
    # kl loss
    kl_losses, kl_metrics = self.rssm.loss(post, prior)
    
    losses = {**kl_losses}

    # reconstruction loss
    feat = self.rssm.get_feat(post) 
    recon = self.decoder(post['deter'], post['stoch'])
    losses['recon'] = F.mse_loss(recon, 
                                data['observation'],  # [0, 1]
                                reduction='none'
                                ).sum([2,3,4]).mean([0, 1]) # sum over pixels and mean over batch and time

    model_loss = sum(
        self.cfg.loss_scales.get(k, 1.0) * v for k, v in losses.items())
    
    outs = dict(
        embed=embed, feat=feat, post=post,
        prior=prior,)
    
    metrics = {f'{name}_loss': value.item() for name, value in losses.items()}
    metrics.update(**kl_metrics)
    last_state = {k: v[:, -1] for k, v in post.items()}
    return model_loss, last_state, outs, metrics

  def imagine(self, policy, start, is_terminal, horizon, eval_policy=False):
    flatten = lambda x: x.reshape([-1] + list(x.shape[2:]))
    start = {k: flatten(v) for k, v in start.items() if v is not None}
    if 'task_id' in start.keys():
      task_id = start['task_id']
    else:
      task_id = None
    
    start['feat'] = self.rssm.get_feat(start)
    start['action'] = torch.zeros_like(policy(start['deter'], start['stoch'], task_id).mean, device=self.device) #.mode())
    seq = {k: [v] for k, v in start.items()}
    
    for _ in range(horizon):
      deter = seq['deter'][-1]
      stoch = seq['stoch'][-1]
      policy_output = policy(stop_gradient(deter), stop_gradient(stoch), task_id)
      action = policy_output.sample if not eval_policy else policy_output.mean
      state = self.rssm.img_step({k: v[-1] for k, v in seq.items()}, action)
      feat = self.rssm.get_feat(state)
      for key, value in {**state, 'action': action, 'feat': feat}.items():
        seq[key].append(value)
      if task_id is not None:
        seq['task_id'].append(task_id)

    # shape will be (T, B, *DIMS)
    seq = {k: torch.stack(v, 0) for k, v in seq.items()}

    disc = self.cfg.discount * torch.ones(list(seq['feat'].shape[:-1]) + [1], device=self.device)
    seq['discount'] = disc
    # Shift discount factors because they imply whether the following state
    # will be valid, not whether the current state is valid.
    seq['weight'] = torch.cumprod(
        torch.cat([torch.ones_like(disc[:1], device=self.device), disc[:-1]], 0), 0)
    return seq

  def video_pred(self, data, nvid=8):
    # put data on device
    data = {k: torch.as_tensor(np.copy(v), device=self.device) for k, v in data.items()}
    truth = data['observation'][:nvid] + 0.5
    embed = self.encoder(data['observation'])
    states, _ = self.rssm.observe(
        embed[:nvid, :5], data['action'][:nvid, :5], data['is_first'][:nvid, :5])
    # recon = self.decoder(self.rssm.get_feat(states))[:nvid]
    recon = self.decoder(states['deter'], states['stoch'])[:nvid]
    init = {k: v[:, -1] for k, v in states.items()}
    prior = self.rssm.imagine(data['action'][:nvid, 5:], init)
    prior_recon = self.decoder(prior['deter'], prior['stoch'])
    # prior_recon = self.decoder(self.rssm.get_feat(prior))
    model = torch.clip(torch.cat([recon[:, :5] + 0.5, prior_recon + 0.5], 1), 0, 1)
    error = (model - truth + 1) / 2

    video = torch.cat([truth, model, error], 3)
    B, T, C, H, W = video.shape
    return video 
