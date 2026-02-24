import torch.nn as nn
import torch
import torch.nn.functional as F

import utils
import dreamer.dreamer_utils as common
import dreamer.nets as nets
from collections import OrderedDict
import numpy as np
from omegaconf import open_dict

def stop_gradient(x):
  return x.detach()

Module = nn.Module

class ValueEMA:
    """running offset and scale. """

    def __init__(self, device, alpha=1e-2):
        self.device = device
        self.alpha = alpha
        self.range = torch.tensor([0.05, 0.95], device=device)

    def __call__(self, x, ema_vals):
        flat_x = torch.flatten(x.detach())
        x_quantile = torch.quantile(input=flat_x, q=self.range)
        # this should be in-place operation
        ema_vals[:] = self.alpha * x_quantile + (1 - self.alpha) * ema_vals
        scale = torch.clip(ema_vals[1] - ema_vals[0], min=1.0)
        offset = ema_vals[0]
        return offset.detach(), scale.detach()

class ActorCritic(Module):
  def __init__(self, config, act_dim, padded_act_dim, use_aux_critic=False):
    super().__init__()
    self.cfg = config
    self._use_amp = (config.precision == 16)
    self.device = config.device
    
    self.actor = nets.GaussianPolicy(config.rssm.deter, config.rssm.stoch * config.rssm.classes, act_dim, padded_act_dim, **config.actor)
    self.critic = nets.MLPs(config.rssm.deter, config.rssm.stoch * config.rssm.classes, 255, **config.critic)
    
    if self.cfg.slow_target:
      self._target_critic = nets.MLPs(config.rssm.deter, config.rssm.stoch * config.rssm.classes, 255, **config.critic)
      self._updates = 0 
    else:
      self._target_critic = self.critic
    
    self.two_hot = common.SymTwoHot(255, -20, 20)

    self.critic_opt = common.Optimizer('critic', self.critic.parameters(), **self.cfg.critic_opt, use_amp=self._use_amp)
    self.actor_opt = common.Optimizer('actor', self.actor.parameters(), **self.cfg.actor_opt, use_amp=self._use_amp)

    self.register_buffer('ema_vals', torch.zeros((2,), device=self.device))
    self.return_ema = ValueEMA(self.device)


  def update(self, world_model, start, is_terminal, reward_fn):
    metrics = {}
    hor = self.cfg.imag_horizon
    # The weights are is_terminal flags for the imagination start states.
    # Technically, they should multiply the losses from the second trajectory
    # step onwards, which is the first imagined step. However, we are not
    # training the action that led into the first step anyway, so we can use
    # them to scale the whole sequence.

    with common.RequiresGrad(self.actor):
      with torch.amp.autocast(device_type='cuda', enabled=self._use_amp):
        seq = world_model.imagine(self.actor, start, is_terminal, hor)
        # import ipdb; ipdb.set_trace()        
        seq['reward'] = reward = reward_fn(seq)
        target, meta, mets2 = self.target(seq)
        actor_loss, mets3 = self.actor_loss(seq, target)
      metrics.update(self.actor_opt(actor_loss, self.actor.parameters()))
    
    with common.RequiresGrad(self.critic):
      with torch.amp.autocast(device_type='cuda', enabled=self._use_amp):
        seq = {k: stop_gradient(v) for k,v in seq.items()}
        critic_loss, mets4 = self.critic_loss(seq, meta['target'])
      metrics.update(self.critic_opt(critic_loss, self.critic.parameters()))

    metrics.update(**mets2, **mets3, **mets4,)
    self.update_slow_target()  # Variables exist after first forward pass.
    return metrics

  def actor_loss(self, seq, target): #, step):
    # Actions:      0   [a1]  [a2]   a3
    #                  ^  |  ^  |  ^  |
    #                 /   v /   v /   v
    # States:     [z0]->[z1]-> z2 -> z3
    # Targets:     t0   [t1]  [t2]
    # Baselines:  [v0]  [v1]   v2    v3
    # Entropies:        [e1]  [e2]
    # Weights:    [ 1]  [w1]   w2    w3
    # Loss:              l1    l2
    metrics = {}
    # Two states are lost at the end of the trajectory, one for the boostrap
    # value prediction and one because the corresponding action does not lead
    # anywhere anymore. One target is lost at the start of the trajectory
    # because the initial state comes from the replay buffer.
    policy = self.actor(stop_gradient(seq['deter'][:-2]),
                        stop_gradient(seq['stoch'][:-2]),
                        action=stop_gradient(seq['action'][1:-1]), # log_prob(action) will be returned in policy.log_pi
                        )
    if self.cfg.actor_grad == 'dynamics':
      objective = target[1:]
    elif self.cfg.actor_grad == 'reinforce':
      offset, scale = self.return_ema(target[1:], self.ema_vals)
      normed_target = (target[1:] - offset) / scale
      baseline = self.two_hot.decode(self._target_critic(seq['deter'][:-2],
                                                          seq['stoch'][:-2]))
      normed_baseline = (baseline - offset) / scale
      adv = stop_gradient(normed_target - normed_baseline)
      objective = policy.log_pi[:,:,None] * adv
    ent = policy.entropy[:, :, None]
    assert ent.ndim == objective.ndim, "entropy and objective should have the same shape"
    objective += self.cfg.actor_ent * ent
    weight = stop_gradient(seq['weight'])
    actor_loss = -(weight[:-2] * objective).mean() 
    metrics['actor_ent'] = ent.mean()
    if self.cfg.actor_grad == 'reinforce':
      metrics['actor_adv'] = adv.mean()
      metrics['ret_offset'] = offset.mean()
      metrics['ret_scale'] = scale.mean()
      metrics['ret_005quantile'] = self.ema_vals[0].mean()
      metrics['ret_095quantile'] = self.ema_vals[1].mean()
    return actor_loss, metrics

  def critic_loss(self, seq, target):
    # States:     [z0]  [z1]  [z2]   z3
    # Rewards:    [r0]  [r1]  [r2]   r3
    # Values:     [v0]  [v1]  [v2]   v3
    # Weights:    [ 1]  [w1]  [w2]   w3
    # Targets:    [t0]  [t1]  [t2]
    # Loss:        l0    l1    l2
    pred = self.critic(seq['deter'][:-1], seq['stoch'][:-1])
    # L, B, M, _ = pred.shape
    target = stop_gradient(target)
    weight = stop_gradient(seq['weight'])[:-1]

    critic_loss = self.two_hot.loss(pred, target, reduction='none')
    loss = (critic_loss * weight.squeeze(-1)).mean()

    metrics = {'critic': self.two_hot.decode(pred).mean().item()} # .mode().mean()}
    return loss, metrics

  def target(self, seq):
    # States:     [z0]  [z1]  [z2]  [z3]
    # Rewards:    [r0]  [r1]  [r2]   r3
    # Values:     [v0]  [v1]  [v2]  [v3]
    # Discount:   [d0]  [d1]  [d2]   d3
    # Targets:     t0    t1    t2
    meta = {}
    reward = seq['reward'] 
    disc = seq['discount']

    value = self.two_hot.decode(
      self._target_critic(seq['deter'], seq['stoch'])
    )

    # Skipping last time step because it is used for bootstrapping.
    target = common.lambda_return(
        reward[:-1], value[:-1], disc[:-1],
        bootstrap=value[-1],
        lambda_=self.cfg.discount_lambda,
        axis=0)
    meta['target'] = target

    metrics = {}
    metrics['critic_slow'] = value.mean()
    metrics['critic_target'] = target.mean()
    return target, meta, metrics

  def update_slow_target(self):
    """ Slowly update the target_critic weights."""
    if self.cfg.slow_target:
      if self._updates % self.cfg.slow_target_update == 0:
        mix = 1.0 if self._updates == 0 else float(
            self.cfg.slow_target_fraction)
        for s, d in zip(self.critic.parameters(), self._target_critic.parameters()):
          d.data = mix * s.data + (1 - mix) * d.data
      self._updates += 1 

