import re
import numpy as np

import utils
import torch.nn as nn
import torch
import torch.distributions as D
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange
Module = nn.Module

import dreamer.nets as nets
import dreamer.dreamer_utils as common


class BlockGRUCell(Module):
  """ GRUCell with Block Linear Lyaers. """
  def __init__(self, deter, stoch, action, hidden, blocks, act):
    super().__init__()
    self._blocks = blocks
    self._act = getattr(nn, act)()

    self._in_stoch = nn.Sequential(nn.Linear(stoch, hidden), nn.LayerNorm(hidden), self._act)
    self._in_deter = nn.Sequential(nn.Linear(deter, hidden), nn.LayerNorm(hidden), self._act)
    self._in_action = nn.Sequential(nn.Linear(action, hidden), nn.LayerNorm(hidden), self._act)

    inp_size = deter + 3 * hidden * blocks
    self._dynlayers = nn.Sequential(
        nets.BlockLinear(inp_size, deter, blocks, bias=True),
        nn.LayerNorm(deter),
        self._act
    )
    self._dyngru = nets.BlockLinear(deter, 3*deter, blocks, bias=True)

    self._flat2group = lambda x: rearrange(x, '... (g h) -> ... g h', g=blocks)
    self._group2flat = lambda x: rearrange(x, '... g h -> ... (g h)', g=blocks)

  def forward(self, deter, stoch, action):
    assert deter.ndim == stoch.ndim == action.ndim, "deter, stoch, action must have the same dimension."
    x0 = self._in_deter(deter)
    x1 = self._in_stoch(stoch)
    x2 = self._in_action(action)
    x = torch.cat([x0, x1, x2], -1)[:, None, :].repeat(1, self._blocks, 1) # shpae [B, blocks, D]
    x = self._group2flat(torch.cat([self._flat2group(deter), x], -1))
    # dynlayers
    x = self._dynlayers(x)
    # dyngru
    x = self._dyngru(x)
    reset, cand, update = torch.chunk(x, 3, -1)
    reset = torch.sigmoid(reset)
    cand = F.tanh(reset * cand)
    update = torch.sigmoid(update - 1)
    output = update * cand + (1 - update) * deter
    return output, output

class OneHotDist(D.OneHotCategorical):
  """ OneHotCategorical distribution with unimix_ratio. """
  def __init__(self, logits=None, probs=None, unimix_ratio=0.0):
    if logits is not None and unimix_ratio > 0.0:
      probs = F.softmax(logits, dim=-1)
      probs = probs * (1.0 - unimix_ratio) + unimix_ratio / probs.shape[-1]
      logits = torch.log(probs)
      super().__init__(logits=logits, probs=None)
    else:
      super().__init__(logits=logits, probs=probs)

  def mode(self):
    _mode = F.one_hot(
        torch.argmax(super().logits, axis=-1), super().logits.shape[-1]
    )
    return _mode.detach() + super().logits - super().logits.detach()

  def sample(self, sample_shape=(), seed=None):
    if seed is not None:
      raise ValueError("need to check")
    sample = super().sample(sample_shape).detach()
    probs = super().probs
    while len(probs.shape) < len(sample.shape):
      probs = probs[None]
    sample += probs - probs.detach()
    return sample


class RSSM(Module):
  def __init__(self,
          embed_dim: int, # dimension of enc(observation)
          action_dim: int, # dimension of action
          deter: int = 4096,
          hidden: int = 2048,
          stoch: int = 32,
          classes: int = 32,
          act: str = 'silu',
          unimix: float = 0.01,
          blocks: int = 8,
          free_nats: float = 1.0,
          **kwargs):
    super().__init__()
    assert deter % blocks == 0, "deter must be divisible by blocks"
    self._deter = deter
    self._stoch = stoch
    self._classes = classes
    self._act = getattr(nn, act)()
    self._free_nats = free_nats

    ###### inner nets ######
    # gru 
    self._gru = BlockGRUCell(deter, stoch * classes, action_dim, hidden, blocks, act)

    # prior head (deter -> layers)
    self._prior = nn.Sequential( # return prior logits
        nn.Linear(deter, hidden), nn.LayerNorm(hidden), self._act,
        nn.Linear(hidden, hidden), nn.LayerNorm(hidden), self._act,
        nn.Linear(hidden, stoch * classes),
        Rearrange('... (s c) -> ... s c', s=stoch, c=classes), 
    )

    # posterior head ([deter, embed] -> logits)
    inp_size = self._deter + embed_dim
    self._posterior = nn.Sequential( # return posterior logits
        nn.Linear(inp_size, hidden), nn.LayerNorm(hidden), self._act,
        nn.Linear(hidden, stoch * classes),
        Rearrange('... (s c) -> ... s c', s=stoch, c=classes), 
    )

    self._get_dist = lambda logits: D.Independent(OneHotDist(logits, unimix_ratio=unimix), 1)

  @property
  def device(self):
    return next(self.parameters()).device

  def initial(self, batch_size: int):
    return dict(
        logits=torch.zeros([batch_size, self._stoch, self._classes], device=self.device),
        stoch=torch.zeros([batch_size, self._stoch, self._classes], device=self.device),
        deter=torch.zeros([batch_size, self._deter], device=self.device))

  def get_feat(self, state):
    # get representation [z, h]
    stoch = state['stoch']
    shape = list(stoch.shape[:-2]) + [self._stoch * self._classes]
    stoch = stoch.reshape(shape)
    return torch.cat([stoch, state['deter']], -1)

  #### Observation and Imagination ####

  def observe(self, embed, action, reset, state=None):
    """ Calculate the prior and posterior given inp, action, reset and last_state."""
    swap = lambda x: x.permute([1, 0] + list(range(2, len(x.shape))))
    if state is None:
      state = self.initial(action.shape[0])
    post, prior = common.static_scan(
        lambda prev, inputs: self.obs_step(prev[0], *inputs),
        (swap(action), swap(embed), swap(reset)), (state, state))
    post = {k: swap(v) for k, v in post.items()}
    prior = {k: swap(v) for k, v in prior.items()}
    return post, prior
    
  def imagine(self, action, state=None):
      swap = lambda x: x.permute([1, 0] + list(range(2, len(x.shape))))
      if state is None:
        state = self.initial(action.shape[0])
      assert isinstance(state, dict), state
      action = swap(action)
      prior = common.static_scan(self.img_step, [action], state, unpack=True)[0] 
      prior = {k: swap(v) for k, v in prior.items()}
      return prior

  def obs_step(self, prev_state, prev_action, embed, reset, should_sample=True):
    # maskout state and action if reset 
    prev_state = { k: torch.einsum('b,b...->b...', 1.0 - reset.float(), x) for k, x in prev_state.items()}
    prev_action = torch.einsum('b,b...->b...', 1.0 - reset.float(), prev_action)
    # calculate prior
    prior = self.img_step(prev_state, prev_action, should_sample)
    # calculate posterior 
    x = torch.cat([prior['deter'], embed], -1)
    logits = self._posterior(x)
    dist = self._get_dist(logits)
    stoch = dist.sample() if should_sample else None 
    post = {'stoch': stoch, 'deter': prior['deter'], 'logits': logits}
    return post, prior

  def img_step(self, prev_state, prev_action, sample=True):
    prev_stoch = prev_state['stoch'] 
    shape = list(prev_stoch.shape[:-2]) + [self._stoch * self._classes]
    prev_stoch = prev_stoch.reshape(shape) 
    deter = prev_state['deter']
    # gru step
    x, deter = self._gru(deter, prev_stoch, prev_action)     
    # calculate prior
    logits = self._prior(x)
    dist = self._get_dist(logits)
    stoch = dist.sample() if sample else None 
    prior = {'stoch': stoch, 'deter': deter, 'logits': logits}
    return prior    

  def loss(self, posterior, prior):
    kld = D.kl_divergence
    sg = lambda x: {k: v.detach() for k, v in x.items()} 
    
    dyn_loss = kld(self._get_dist(posterior['logits'].detach()), 
                    self._get_dist(prior['logits'])).mean()
    repr_loss = kld(self._get_dist(posterior['logits']),
                    self._get_dist(prior['logits'].detach())).mean()
    if self._free_nats:
      free_tensor = torch.tensor([self._free_nats], dtype=dyn_loss.dtype, device=self.device)
      dyn_loss = torch.maximum(dyn_loss, free_tensor)
      repr_loss = torch.maximum(repr_loss, free_tensor)
    losses = {'dyn_loss': dyn_loss, 'repr_loss': repr_loss}
    metrics = {}
    metrics['dyn_ent'] = self._get_dist(prior['logits']).entropy().mean().item()
    metrics['repr_ent'] = self._get_dist(posterior['logits']).entropy().mean().item()

    return losses, metrics
