import re
from typing import Any, Union
import math
from collections import namedtuple
import warnings
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
import torch.distributions as D
from torch.distributions.utils import _standard_normal
import torch.nn.functional as F
from torch import FloatTensor
from torch.masked import as_masked_tensor, MaskedTensor
from tensordict import from_modules
from einops import rearrange

import utils
from envs import TASK_ACT_DIM, TASK_DICT

# Disable prototype warnings and such
warnings.filterwarnings(action='ignore', category=UserWarning)

Module = nn.Module


class TruncatedNormal(D.Normal):
    def __init__(self, loc, scale, low=-1.0, high=1.0, eps=1e-6):
        super().__init__(loc, scale, validate_args=False)
        self.low = low
        self.high = high
        self.eps = eps

    def _clamp(self, x):
        clamped_x = torch.clamp(x, self.low + self.eps, self.high - self.eps)
        x = x - x.detach() + clamped_x.detach()
        return x

    def rsample(self, sample_shape=torch.Size(), clip=None):
        shape = self._extended_shape(sample_shape)
        eps = _standard_normal(shape,
                               dtype=self.loc.dtype,
                               device=self.loc.device)
        eps *= self.scale
        if clip is not None:
            eps = torch.clamp(eps, -clip, clip)
        x = self.loc + eps
        return self._clamp(x)

    def sample(self, shape=torch.Size(), clip=None):
        return self.rsample(shape, clip)


class BlockLinear(nn.Module):
    """ 
    Block Linear Layer    
    Ref 1 https://pytorch.org/docs/stable/_modules/torch/nn/modules/linear.html#Linear
        2. https://github.com/danijar/dreamerv3/blob/f8817c4040cebada9bb9712554b0234c70c291d5/embodied/jax/nets.py#L254
    """
    def __init__(self, in_features, out_features, blocks, bias=True, device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(BlockLinear, self).__init__()
        
        assert out_features % blocks == 0, "out_features must be divisible by blocks"
        assert in_features % blocks == 0, "in_features must be divisible by blocks"
        
        self.in_features = in_features
        self.out_features = out_features
        self.blocks = blocks
        self.bias_flag = bias
        
        block_in = in_features // blocks
        block_out = out_features // blocks
        
        self.weight = nn.Parameter(torch.empty((blocks, block_in, block_out), **factory_kwargs))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)
        
        self.reset_parameters()

    def reset_parameters(self):
                # Initialize weights using nn.Linear defaults
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias_flag:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        batch_shape = x.shape[:-1]
        x = x.view(*batch_shape, self.blocks, self.in_features // self.blocks)
        x = torch.einsum('...ki,kio->...ko', x, self.weight)
        x = x.reshape(*batch_shape, self.out_features)
        if self.bias_flag:
            x = x + self.bias
        return x


class ChannelNorm(Module): 
  """ Follow the implementation of ConvNeXt's Layer normalization.
      In ConNeXt, the layer normalization is applied to the channel dimension and not the spatial dimension.
      https://github.com/facebookresearch/ConvNeXt/blob/048efcea897d999aed302f2639b6270aedf8d4c8/models/convnext.py#L119
  """
  def __init__(self, norm, channels, eps=1e-03):
    super().__init__()
    if norm == 'layer':
      self._norm = nn.LayerNorm
    else:
      raise NotImplementedError(f'norm {norm} is not implemented')
    self._norm = self._norm(channels, eps=eps)

  def forward(self, x):
    # x shape: [B, C, H, W]
    x = x.permute(0, 2, 3, 1)
    x = self._norm(x)
    x = x.permute(0, 3, 1, 2)
    return x


class Encoder(Module):
  def __init__(
      self, obs_shape, act='ELU',
      cnn_depth=48, use_norm=True, cnn_kernels=(4, 4, 4, 4)):
    super().__init__()
    self.obs_shape = obs_shape
    self._channel_dim = obs_shape[0]
    self._act = getattr(nn, act)()
    self._cnn_depth = cnn_depth
    self._cnn_kernels = cnn_kernels

    # cnn model
    _conv_model = []
    for i, kernel in enumerate(self._cnn_kernels):
      if i == 0:
        prev_depth = self._channel_dim
      else:
        prev_depth = 2 ** (i-1) * self._cnn_depth  
      depth = 2 ** i * self._cnn_depth
      _conv_model.append(nn.Conv2d(prev_depth, depth, kernel, stride=2))
      if use_norm:
        _conv_model.append(ChannelNorm('layer', depth))
      _conv_model.append(self._act)
    self._conv_model = nn.Sequential(*_conv_model)

    # reshape
    self._reshape = lambda x, batch_dims: x.reshape((-1,) + tuple(x.shape)[len(batch_dims):])

  def forward(self, obs):
    batch_dims = obs.shape[:-len(self.obs_shape)] # batch_dims can be B or (B, T)
    x = self._reshape(obs, batch_dims)
    x = self._conv_model(x)
    x = x.reshape(tuple(x.shape[:-3]) + (-1,)) # [B*T, -1]
    x = x.reshape(batch_dims + x.shape[1:]) # [B, T, -1]    
    return x


class Decoder(Module):
  def __init__(
      self, obs_shape, deter_dim, stoch_dim, units, act='ELU',
      cnn_depth=48, use_norm=True, cnn_kernels=(4, 4, 4, 4), num_block=8):
    super().__init__()
    self._obs_shape = obs_shape

    self._act = getattr(nn, act)()
    self._cnn_depth = cnn_depth
    self._cnn_kernels = cnn_kernels
    self._feature_dim  = 32 * self._cnn_depth
    # cnn
    if num_block > 0:
      assert deter_dim % num_block == 0, 'deter_dim should be divisible by num_block'
      self._fc_deter = nn.Sequential(BlockLinear(deter_dim, 2*units, num_block),
                                    nn.LayerNorm(2*units), self._act,
                                    nn.Linear(2*units, self._feature_dim))

    else:
      self._fc_deter = nn.Sequential(nn.Linear(deter_dim, 2*units),
                                    nn.LayerNorm(2*units), self._act,
                                    nn.Linear(2*units, self._feature_dim))
    self._fc_stoch = nn.Sequential(nn.Linear(stoch_dim, 2*units),
                                  nn.LayerNorm(2*units), self._act,
                                  nn.Linear(2*units, self._feature_dim))
    _conv_model = []
    for i, kernel in enumerate(self._cnn_kernels):
      if i == 0:
        prev_depth = 32*self._cnn_depth
      else:
        prev_depth = 2 ** (len(self._cnn_kernels) - (i - 1) - 2) * self._cnn_depth
      
      depth, act = 2 ** (len(self._cnn_kernels) - i - 2) * self._cnn_depth, self._act
      if i == len(self._cnn_kernels) - 1: # for the output layer
        depth, act = self._obs_shape[0], nn.Identity()
      _conv_model.append(nn.ConvTranspose2d(prev_depth, depth, kernel, stride=2))
      if use_norm and i != len(self._cnn_kernels) - 1: # no norm for the last layer
        _conv_model.append(ChannelNorm('layer', depth))
      _conv_model.append(act)
    self._conv_model = nn.Sequential(*_conv_model)

  def forward(self, deter, stoch):
    assert stoch.ndim - deter.ndim == 1, 'stoch feature is expected not to be flattened (*, stoch, classes)'
    batch_dims = deter.shape[:-1]

    stoch = stoch.reshape(list(stoch.shape[:-2]) + [-1]) # shape [B, T, flatten_stoch_dim]
    deter = deter.reshape(-1, deter.shape[-1]) # shape [B*T, deter_dim]
    stoch = stoch.reshape(-1, stoch.shape[-1]) # shape [B*T, stoch_dim]
    
    deter = self._fc_deter(deter)
    stoch = self._fc_stoch(stoch)
    x = deter + stoch
    x = x.reshape([-1, 32 * self._cnn_depth, 1, 1,]) # shape [B*T, 32*cnn_depth, 1, 1]
    x = self._conv_model(x) 
    x = x.reshape(list(batch_dims) + list(x.shape[1:])) # shape [B, T, obs_shape]
    return x


class MLPs(Module):
  def __init__(self, deter_dim, stoch_dim, out_dim, layers, units, act, blocks=0, num_nets=1, prior_scale=0., use_norm=True):
    super().__init__()
    self._feature_dim = units # both deter and stoch are mapped to the same feature dim
    self._layers = layers
    self._act = getattr(nn, act)()
    if blocks > 0:
      assert deter_dim % blocks == 0, 'deter_dim should be divisible by blocks'
      self._fc_deter = nn.Sequential(BlockLinear(deter_dim, 2*units, blocks),
                                    nn.LayerNorm(2*units), self._act,
                                    nn.Linear(2*units, self._feature_dim))
    else:
      self._fc_deter = nn.Linear(deter_dim, self._feature_dim)
    self._fc_stoch = nn.Linear(stoch_dim, self._feature_dim)
    
    # for ensembles
    self._num_nets = num_nets
    self._prior_scale = prior_scale

    # build the model
    def build_net():
      net = []
      last_units = self._feature_dim
      for index in range(self._layers):
        net.append(nn.Linear(last_units, units))
        if use_norm:
          net.append(nn.LayerNorm(units))
        net.append(self._act)
        last_units = units
      net.append(nn.Linear(units, out_dim))
      return nn.Sequential(*net)

    if num_nets > 1:
      self._net = Ensemble([build_net() for _ in range(num_nets)])
      if prior_scale > 0:
        self._prior_scale = prior_scale
        self._prior = Ensemble([build_net() for _ in range(num_nets)])
    else:
      self._net = build_net()

  @property
  def num_nets(self):
    return self._num_nets

  def forward(self, deter, stoch):
    """ Forward pass of the model.
        Args:
          deter: torch.Tensor, shape (B, T, deter_dim)
          stoch: torch.Tensor, shape (B, T, stoch, classes)
    """
    assert stoch.ndim - deter.ndim == 1, 'stoch feature is expected not to be flattened (*, stoch, classes)'
    batch_dims = deter.shape[:-1]

    stoch = stoch.reshape(list(stoch.shape[:-2]) + [-1]) # shape [B, T, flatten_stoch_dim]
    deter = deter.reshape(-1, deter.shape[-1]) # shape [B*T, deter_dim]
    stoch = stoch.reshape(-1, stoch.shape[-1]) # shape [B*T, stoch_dim]

    deter = self._fc_deter(deter)
    stoch = self._fc_stoch(stoch)
    features = deter + stoch # shape [B*T, feature_dim]

    if self._num_nets > 1:
      # use ensemble
      x = self._net(features)
      if self._prior_scale > 0:
        prior = self._prior(features)
        x = x + self._prior_scale * prior.detach()
        x = x.reshape([x.shape[0]] + list(batch_dims) + [x.shape[-1]]) # shape: num_enesmble, B, T, out_dim
    else:
      x = self._net(features)
      x = x.reshape(list(batch_dims) + [x.shape[-1]]) # reshape back
    return x

class Ensemble(nn.Module):
	"""
	Vectorized ensemble of modules.
    Reference: https://github.com/nicklashansen/tdmpc2/blob/main/tdmpc2/common/layers.py#L7
	"""

	def __init__(self, modules, **kwargs):
		super().__init__()
		# combine_state_for_ensemble causes graph breaks
		self.params = from_modules(*modules, as_module=True)
		with self.params[0].data.to("meta").to_module(modules[0]):
			self.module = deepcopy(modules[0])
		self._repr = str(modules[0])
		self._n = len(modules)

	def __len__(self):
		return self._n

	def _call(self, params, *args, **kwargs):
		with params.to_module(self.module):
			return self.module(*args, **kwargs)

	def forward(self, *args, **kwargs):
		return torch.vmap(self._call, (0, None), randomness="different")(self.params, *args, **kwargs)

	def __repr__(self):
		return f'Vectorized {len(self)}x ' + self._repr


policy_outputs = namedtuple('policy_outputs', ['mean', 'sample', 'entropy', 'log_pi'])
class GaussianPolicy(Module):
  """ Gaussian policy with learned mean and std."""
  def __init__(self, deter_dim, stoch_dim, act_dim, padded_act_dim, layers, units, act, blocks=0, use_norm=True, 
                    min_std=0.1, max_std=1.0, policy_dist='normal', use_action_padding=False):
    super().__init__()
    self._policy_dit = policy_dist

    self._use_action_padding = use_action_padding
    self._act_dim = act_dim
    self._padded_act_dim = padded_act_dim

    self.mlp = MLPs(deter_dim, stoch_dim, 2*padded_act_dim, 
                  layers=layers, 
                  units=units, 
                  act=act, 
                  blocks=blocks, 
                  use_norm=use_norm)
    
    self._min_std, self._max_std = min_std, max_std

    self._reshape = lambda x, batch_dims: x.reshape(list(batch_dims) + [x.shape[-1]]) # reshape back
    

  def forward(self, deter, stoch, action=None):
    batch_dims = deter.shape[:-1]

    x = self.mlp(deter, stoch)
    mu, std = x.chunk(2, dim=-1)

    mu = torch.tanh(mu)
    std = (self._max_std - self._min_std) * torch.sigmoid(std + 2.0) + self._min_std
    
    if self._policy_dit == 'normal':
      dist = D.Normal(mu, std)
    elif self._policy_dit == 'truncated_normal':
      dist = TruncatedNormal(mu, std)
    else:
      raise NotImplementedError(f'{self._policy_dit} is not implemented')

    sample = dist.rsample()
    ent = -dist.log_prob(sample)

    if action is not None: # compute log probability of the action
      log_pi = dist.log_prob(action)
    else:
      log_pi = None

    if self._use_action_padding:
      # apply action mask
      action_mask = torch.zeros(mu.shape, device=mu.device)
      action_mask[..., :self._act_dim] = 1.
      mu = mu * action_mask
      sampe = sample * action_mask
      ent = (ent * action_mask).sum(-1)
      if log_pi is not None:
        log_pi = (log_pi * action_mask).sum(-1)
    else:
      ent = ent.sum(-1)
      if log_pi is not None:
        log_pi = log_pi.sum(-1)
    return policy_outputs(
        mean=mu, sample=sample, entropy=ent, log_pi=log_pi)