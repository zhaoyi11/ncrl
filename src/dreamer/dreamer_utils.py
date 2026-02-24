
import torch.nn as nn
import torch
import torch.nn.functional as F
# We thank the authors of the repo: https://github.com/jsikyoon/dreamer-torch
# For their open source re-implementation, which was used as a reference to develop our code faster


class SymTwoHot(nn.Module):
  """ From https://github.com/weipu-zhang/STORM/blob/main/sub_models/functions_losses.py#L38. """
  def __init__(self, num_classes, lower_bound, upper_bound):
    super().__init__()
    self.num_classes = num_classes
    self.lower_bound = lower_bound
    self.upper_bound = upper_bound
    self.bin_length = (upper_bound - lower_bound) / (num_classes-1)

    # use register buffer so that bins move with .cuda() automatically
    self.bins: torch.Tensor
    self.register_buffer(
        'bins', torch.linspace(-20, 20, num_classes), persistent=False)

  def loss(self, output, target, reduction='mean'):
    """ Compute the symexp twohot loss, eq 11 in the Dreamer v3 paper."""
    target = symlog(target)
    assert target.min() >= self.lower_bound and target.max() <= self.upper_bound
    # calculate the target prob
    with torch.no_grad():
      index = torch.bucketize(target, self.bins)
      diff = target - self.bins[index-1]  # -1 to get the lower bound
      weight = diff / self.bin_length
      weight = torch.clamp(weight, 0, 1)

      # change the shape
      index = index.squeeze(-1) # F.one_hot take index of shape (*) to (*, num_classes)      
      if index.ndim == weight.ndim:
        weight = weight.unsqueeze(-1)
      # weight should have one more dimension than index, such that after using F.one_hot, they have the same dimensions.
      assert weight.ndim - index.ndim == 1 and weight.shape[-1] == 1, (weight.shape, index.shape)

      target_prob = (1-weight)*F.one_hot(index-1, self.num_classes) + weight*F.one_hot(index, self.num_classes)

    loss = -target_prob * F.log_softmax(output, dim=-1)
    loss = loss.sum(dim=-1)
    
    if reduction == 'mean':
      return loss.mean()
    elif reduction == 'none':
      return loss
    else:
      raise ValueError(f"Invalid reduction: {reduction}, only support 'mean' and 'none'.")

  def decode(self, output):
    """ Decode the output to the original scale. """
    _mean = F.softmax(output, dim=-1) * self.bins
    return symexp(_mean.sum(dim=-1, keepdim=True))


def symlog(x):
	"""
	Symmetric logarithmic function.
	Adapted from https://github.com/danijar/dreamerv3.
	"""
	return torch.sign(x) * torch.log(1 + torch.abs(x))


def symexp(x):
	"""
	Symmetric exponential function.
	Adapted from https://github.com/danijar/dreamerv3.
	"""
	return torch.sign(x) * (torch.exp(torch.abs(x)) - 1)


def two_hot(x, cfg):
	"""Converts a batch of scalars to soft two-hot encoded targets for discrete regression."""
	if cfg.num_bins == 0:
		return x
	elif cfg.num_bins == 1:
		return symlog(x)
	x = torch.clamp(symlog(x), cfg.vmin, cfg.vmax).squeeze(1)
	bin_idx = torch.floor((x - cfg.vmin) / cfg.bin_size)
	bin_offset = ((x - cfg.vmin) / cfg.bin_size - bin_idx).unsqueeze(-1)
	soft_two_hot = torch.zeros(x.shape[0], cfg.num_bins, device=x.device, dtype=x.dtype)
	bin_idx = bin_idx.long()
	soft_two_hot = soft_two_hot.scatter(1, bin_idx.unsqueeze(1), 1 - bin_offset)
	soft_two_hot = soft_two_hot.scatter(1, (bin_idx.unsqueeze(1) + 1) % cfg.num_bins, bin_offset)
	return soft_two_hot


def two_hot_inv(x, cfg):
	"""Converts a batch of soft two-hot encoded vectors to scalars."""
	if cfg.num_bins == 0:
		return x
	elif cfg.num_bins == 1:
		return symexp(x)
	dreg_bins = torch.linspace(cfg.vmin, cfg.vmax, cfg.num_bins, device=x.device, dtype=x.dtype)
	x = F.softmax(x, dim=-1)
	x = torch.sum(x * dreg_bins, dim=-1, keepdim=True)
	return symexp(x)


def static_scan_for_lambda_return(fn, inputs, start):
  last = start
  indices = range(inputs[0].shape[0])
  indices = reversed(indices)
  flag = True
  for index in indices:
    inp = lambda x: (_input[x].unsqueeze(0) for _input in inputs)
    last = fn(last, *inp(index))
    if flag:
      outputs = last
      flag = False
    else:
      outputs = torch.cat([last, outputs], dim=0) 
  return outputs

def lambda_return(
    reward, value, pcont, bootstrap, lambda_, axis):
  # Setting lambda=1 gives a discounted Monte Carlo return.
  # Setting lambda=0 gives a fixed 1-step return.
  #assert reward.shape.ndims == value.shape.ndims, (reward.shape, value.shape)
  assert len(reward.shape) == len(value.shape), (reward.shape, value.shape)
  if isinstance(pcont, (int, float)):
    pcont = pcont * torch.ones_like(reward, device=reward.device)
  dims = list(range(len(reward.shape)))
  dims = [axis] + dims[1:axis] + [0] + dims[axis + 1:]
  if axis != 0:
    reward = reward.permute(dims)
    value = value.permute(dims)
    pcont = pcont.permute(dims)
  if bootstrap is None:
    bootstrap = torch.zeros_like(value[-1], device=reward.device)
  if len(bootstrap.shape) < len(value.shape):
    bootstrap = bootstrap[None]
  next_values = torch.cat([value[1:], bootstrap], 0)
  inputs = reward + pcont * next_values * (1 - lambda_)
  returns = static_scan_for_lambda_return(
      lambda agg, cur0, cur1: cur0 + cur1 * lambda_ * agg,
      (inputs, pcont), bootstrap)
  if axis != 0:
    returns = returns.permute(dims)
  return returns

def static_scan(fn, inputs, start, reverse=False, unpack=False):
  last = start
  indices = range(inputs[0].shape[0])
  flag = True
  for index in indices:
    inp = lambda x: (_input[x] for _input in inputs)
    if unpack:
      last = fn(last, inputs[0][index]) 
    else:
      last = fn(last, inp(index)) 
    if flag:
      if type(last) == type({}):
        outputs = {key: value.clone().unsqueeze(0) for key, value in last.items()}
      else:
        outputs = []
        for _last in last:
          if type(_last) == type({}):
            outputs.append({key: value.clone().unsqueeze(0) for key, value in _last.items()})
          else:
            outputs.append(_last.clone().unsqueeze(0))
      flag = False
    else:
      if type(last) == type({}):
        for key in last.keys():
          outputs[key] = torch.cat([outputs[key], last[key].unsqueeze(0)], dim=0)
      else:
        for j in range(len(outputs)):
          if type(last[j]) == type({}):
            for key in last[j].keys():
              outputs[j][key] = torch.cat([outputs[j][key],
                  last[j][key].unsqueeze(0)], dim=0)
          else:
            outputs[j] = torch.cat([outputs[j], last[j].unsqueeze(0)], dim=0)
  if type(last) == type({}):
    outputs = [outputs]
  return outputs


class Optimizer:

  def __init__(
      self, name, parameters, lr, eps=1e-4, clip=None, wd=None,
      opt='adam', wd_pattern=r'.*', use_amp=False):
    assert 0 <= wd < 1
    assert not clip or 1 <= clip
    self._name = name
    self._clip = clip
    self._wd = wd
    self._wd_pattern = wd_pattern
    self._opt = {
        'adam': lambda: torch.optim.Adam(parameters, lr, eps=eps),
        'nadam': lambda: torch.optim.Nadam(parameters, lr, eps=eps),
        'adamax': lambda: torch.optim.Adamax(parameters, lr, eps=eps),
        'sgd': lambda: torch.optim.SGD(parameters, lr),
        'momentum': lambda: torch.optim.SGD(lr, momentum=0.9),
    }[opt]()
    self._scaler = torch.GradScaler("cuda", enabled=use_amp)
    self._once = True

  def __call__(self, loss, params):
    params = list(params)
    assert len(loss.shape) == 0 or (len(loss.shape) == 1 and loss.shape[0] == 1), (self._name, loss.shape)
    metrics = {}

    # Count parameters.
    if self._once:
      count = sum(p.numel() for p in params if p.requires_grad) 
      print(f'Found {count} {self._name} parameters.')
      self._once = False

    # Check loss.
    metrics[f'{self._name}_loss'] = loss.detach().cpu().numpy()

    # Compute scaled gradient.
    self._scaler.scale(loss).backward()
    self._scaler.unscale_(self._opt)

    # Gradient clipping.
    if self._clip:
      norm = torch.nn.utils.clip_grad_norm_(params, self._clip)
      metrics[f'{self._name}_grad_norm'] = norm.item()
  
    # Weight decay.
    if self._wd:
      self._apply_weight_decay(params)
    
    # # Apply gradients.
    self._scaler.step(self._opt)
    self._scaler.update()
    
    self._opt.zero_grad() 
    return metrics

  def _apply_weight_decay(self, varibs):
    nontrivial = (self._wd_pattern != r'.*')
    if nontrivial:
      raise NotImplementedError('Non trivial weight decay')
    else:
      for var in varibs:
        var.data = (1 - self._wd) * var.data


class StreamNorm:

  def __init__(self, shape=(), momentum=0.99, scale=1.0, eps=1e-8, device='cuda'):
    # Momentum of 0 normalizes only based on the current batch.
    # Momentum of 1 disables normalization.
    self.device = device
    self._shape = tuple(shape)
    self._momentum = momentum
    self._scale = scale
    self._eps = eps
    self.mag = torch.ones(shape).to(self.device) 

  def __call__(self, inputs):
    metrics = {}
    self.update(inputs)
    metrics['mean'] = inputs.mean()
    metrics['std'] = inputs.std()
    outputs = self.transform(inputs)
    metrics['normed_mean'] = outputs.mean()
    metrics['normed_std'] = outputs.std()
    return outputs, metrics

  def reset(self):
    self.mag = torch.ones_like(self.mag).to(self.device)

  def update(self, inputs):
    batch = inputs.reshape((-1,) + self._shape)
    mag = torch.abs(batch).mean(0) 
    self.mag.data = self._momentum * self.mag.data + (1 - self._momentum) * mag

  def transform(self, inputs):
    values = inputs.reshape((-1,) + self._shape)
    values /= self.mag[None] + self._eps 
    values *= self._scale
    return values.reshape(inputs.shape)


class RequiresGrad:

  def __init__(self, model):
    self._model = model

  def __enter__(self):
    self._model.requires_grad_(requires_grad=True)

  def __exit__(self, *args):
    self._model.requires_grad_(requires_grad=False)