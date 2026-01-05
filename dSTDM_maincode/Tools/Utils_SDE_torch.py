# -*- coding: utf-8 -*-
"""
Utils code for SDE - pytorch

Created on 2024/06/04

@author: Zi Wang

Modified from 2020 The Google Research.

If you want to use this code, please cite following paper:
Zi Wang et al., Robust cardiac cine MRI reconstruction with spatiotemporal diffusion model, IEEE Transactions on Computational Imaging, 11: 1258-1270, 2025.

Email: Dr. Zi Wang (zi.wang@imperial.ac.uk)
GitHub: https://github.com/wangziblake/STDM
"""

import torch
import numpy as np
import ml_collections
from pathlib import Path
from Tools import SDE_Lib as sde_lib

_MODELS = {}


def get_default_configs():
    config = ml_collections.ConfigDict()

    # Data
    config.data = data = ml_collections.ConfigDict()
    data.centered = False  # True: Input range is [-1, 1], False: Input range is [0, 1]

    # Training
    config.training = training = ml_collections.ConfigDict()
    training.likelihood_weighting = False
    training.continuous = True
    training.reduce_mean = True

    # Sampling
    config.sampling = sampling = ml_collections.ConfigDict()
    sampling.n_steps_each = 1  # Only used in corrector sampler
    sampling.noise_removal = True
    sampling.probability_flow = False
    sampling.snr = 0.16  # [0,1] Higher values mean more noise, lower values mean more undersampling artifacts, only used in corrector sampler

    # Model
    config.model = model = ml_collections.ConfigDict()
    model.num_scales = 1000
    # sigma only used in VE-SDE
    model.sigma_max = 70  # Following the maximum Euclidean Distance between all pairs of training data
    model.sigma_min = 0.01
    # beta only used in VP-SDE
    model.beta_min = 0.1  # 0.1 = 0.0001*1000, see in DDPM paper
    model.beta_max = 20.  # 20 = 0.02*1000, see in DDPM paper
    model.dropout = 0.

    # Optimization
    config.optim = optim = ml_collections.ConfigDict()
    optim.weight_decay = 0
    optim.optimizer = 'Adam'
    optim.lr = 2e-4
    optim.beta1 = 0.9
    optim.eps = 1e-8
    optim.warmup = 5000
    optim.grad_clip = 1.

    return config


def restore_checkpoint(ckpt_dir, state, device, skip_sigma=False, skip_optimizer=False):
    ckpt_dir = Path(ckpt_dir)
    # ckpt = ckpt_dir / "checkpoint.pth"
    loaded_state = torch.load(ckpt_dir, map_location=device)
    if not skip_optimizer:
        state['optimizer'].load_state_dict(loaded_state['optimizer'])
    # if skip_sigma:
    #     loaded_model_state.pop('module.sigmas')

    state['model'].load_state_dict(loaded_state['model'], strict=False)
    state['ema'].load_state_dict(loaded_state['ema'])
    state['step'] = loaded_state['step']
    print(f'loaded checkpoint dir from {ckpt_dir}')
    return state


def save_checkpoint(state, name="checkpoint.pth"):
    saved_state = {
        'optimizer': state['optimizer'].state_dict(),
        'model': state['model'].state_dict(),
        'ema': state['ema'].state_dict(),
        'step': state['step']
    }
    torch.save(saved_state, name)


def get_data_scaler(config):
    """Data normalizer. Assume data are always in [0, 1]."""
    if config.data.centered:
        # Rescale to [-1, 1]
        return lambda x: x * 2.0 - 1.0
    else:
        return lambda x: x


def get_data_inverse_scaler(config):
    """Inverse data normalizer."""
    if config.data.centered:
        # Rescale [-1, 1] to [0, 1]
        return lambda x: (x + 1.0) / 2.0
    else:
        return lambda x: x


def register_model(cls=None, *, name=None):
    """A decorator for registering model classes."""

    def _register(cls):
        if name is None:
            local_name = cls.__name__
        else:
            local_name = name
        if local_name in _MODELS:
            raise ValueError(f'Already registered model with name: {local_name}')
        _MODELS[local_name] = cls
        return cls

    if cls is None:
        return _register
    else:
        return _register(cls)


def get_model(name):
    return _MODELS[name]


def get_sigmas(config):
    """Get sigmas --- the set of noise levels for SMLD from config files.
  Args:
    config: A ConfigDict object parsed from the config file
  Returns:
    sigmas: a jax numpy arrary of noise levels
  """
    sigmas = np.exp(
        np.linspace(np.log(config.model.sigma_max), np.log(config.model.sigma_min), config.model.num_scales))

    return sigmas


def get_ddpm_params(config):
    """Get betas and alphas --- parameters used in the original DDPM paper."""
    num_diffusion_timesteps = 1000
    # parameters need to be adapted if number of time steps differs from 1000
    beta_start = config.model.beta_min / config.model.num_scales
    beta_end = config.model.beta_max / config.model.num_scales
    betas = np.linspace(beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64)

    alphas = 1. - betas
    alphas_cumprod = np.cumprod(alphas, axis=0)
    sqrt_alphas_cumprod = np.sqrt(alphas_cumprod)
    sqrt_1m_alphas_cumprod = np.sqrt(1. - alphas_cumprod)

    return {
        'betas': betas,
        'alphas': alphas,
        'alphas_cumprod': alphas_cumprod,
        'sqrt_alphas_cumprod': sqrt_alphas_cumprod,
        'sqrt_1m_alphas_cumprod': sqrt_1m_alphas_cumprod,
        'beta_min': beta_start * (num_diffusion_timesteps - 1),
        'beta_max': beta_end * (num_diffusion_timesteps - 1),
        'num_diffusion_timesteps': num_diffusion_timesteps
    }


def create_model(config):
    """Create the score model."""
    model_name = config.model.name
    score_model = get_model(model_name)(config)
    return score_model


def get_model_fn(model, train=False):
    """Create a function to give the output of the score-based model.

  Args:
    model: The score model.
    train: `True` for training and `False` for evaluation.

  Returns:
    A model function.
  """

    def model_fn(x, labels):
        """Compute the output of the score-based model.

    Args:
      x: A mini-batch of input data.
      labels: A mini-batch of conditioning variables for time steps. Should be interpreted differently
        for different models.

    Returns:
      A tuple of (model output, new mutable states)
    """
        if not train:
            model.eval()
            return model(x, labels)
        else:
            model.train()
            return model(x, labels)

    return model_fn


def get_score_fn(sde, model, train=False, continuous=False):
    """Wraps `score_fn` so that the model output corresponds to a real time-dependent score function.

  Args:
    sde: An `sde_lib.SDE` object that represents the forward SDE.
    model: A score model.
    train: `True` for training and `False` for evaluation.
    continuous: If `True`, the score-based model is expected to directly take continuous time steps.

  Returns:
    A score function.
  """
    model_fn = get_model_fn(model, train=train)

    if isinstance(sde, sde_lib.VPSDE) or isinstance(sde, sde_lib.subVPSDE) or isinstance(sde, sde_lib.cosVPSDE):
        def score_fn(x, t):
            # Scale neural network output by standard deviation and flip sign
            if continuous or isinstance(sde, sde_lib.subVPSDE):
                # For VP-trained models, t=0 corresponds to the lowest noise level
                # The maximum value of time embedding is assumed to 999 for
                # continuously-trained models.
                labels = t * 999
                score = model_fn(x, labels)
                std = sde.marginal_prob(torch.zeros_like(x), t)[1]
            else:
                # For VP-trained models, t=0 corresponds to the lowest noise level
                labels = t * (sde.N - 1)
                score = model_fn(x, labels)
                std = sde.sqrt_1m_alphas_cumprod.to(labels.device)[labels.long()]

            score = - score / std[:, None, None, None]
            return score

    elif isinstance(sde, sde_lib.VESDE) or isinstance(sde, sde_lib.csmVESDE) or isinstance(sde, sde_lib.ddsVESDE):
        def score_fn(x, t):
            if continuous:
                labels = sde.marginal_prob(torch.zeros_like(x), t)[1]
            else:
                # For VE-trained models, t=0 corresponds to the highest noise level
                labels = sde.T - t
                labels *= sde.N - 1
                labels = torch.round(labels).long()

            score = model_fn(x, labels)
            return score

    else:
        raise NotImplementedError(f"SDE class {sde.__class__.__name__} not yet supported.")

    return score_fn


def to_flattened_numpy(x):
    """Flatten a torch tensor `x` and convert it to numpy."""
    return x.detach().cpu().numpy().reshape((-1,))


def from_flattened_numpy(x, shape):
    """Form a torch tensor with the given `shape` from a flattened numpy array `x`."""
    return torch.from_numpy(x.reshape(shape))


class lambda_schedule_linear:
  def __init__(self, num_total, start_lamb=0.0, end_lamb=10.0):
    super().__init__()
    self.start_lamb = start_lamb
    self.end_lamb = end_lamb
    self.total = num_total

  def get_current_lambda(self, i):
    return self.start_lamb + (self.end_lamb - self.start_lamb) * (i / self.total)
