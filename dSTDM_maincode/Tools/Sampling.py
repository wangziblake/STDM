# -*- coding: utf-8 -*-
"""
Various sampling methods code for SDE - pytorch

Created on 2024/05/26

@author: Zi Wang

Modified from 2020 The Google Research.

If you want to use this code, please cite following paper:
Zi Wang et al., Robust cardiac cine MRI reconstruction with spatiotemporal diffusion model, IEEE Transactions on Computational Imaging, 11: 1258-1270, 2025.

Email: Dr. Zi Wang (zi.wang@imperial.ac.uk)
GitHub: https://github.com/wangziblake/STDM
"""

import functools
import time
import torch
import numpy as np
import abc
from scipy import integrate
from Tools.Utils_SDE_torch import from_flattened_numpy, to_flattened_numpy, get_score_fn
from Tools import SDE_Lib as sde_lib
from Tools import Utils_SDE_torch as mutils
from Tools.Tools_torch import *
from tqdm import tqdm

_CORRECTORS = {}
_PREDICTORS = {}


def register_predictor(cls=None, *, name=None):
    """A decorator for registering predictor classes."""

    def _register(cls):
        if name is None:
            local_name = cls.__name__
        else:
            local_name = name
        if local_name in _PREDICTORS:
            raise ValueError(f'Already registered model with name: {local_name}')
        _PREDICTORS[local_name] = cls
        return cls

    if cls is None:
        return _register
    else:
        return _register(cls)


def register_corrector(cls=None, *, name=None):
    """A decorator for registering corrector classes."""

    def _register(cls):
        if name is None:
            local_name = cls.__name__
        else:
            local_name = name
        if local_name in _CORRECTORS:
            raise ValueError(f'Already registered model with name: {local_name}')
        _CORRECTORS[local_name] = cls
        return cls

    if cls is None:
        return _register
    else:
        return _register(cls)


def get_predictor(name):
    return _PREDICTORS[name]


def get_corrector(name):
    return _CORRECTORS[name]


def get_sampling_fn_withDC(config, sde, eps):
    """Create a sampling function - with data consistency.

  Args:
    config: A `ml_collections.ConfigDict` object that contains all configuration information.
    sde: A `sde_lib.SDE` object that represents the forward SDE.
    eps: A `float` number. The reverse-time SDE is only integrated to `eps` for numerical stability.

  Returns:
    A function that takes random states and a replicated training state and outputs samples with the
      trailing dimensions matching `data.shape`.
  """

    sampler_name = config.sampling.method
    if sampler_name.lower() == 'reverse':
        predictor = get_predictor(config.sampling.predictor.lower())
        sampling_fn = get_reverse_sampler_withDC(config=config,
                                                sde=sde,
                                                predictor=predictor,
                                                probability_flow=config.sampling.probability_flow,
                                                continuous=config.training.continuous,
                                                denoise=config.sampling.noise_removal,
                                                eps=eps,
                                                device=config.device)
    elif sampler_name.lower() == 'reverse_sdual':
        predictor = get_predictor(config.sampling.predictor.lower())
        sampling_fn = get_reverse_sampler_withDC_sdual(config=config,
                                                sde=sde,
                                                predictor=predictor,
                                                probability_flow=config.sampling.probability_flow,
                                                continuous=config.training.continuous,
                                                denoise=config.sampling.noise_removal,
                                                eps=eps,
                                                device=config.device)
    elif sampler_name.lower() == 'reverse_pdual':
        predictor = get_predictor(config.sampling.predictor.lower())
        sampling_fn = get_reverse_sampler_withDC_pdual(config=config,
                                                sde=sde,
                                                predictor=predictor,
                                                probability_flow=config.sampling.probability_flow,
                                                continuous=config.training.continuous,
                                                denoise=config.sampling.noise_removal,
                                                eps=eps,
                                                device=config.device)
    else:
        raise ValueError(f"Sampler name {sampler_name} unknown.")

    return sampling_fn


class Predictor(abc.ABC):
    """The abstract class for a predictor algorithm."""

    def __init__(self, sde, score_fn, probability_flow=False):
        super().__init__()
        self.sde = sde
        # Compute the reverse SDE/ODE
        self.rsde = sde.reverse(score_fn, probability_flow)
        self.score_fn = score_fn

    @abc.abstractmethod
    def update_fn(self, x, t, k_inputs, mask, CSM, CSM_conj, tfactor_norm):
        """One update of the predictor.

    Args:
      x: A PyTorch tensor representing the current state
      t: A Pytorch tensor representing the current time step.

    Returns:
      x: A PyTorch tensor of the next state.
      x_mean: A PyTorch tensor. The next state without random noise. Useful for denoising.
    """
        pass


class Corrector(abc.ABC):
    """The abstract class for a corrector algorithm."""

    def __init__(self, sde, score_fn, snr, n_steps):
        super().__init__()
        self.sde = sde
        self.score_fn = score_fn
        self.snr = snr
        self.n_steps = n_steps

    @abc.abstractmethod
    def update_fn(self, x, t):
        """One update of the corrector.

    Args:
      x: A PyTorch tensor representing the current state
      t: A PyTorch tensor representing the current time step.

    Returns:
      x: A PyTorch tensor of the next state.
      x_mean: A PyTorch tensor. The next state without random noise. Useful for denoising.
    """
        pass


@register_predictor(name='euler_maruyama')
class EulerMaruyamaPredictor(Predictor):
    def __init__(self, sde, score_fn, probability_flow=False):
        super().__init__(sde, score_fn, probability_flow)

    def update_fn(self, x, t, k_inputs, mask, CSM, CSM_conj, tfactor_norm):
        dt = -1. / self.rsde.N
        z = torch.randn_like(x)
        drift, diffusion = self.rsde.sde(x, t)
        x_mean = x + drift * dt
        x = x_mean + diffusion[:, None, None, None] * np.sqrt(-dt) * z
        return x, x_mean


@register_predictor(name='reverse_diffusion')
class ReverseDiffusionPredictor(Predictor):
    def __init__(self, sde, score_fn, probability_flow=False):
        super().__init__(sde, score_fn, probability_flow)

    def update_fn(self, x, t, k_inputs, mask, CSM, CSM_conj, tfactor_norm):
        if isinstance(self.sde, sde_lib.csmVESDE):
            f, G, score = self.rsde.discretize_csm(x, t, CSM, CSM_conj)
            z = torch.randn_like(x)
            z_csm = torch_complex2double(torch.sum((torch_double2complex(z) * CSM) * CSM_conj, dim=1, keepdim=True))
            x_mean = x - f
            x = x_mean + G[:, None, None, None] * z_csm
        else:
            f, G, score = self.rsde.discretize(x, t)
            z = torch.randn_like(x)
            x_mean = x - f
            x = x_mean + G[:, None, None, None] * z
        return x, x_mean, score


@register_predictor(name='none')
class NonePredictor(Predictor):
    """An empty predictor that does nothing."""

    def __init__(self, sde, score_fn, probability_flow=False):
        pass

    def update_fn(self, x, t):
        return x, x


@register_corrector(name='langevin')
class LangevinCorrector(Corrector):
    def __init__(self, sde, score_fn, snr, n_steps):
        super().__init__(sde, score_fn, snr, n_steps)
        if not isinstance(sde, sde_lib.VPSDE) \
                and not isinstance(sde, sde_lib.VESDE) \
                and not isinstance(sde, sde_lib.subVPSDE):
            raise NotImplementedError(f"SDE class {sde.__class__.__name__} not yet supported.")

    def update_fn(self, x, t):
        sde = self.sde
        score_fn = self.score_fn
        n_steps = self.n_steps
        target_snr = self.snr
        if isinstance(sde, sde_lib.VPSDE) or isinstance(sde, sde_lib.subVPSDE):
            timestep = (t * (sde.N - 1) / sde.T).long()
            alpha = sde.alphas.to(t.device)[timestep]
        else:
            alpha = torch.ones_like(t)

        for i in range(n_steps):
            grad = score_fn(x, t)
            noise = torch.randn_like(x)
            grad_norm = torch.norm(grad.reshape(grad.shape[0], -1), dim=-1).mean()
            noise_norm = torch.norm(noise.reshape(noise.shape[0], -1), dim=-1).mean()
            step_size = (target_snr * noise_norm / grad_norm) ** 2 * 2 * alpha
            x_mean = x + step_size[:, None, None, None] * grad
            x = x_mean + torch.sqrt(step_size * 2)[:, None, None, None] * noise

        return x, x_mean


@register_corrector(name='none')
class NoneCorrector(Corrector):
    """An empty corrector that does nothing."""

    def __init__(self, sde, score_fn, snr, n_steps):
        pass

    def update_fn(self, x, t):
        return x, x


def shared_predictor_update_fn(x, t, k_inputs, mask, CSM, CSM_conj, tfactor_norm, sde, model, predictor, probability_flow, continuous):
    """A wrapper that configures and returns the update function of predictors."""
    score_fn = mutils.get_score_fn(sde, model, train=False, continuous=continuous)
    if predictor is None:
        # Corrector-only sampler
        predictor_obj = NonePredictor(sde, score_fn, probability_flow)
    else:
        predictor_obj = predictor(sde, score_fn, probability_flow)
    return predictor_obj.update_fn(x, t, k_inputs, mask, CSM, CSM_conj, tfactor_norm)


def shared_corrector_update_fn(x, t, sde, model, corrector, continuous, snr, n_steps):
    """A wrapper tha configures and returns the update function of correctors."""
    score_fn = mutils.get_score_fn(sde, model, train=False, continuous=continuous)
    if corrector is None:
        # Predictor-only sampler
        corrector_obj = NoneCorrector(sde, score_fn, snr, n_steps)
        fn = corrector_obj.update_fn(x, t)
    else:
        corrector_obj = corrector(sde, score_fn, snr, n_steps)
        fn = corrector_obj.update_fn(x, t)

    return fn


def get_reverse_sampler_withDC(config, sde, predictor, probability_flow=False, continuous=False, denoise=True, eps=1e-3, device='cuda'):
    """Create a reverse diffusion sampler - 2DSpatiotemporal with data consistency.

  Args:
    sde: An `sde_lib.SDE` object representing the forward SDE.
    predictor: A subclass of `sampling.Predictor` representing the predictor algorithm.
    probability_flow: If `True`, solve the reverse-time probability flow ODE when running the predictor.
    continuous: `True` indicates that the score model was continuously trained.
    denoise: If `True`, add one-step denoising to the final samples.
    eps: A `float` number. The reverse-time SDE and ODE are integrated to `epsilon` to avoid numerical issues.
    device: PyTorch device.

  Returns:
    A sampling function that returns samples and the number of function evaluations during sampling.
  """
    # Create predictor update functions
    predictor_update_fn = functools.partial(shared_predictor_update_fn,
                                            sde=sde,
                                            predictor=predictor,
                                            probability_flow=probability_flow,
                                            continuous=continuous)

    def reverse_sampler_withDC(model, cc_inputs, k_inputs, mask, CSM, CSM_conj, tfactor_norm, lamb_schedule, initial_conditional):
        """ The reverse diffusion sampler funciton - with data consistency.

    Args:
      model: A score model.
      cc_inputs: Undersampled coil-combined image
      k_inputs: Undersampled multi-coil k-space
      mask: Undersampling mask
      CSM: Coil sensitivity map
      CSM_conj: Conjugation of coil sensitivity map
      lamb_schedule: Regularization parameter from zero to infinite, default: 0.0 (trust all sampled data)
      initial_conditional: Sample initialization scheme
    Returns:
      Samples, number of function evaluations.
    """
        with torch.no_grad():
            # Initial sample
            cc_inputs_real = torch_complex2double(cc_inputs)  # [batchsize=nx, 2*ncoil, nframe, ny]

            if initial_conditional == 'ZeroFilled':
                z = sde.prior_sampling(cc_inputs_real.shape).to(device)
                x = torch_get_mask_conditional(z, k_inputs, mask, CSM, CSM_conj)
                x = x
            elif initial_conditional == 'NNRecon':
                pass
            elif initial_conditional == 'RandomNoise':
                x = sde.prior_sampling(cc_inputs_real.shape).to(device)
            else:
                raise NotImplementedError(f"Given initial {initial_conditional} not implemented yet!")

            timesteps = torch.linspace(sde.T, eps, sde.N, device=device)

            for i in range(sde.N):
                t = timesteps[i]
                vec_t = torch.ones(cc_inputs_real.shape[0], device=t.device) * t

                # Predictor
                x, x_mean, _ = predictor_update_fn(x, vec_t, k_inputs, mask, CSM, CSM_conj, tfactor_norm, model=model)
                # Data_consistency_CSM
                lamb_dc = lamb_schedule.get_current_lambda(i)
                x = torch_dc_csm_SDE(x, k_inputs, mask, CSM, CSM_conj, lamb_dc)
                x = x
                if i == (sde.N - 1):
                    x_mean = torch_dc_csm_SDE(x_mean , k_inputs, mask, CSM, CSM_conj, lamb_dc)
                    x_mean = x_mean

            return x_mean if denoise else x  # [batchsize=nx, 2*ncoil, nframe, ny]

    return reverse_sampler_withDC


def get_reverse_sampler_withDC_sdual(config, sde, predictor, probability_flow=False, continuous=False, denoise=True, eps=1e-3, device='cuda'):
    """Create a reverse diffusion sampler - 2DSpatiotemporal with data consistency - dual-direction (sequential xt+yt).

  Args:
    sde: An `sde_lib.SDE` object representing the forward SDE.
    predictor: A subclass of `sampling.Predictor` representing the predictor algorithm.
    probability_flow: If `True`, solve the reverse-time probability flow ODE when running the predictor.
    continuous: `True` indicates that the score model was continuously trained.
    denoise: If `True`, add one-step denoising to the final samples.
    eps: A `float` number. The reverse-time SDE and ODE are integrated to `epsilon` to avoid numerical issues.
    device: PyTorch device.

  Returns:
    A sampling function that returns samples and the number of function evaluations during sampling.
  """
    # Create predictor update functions
    predictor_update_fn = functools.partial(shared_predictor_update_fn,
                                            sde=sde,
                                            predictor=predictor,
                                            probability_flow=probability_flow,
                                            continuous=continuous)

    def reverse_sampler_withDC_sdual(model, cc_inputs, k_inputs, mask, CSM, CSM_conj, tfactor_norm, lamb_schedule, initial_conditional):
        """ The reverse diffusion sampler funciton - with data consistency.

    Args:
      model: A score model.
      cc_inputs: Undersampled coil-combined image
      k_inputs: Undersampled multi-coil k-space
      mask: Undersampling mask
      CSM: Coil sensitivity map
      CSM_conj: Conjugation of coil sensitivity map
      lamb_schedule: Regularization parameter from zero to infinite, default: 0.0 (trust all sampled data)
      initial_conditional: Sample initialization scheme
    Returns:
      Samples, number of function evaluations.
    """
        with torch.no_grad():
            # Initial sample
            cc_inputs_real = torch_complex2double(cc_inputs)  # [batchsize=nx, 2*ncoil, nframe, ny]
            k_inputsT = k_inputs.permute(3, 1, 2, 0)
            maskT = mask.permute(3, 1, 2, 0)
            CSMT = CSM.permute(3, 1, 2, 0)
            CSM_conjT = CSM_conj.permute(3, 1, 2, 0)

            if initial_conditional == 'ZeroFilled':
                z = sde.prior_sampling(cc_inputs_real.shape).to(device)
                x = torch_get_mask_conditional(z, k_inputs, mask, CSM, CSM_conj)
                x = x
            elif initial_conditional == 'NNRecon':
                pass
            elif initial_conditional == 'RandomNoise':
                x = sde.prior_sampling(cc_inputs_real.shape).to(device)
            else:
                raise NotImplementedError(f"Given initial {initial_conditional} not implemented yet!")

            timesteps = torch.linspace(sde.T, eps, sde.N, device=device)

            for i in range(sde.N):
                t = timesteps[i]

                # Predictor - yt
                vec_t = torch.ones(cc_inputs_real.shape[0], device=t.device) * t
                x, x_mean, _ = predictor_update_fn(x, vec_t, k_inputs, mask, CSM, CSM_conj, tfactor_norm, model=model)
                # Predictor - xt
                x = x.permute(3, 1, 2, 0)  # [nx, ncoil, nframe, ny] --> [ny, ncoil, nframe, nx]
                vec_tT = torch.ones(x.shape[0], device=t.device) * t
                x, x_mean, _ = predictor_update_fn(x, vec_tT, k_inputsT, maskT, CSMT, CSM_conjT, tfactor_norm, model=model)
                x = x.permute(3, 1, 2, 0)  # [ny, ncoil, nframe, nx] --> [nx, ncoil, nframe, ny]
                x_mean = x_mean.permute(3, 1, 2, 0)

                # Data_consistency_CSM
                lamb_dc = lamb_schedule.get_current_lambda(i)
                x = torch_dc_csm_SDE(x, k_inputs, mask, CSM, CSM_conj, lamb_dc)
                x = x
                if i == (sde.N - 1):
                    x_mean = torch_dc_csm_SDE(x_mean, k_inputs, mask, CSM, CSM_conj, lamb_dc)
                    x_mean = x_mean

            return x_mean if denoise else x  # [batchsize=nx, 2*ncoil, nframe, ny]

    return reverse_sampler_withDC_sdual


def get_reverse_sampler_withDC_pdual(config, sde, predictor, probability_flow=False, continuous=False, denoise=True, eps=1e-3, device='cuda'):
    """Create a reverse diffusion sampler - 2DSpatiotemporal with data consistency - dual-direction (parallel xt+yt).

  Args:
    sde: An `sde_lib.SDE` object representing the forward SDE.
    predictor: A subclass of `sampling.Predictor` representing the predictor algorithm.
    probability_flow: If `True`, solve the reverse-time probability flow ODE when running the predictor.
    continuous: `True` indicates that the score model was continuously trained.
    denoise: If `True`, add one-step denoising to the final samples.
    eps: A `float` number. The reverse-time SDE and ODE are integrated to `epsilon` to avoid numerical issues.
    device: PyTorch device.

  Returns:
    A sampling function that returns samples and the number of function evaluations during sampling.
  """
    # Create predictor update functions
    predictor_update_fn = functools.partial(shared_predictor_update_fn,
                                            sde=sde,
                                            predictor=predictor,
                                            probability_flow=probability_flow,
                                            continuous=continuous)

    def reverse_sampler_withDC_pdual(model, cc_inputs, k_inputs, mask, CSM, CSM_conj, tfactor_norm, lamb_schedule, initial_conditional):
        """ The reverse diffusion sampler funciton - with data consistency.

    Args:
      model: A score model.
      cc_inputs: Undersampled coil-combined image
      k_inputs: Undersampled multi-coil k-space
      mask: Undersampling mask
      CSM: Coil sensitivity map
      CSM_conj: Conjugation of coil sensitivity map
      lamb_schedule: Regularization parameter from zero to infinite, default: 0.0 (trust all sampled data)
      initial_conditional: Sample initialization scheme
    Returns:
      Samples, number of function evaluations.
    """
        with torch.no_grad():
            # Initial sample
            cc_inputs_real = torch_complex2double(cc_inputs)  # [batchsize=nx, 2*ncoil, nframe, ny]
            k_inputsT = k_inputs.permute(3, 1, 2, 0)
            maskT = mask.permute(3, 1, 2, 0)
            CSMT = CSM.permute(3, 1, 2, 0)
            CSM_conjT = CSM_conj.permute(3, 1, 2, 0)

            if initial_conditional == 'ZeroFilled':
                z = sde.prior_sampling(cc_inputs_real.shape).to(device)
                x = torch_get_mask_conditional(z, k_inputs, mask, CSM, CSM_conj)
                x = x
            elif initial_conditional == 'NNRecon':
                pass
            elif initial_conditional == 'RandomNoise':
                x = sde.prior_sampling(cc_inputs_real.shape).to(device)
            else:
                raise NotImplementedError(f"Given initial {initial_conditional} not implemented yet!")

            if config.sampling.fastsampling:
                sde.N = config.sampling.N

            timesteps = torch.linspace(sde.T, eps, sde.N, device=device)

            for i in range(sde.N):
                t = timesteps[i]

                # Predictor - yt
                vec_t = torch.ones(cc_inputs_real.shape[0], device=t.device) * t
                x_yt, x_yt_mean, _ = predictor_update_fn(x, vec_t, k_inputs, mask, CSM, CSM_conj, tfactor_norm, model=model)
                # Predictor - xt
                x = x.permute(3, 1, 2, 0)  # [nx, ncoil, nframe, ny] --> [ny, ncoil, nframe, nx]
                vec_tT = torch.ones(x.shape[0], device=t.device) * t
                x_xt, x_xt_mean, _ = predictor_update_fn(x, vec_tT, k_inputsT, maskT, CSMT, CSM_conjT, tfactor_norm, model=model)
                x_xt = x_xt.permute(3, 1, 2, 0)  # [ny, ncoil, nframe, nx] --> [nx, ncoil, nframe, ny]
                x_xt_mean = x_xt_mean.permute(3, 1, 2, 0)

                # Data_consistency_CSM
                lamb_dc = lamb_schedule.get_current_lambda(i)

                x = (x_yt + x_xt) / 2.0
                x = torch_dc_csm_SDE(x, k_inputs, mask, CSM, CSM_conj, lamb_dc)
                x = x
                if i == (sde.N - 1):
                    x_mean = (x_yt_mean + x_xt_mean) / 2.0
                    x_mean = torch_dc_csm_SDE(x_mean, k_inputs, mask, CSM, CSM_conj, lamb_dc)
                    x_mean = x_mean

            return x_mean if denoise else x  # [batchsize=nx, 2*ncoil, nframe, ny]

    return reverse_sampler_withDC_pdual
