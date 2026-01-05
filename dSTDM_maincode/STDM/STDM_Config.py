# -*- coding: utf-8 -*-
"""
Training and Testing Configuration Code for dSTDM (Dual-directional SpatioTemporal Diffusion Model for Dynamic MRI) - torch

Created on 2024/05/09

@author: Zi Wang

If you want to use this code, please cite following paper:
Zi Wang et al., Robust cardiac cine MRI reconstruction with spatiotemporal diffusion model, IEEE Transactions on Computational Imaging, 11: 1258-1270, 2025.

Email: Dr. Zi Wang (zi.wang@imperial.ac.uk)
GitHub: https://github.com/wangziblake/STDM
"""


from Tools import Utils_SDE_torch as mutils


def train_config_stdm():
    config = mutils.get_default_configs()
    # Data config
    config.data.image_size = 192
    config.data.num_channels = 2  # Coil-combined image (real and imaginary part)
    config.data.norm = 'imagemax'  # imagemax
    # Training config
    config.training.sde = 'VESDE'  # VESDE
    config.training.continuous = True
    # Model config
    config.model.name = 'ddpmv22'  # TODO: ncsnpp
    config.model.num_scales = 100  # default: 100
    config.model.sigma_max = 50  # Following the maximum Euclidean Distance between all pairs of training data
    config.model.scale_by_sigma = False  # SDE loss: False, NCSN loss: True, DDPM loss: False
    config.model.ema_rate = 0.999  # VE: 0.999, VP:0.9999
    config.model.normalization = 'GroupNorm'
    config.model.nonlinearity = 'swish'
    config.model.nf = 128  # convfilter number in the frist level of UNet
    config.model.ch_mult = (1, 2, 2)  # 3 level of UNet: 128, 256, 256, since image is 192*12 and only supports 2 downsampling
    config.model.num_res_blocks = 2
    config.model.attn_resolutions = (16,)
    config.model.resamp_with_conv = True
    config.model.conditional = True
    config.model.conv_size = 3
    return config
