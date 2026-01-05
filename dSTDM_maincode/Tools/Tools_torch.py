# -*- coding: utf-8 -*-
"""
Tools code - pytorch

Created on 2024/05/26

@author: Zi Wang

If you want to use this code, please cite following paper:
Zi Wang et al., Robust cardiac cine MRI reconstruction with spatiotemporal diffusion model, IEEE Transactions on Computational Imaging, 11: 1258-1270, 2025.

Email: Dr. Zi Wang (zi.wang@imperial.ac.uk)
GitHub: https://github.com/wangziblake/STDM
"""

import torch
import numpy as np


def np_fft2c(x):

    # x = [nslice, kx, ky, ncoil]
    x = np.transpose(x, (0, 3, 1, 2))  # [nslice, ncoil, kx, ky]
    _, _, kx, ky = np.float32(x.shape)
    kxky = np.complex64(kx * ky + 0j)
    axes = (-2, -1)
    x = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(x, axes=axes)), axes=axes) / np.sqrt(kxky)
    x = np.transpose(x, (0, 2, 3, 1))  # [nslice, kx, ky, ncoil]
    return x


def np_ifft2c(x):

    # x = [nslice, kx, ky, ncoil]
    x = np.transpose(x, (0, 3, 1, 2))  # [nslice, ncoil, kx, ky]
    _, _, kx, ky = np.float32(x.shape)
    kxky = np.complex64(kx * ky + 0j)
    axes = (-2, -1)
    x = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(x, axes=axes)), axes=axes) * np.sqrt(kxky)
    x = np.transpose(x, (0, 2, 3, 1))  # [nslice, kx, ky, ncoil]
    return x


def np_fft2c_DMRI(x, dim1, dim2):
    # x = [ncoil, nframe, kx, ky]
    fctr = x.shape[dim1] * x.shape[dim2]
    x = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(x, axes=(dim1, dim2)), axes=(dim1, dim2)), axes=(dim1, dim2)) / np.sqrt(fctr)
    return x


def np_ifft2c_DMRI(x, dim1, dim2):
    # x = [ncoil, nframe, kx, ky]
    fctr = x.shape[dim1] * x.shape[dim2]
    x = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(x, axes=(dim1, dim2)), axes=(dim1, dim2)), axes=(dim1, dim2)) * np.sqrt(fctr)
    return x


def np_ifft2c_5d(x):   # [kx, ky, ncoil, ncoil, nslice]
    fctr = x.shape[0] * x.shape[1]
    result = np.sqrt(fctr) * np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(x, axes=(0, 1)), axes=(0, 1)), axes=(0, 1))
    return result


def torch_fft2c(x):
    # x = [nslice, ncoil, kx, ky]

    kx = int(x.shape[-2])
    ky = int(x.shape[-1])

    dim = (-2, -1)
    x = torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(x, dim=dim)), dim=dim) / np.sqrt(kx*ky)
    return x  # [nslice, ncoil, kx, ky]


def torch_ifft2c(x):
    # x = [nslice, ncoil, kx, ky]

    kx = int(x.shape[-2])
    ky = int(x.shape[-1])

    dim = (-2, -1)
    x = torch.fft.fftshift(torch.fft.ifft2(torch.fft.ifftshift(x, dim=dim)), dim=dim) * np.sqrt(kx*ky)
    return x  # [nslice, ncoil, kx, ky]


def np_fft1c(x):
    # x = [nslice, ky, ncoil]
    x = np.transpose(x, (0, 2, 1))  # [nslice, ncoil, ky]
    _, _, ky = np.float32(x.shape)
    ky = np.complex64(ky + 0j)
    axes = -1
    x = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(x, axes=axes)), axes=axes) / np.sqrt(ky)
    x = np.transpose(x, (0, 2, 1))  # [nslice, ky, ncoil]
    return x


def np_ifft1c(x):
    # x = [nslice, ky, ncoil]
    x = np.transpose(x, (0, 2, 1))  # [nslice, ncoil, ky]
    _, _, ky = np.float32(x.shape)
    ky = np.complex64(ky + 0j)
    axes = -1
    x = np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(x, axes=axes)), axes=axes) * np.sqrt(ky)
    x = np.transpose(x, (0, 2, 1))  # [nslice, ky, ncoil]
    return x


def torch_fft1c(x):
    # x = [nslice, ncoil, ky]

    ky = int(x.shape[-1])

    dim = -1
    x = torch.fft.fftshift(torch.fft.fft(torch.fft.ifftshift(x, dim=dim)), dim=dim) / np.sqrt(ky)
    return x  # [nslice, ncoil, ky]


def torch_ifft1c(x):
    # x = [nslice, ncoil, ky]

    ky = int(x.shape[-1])

    dim = -1
    x = torch.fft.fftshift(torch.fft.ifft(torch.fft.ifftshift(x, dim=dim)), dim=dim) * np.sqrt(ky)
    return x  # [nslice, ncoil, ky]


def np_fft1c_hybrid(x, dim):
    # x = [nslice, kx, ky, ncoil]
    if dim == 1:
        x = np.transpose(x, (0, 2, 3, 1))  # [nslice, ky, ncoil, kx]
        _, _, _, kx = np.float32(x.shape)
        kx = np.complex64(kx + 0j)
        axes = -1
        x = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(x, axes=axes)), axes=axes) / np.sqrt(kx)
        x = np.transpose(x, (0, 3, 1, 2))  # [nslice, kx, ky, ncoil]

    if dim == 2:
        x = np.transpose(x, (0, 1, 3, 2))  # [nslice, kx, ncoil, ky]
        _, _, _, ky = np.float32(x.shape)
        ky = np.complex64(ky + 0j)
        axes = -1
        x = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(x, axes=axes)), axes=axes) / np.sqrt(ky)
        x = np.transpose(x, (0, 1, 3, 2))  # [nslice, kx, ky, ncoil]
    return x


def np_ifft1c_hybrid(x, dim):
    # x = [nslice, kx, ky, ncoil]
    if dim == 1:
        x = np.transpose(x, (0, 2, 3, 1))  # [nslice, ky, ncoil, kx]
        _, _, _, kx = np.float32(x.shape)
        kx = np.complex64(kx + 0j)
        axes = -1
        x = np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(x, axes=axes)), axes=axes) * np.sqrt(kx)
        x = np.transpose(x, (0, 3, 1, 2))  # [nslice, kx, ky, ncoil]

    if dim == 2:
        x = np.transpose(x, (0, 1, 3, 2))  # [nslice, kx, ncoil, ky]
        _, _, _, ky = np.float32(x.shape)
        ky = np.complex64(ky + 0j)
        axes = -1
        x = np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(x, axes=axes)), axes=axes) * np.sqrt(ky)
        x = np.transpose(x, (0, 1, 3, 2))  # [nslice, kx, ky, ncoil]
    return x


def np_fft1c_hybrid_DMRI(x, dim):
    # x = [nslice, kx, ky, nframe, ncoil]
    if dim == 1:
        x = np.transpose(x, (0, 2, 3, 4, 1))  # [nslice, ky, nframe, ncoil, kx]
        _, _, _, _, kx = np.float32(x.shape)
        kx = np.complex64(kx + 0j)
        axes = -1
        x = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(x, axes=axes)), axes=axes) / np.sqrt(kx)
        x = np.transpose(x, (0, 4, 1, 2, 3))  # [nslice, kx, ky, nframe, ncoil]

    if dim == 2:
        x = np.transpose(x, (0, 1, 3, 4, 2))  # [nslice, kx, nframe, coil, ky]
        _, _, _, _, ky = np.float32(x.shape)
        ky = np.complex64(ky + 0j)
        axes = -1
        x = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(x, axes=axes)), axes=axes) / np.sqrt(ky)
        x = np.transpose(x, (0, 1, 4, 2, 3))  # [nslice, kx, ky, nframe, ncoil]
    return x


def np_fft1c_hybrid_DMRI_Minibatch(x, dim):
    # x = [ncoil, nframe, kx, ky]
    if dim == 1:
        x = np.transpose(x, (0, 1, 3, 2))  # [ncoil, nframe, ky, kx]
        _, _, _, kx = np.float32(x.shape)
        kx = np.complex64(kx + 0j)
        axes = -1
        x = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(x, axes=axes)), axes=axes) / np.sqrt(kx)
        x = np.transpose(x, (0, 1, 3, 2))  # [ncoil, nframe, kx, ky]

    if dim == 2:
        _, _, _, ky = np.float32(x.shape)
        ky = np.complex64(ky + 0j)
        axes = -1
        x = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(x, axes=axes)), axes=axes) / np.sqrt(ky)
    return x


def np_ifft1c_hybrid_DMRI(x, dim):
    # x = [nslice, kx, ky, nframe, ncoil]
    if dim == 1:
        x = np.transpose(x, (0, 2, 3, 4, 1))  # [nslice, ky, nframe, ncoil, kx]
        _, _, _, _, kx = np.float32(x.shape)
        kx = np.complex64(kx + 0j)
        axes = -1
        x = np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(x, axes=axes)), axes=axes) * np.sqrt(kx)
        x = np.transpose(x, (0, 4, 1, 2, 3))  # [nslice, kx, ky, nframe, ncoil]

    if dim == 2:
        x = np.transpose(x, (0, 1, 3, 4, 2))  # [nslice, kx, nframe, ncoil, ky]
        _, _, _, _, ky = np.float32(x.shape)
        ky = np.complex64(ky + 0j)
        axes = -1
        x = np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(x, axes=axes)), axes=axes) * np.sqrt(ky)
        x = np.transpose(x, (0, 1, 4, 2, 3))  # [nslice, kx, ky, nframe, ncoil]
    return x


def np_ifft1c_hybrid_DMRI_fornorm(x, dim):
    # x = [nslice, ncoil, nframe, kx, ky]
    if dim == 1:
        x = np.transpose(x, (0, 1, 2, 4, 3))  # [nslice, ncoil, nframe, ky, kx]
        _, _, _, _, kx = np.float32(x.shape)
        kx = np.complex64(kx + 0j)
        axes = -1
        x = np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(x, axes=axes)), axes=axes) * np.sqrt(kx)
        x = np.transpose(x, (0, 1, 2, 4, 3))  # [nslice, ncoil, nframe, kx, ky]

    if dim == 2:
        _, _, _, _, ky = np.float32(x.shape)
        ky = np.complex64(ky + 0j)
        axes = -1
        x = np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(x, axes=axes)), axes=axes) * np.sqrt(ky)
    return x


def np_ifft1c_hybrid_DMRI_Minibatch(x, dim):
    # x = [ncoil, nframe, kx, ky]
    if dim == 1:
        x = np.transpose(x, (0, 1, 3, 2))  # [ncoil, nframe, ky, kx]
        _, _, _, kx = np.float32(x.shape)
        kx = np.complex64(kx + 0j)
        axes = -1
        x = np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(x, axes=axes)), axes=axes) * np.sqrt(kx)
        x = np.transpose(x, (0, 1, 3, 2))  # [ncoil, nframe, kx, ky]

    if dim == 2:
        _, _, _, ky = np.float32(x.shape)
        ky = np.complex64(ky + 0j)
        axes = -1
        x = np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(x, axes=axes)), axes=axes) * np.sqrt(ky)
    return x


def np_fft1c_DMRI(x, dim1):
    # x = [ncoil, nframe, kx, ky]
    fctr = x.shape[dim1]
    x = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(x, axes=dim1), axis=dim1), axes=dim1) / np.sqrt(fctr)
    return x


def np_ifft1c_DMRI(x, dim1):
    # x = [ncoil, nframe, kx, ky]
    fctr = x.shape[dim1]
    x = np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(x, axes=dim1), axis=dim1), axes=dim1) * np.sqrt(fctr)
    return x


def torch_fft1c_hybrid(x, dimension):
    # x = [nslice, ncoil, kx, ky]
    if dimension == 1:
        x = x.permute(0, 3, 1, 2)  # [nslice, ky, ncoil, kx]
        kx = int(x.shape[-1])
        dim = -1
        x = torch.fft.fftshift(torch.fft.fft(torch.fft.ifftshift(x, dim=dim)), dim=dim) / np.sqrt(kx)
        x = x.permute(0, 2, 3, 1)  # [nslice, ncoil, kx, ky]

    if dimension == 2:
        x = x.permute(0, 2, 1, 3)  # [nslice, kx, ncoil, ky]
        ky = int(x.shape[-1])
        dim = -1
        x = torch.fft.fftshift(torch.fft.fft(torch.fft.ifftshift(x, dim=dim)), dim=dim) / np.sqrt(ky)
        x = x.permute(0, 2, 1, 3)  # [nslice, ncoil, kx, ky]
    return x


def torch_ifft1c_hybrid(x, dimension):
    # x = [nslice, ncoil, kx, ky]
    if dimension == 1:
        x = x.permute(0, 3, 1, 2)  # [nslice, ky, ncoil, kx]
        kx = int(x.shape[-1])
        dim = -1
        x = torch.fft.fftshift(torch.fft.ifft(torch.fft.ifftshift(x, dim=dim)), dim=dim) * np.sqrt(kx)
        x = x.permute(0, 2, 3, 1)  # [nslice, ncoil, kx, ky]

    if dimension == 2:
        x = x.permute(0, 2, 1, 3)  # [nslice, kx, ncoil, ky]
        ky = int(x.shape[-1])
        dim = -1
        x = torch.fft.fftshift(torch.fft.ifft(torch.fft.ifftshift(x, dim=dim)), dim=dim) * np.sqrt(ky)
        x = x.permute(0, 2, 1, 3)  # [nslice, ncoil, kx, ky]
    return x


def torch_fft1c_hybrid_DMRI(x, dimension):
    # x = [nslice, ncoil, nframe, kx, ky]
    if dimension == 1:
        kx = int(x.shape[-2])
        dim = -2
        x = torch.fft.fftshift(torch.fft.fft(torch.fft.ifftshift(x, dim=dim), dim=dim), dim=dim) / np.sqrt(kx)

    if dimension == 2:
        ky = int(x.shape[-1])
        dim = -1
        x = torch.fft.fftshift(torch.fft.fft(torch.fft.ifftshift(x, dim=dim)), dim=dim) / np.sqrt(ky)

    if dimension == 3:
        kt = int(x.shape[-3])
        dim = -3
        x = torch.fft.fftshift(torch.fft.fft(torch.fft.ifftshift(x, dim=dim), dim=dim), dim=dim) / np.sqrt(kt)
    return x


def torch_ifft1c_hybrid_DMRI(x, dimension):
    # x = [nslice, ncoil, nframe, kx, ky]
    if dimension == 1:
        kx = int(x.shape[-2])
        dim = -2
        x = torch.fft.fftshift(torch.fft.ifft(torch.fft.ifftshift(x, dim=dim), dim=dim), dim=dim) * np.sqrt(kx)

    if dimension == 2:
        ky = int(x.shape[-1])
        dim = -1
        x = torch.fft.fftshift(torch.fft.ifft(torch.fft.ifftshift(x, dim=dim), dim=dim), dim=dim) * np.sqrt(ky)

    if dimension == 3:
        kt = int(x.shape[-3])
        dim = -3
        x = torch.fft.fftshift(torch.fft.ifft(torch.fft.ifftshift(x, dim=dim), dim=dim), dim=dim) * np.sqrt(kt)
    return x


def torch_fft1c_Fast(x, dim):
    # x = [nslice, ncoil, kx, ky]

    fctr = int(x.shape[dim])
    x = torch.fft.fftshift(torch.fft.fft(torch.fft.ifftshift(x, dim=dim), dim=dim), dim=dim) / np.sqrt(fctr)
    return x  # [nslice, ncoil, kx, ky]


def torch_ifft1c_Fast(x, dim):
    # x = [nslice, ncoil, kx, ky]

    fctr = int(x.shape[dim])
    x = torch.fft.fftshift(torch.fft.ifft(torch.fft.ifftshift(x, dim=dim), dim=dim), dim=dim) * np.sqrt(fctr)
    return x  # [nslice, ncoil, kx, ky]


def torch_fft2c_Fast(x, dim1, dim2):
    # x = [nslice, ncoil, kx, ky]

    fctr = int(x.shape[dim1] * x.shape[dim2])
    dim = (dim1, dim2)
    x = torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(x, dim=dim), dim=dim), dim=dim) / np.sqrt(fctr)
    return x  # [nslice, ncoil, kx, ky]


def torch_ifft2c_Fast(x, dim1, dim2):
    # x = [nslice, ncoil, kx, ky]

    fctr = int(x.shape[dim1] * x.shape[dim2])
    dim = (dim1, dim2)
    x = torch.fft.fftshift(torch.fft.ifft2(torch.fft.ifftshift(x, dim=dim), dim=dim), dim=dim) * np.sqrt(fctr)
    return x  # [nslice, ncoil, kx, ky]


def torch_complex2double(x):  # input x=[1-2j, 2+3j, 4+9j]
    x_real, x_imag = torch.real(x), torch.imag(x)
    # if dim=-1, torch.cat([real, imag], dim=1)=[ 1.  2.  4. -2.  3.  9.]
    x_double = torch.cat([x_real, x_imag], dim=1)
    return x_double


def torch_double2complex(x):
    nchannel = x.shape[1]
    nchannel = int(nchannel / 2)
    x_real, x_imag = x[:, :nchannel, ...], x[:, nchannel:, ...]
    x_complex = torch.complex(x_real, x_imag)  # complex
    return x_complex


def np_sos(x):
    # x = [nslice, kx, ky, ncoil] complex
    x = np.sum(np.abs(x**2), axis=-1)
    x = x**(1.0/2)
    return x  # x = [nslice, kx, ky] real


def np_sos_DMRI(x):
    # x = [nslice, kx, ky, nframe, ncoil] complex
    x = np.sum(np.abs(x**2), axis=-1)
    x = x**(1.0/2)
    return x  # x = [nslice, kx, ky, nframe] real


def torch_sos(x):
    # x = [nslice, ncoil, kx, ky] complex
    x = torch.sum(torch.abs(x**2), dim=1)
    x = x**(1.0/2)
    return x  # x = [nslice, kx, ky] real


def torch_sos_DMRI(x):
    # x = [nslice, ncoil, nframe, kx, ky] complex
    x = torch.sum(torch.abs(x**2), dim=1)
    x = x**(1.0/2)
    return x  # x = [nslice, nframe, kx, ky] real


def np_strict_dc(mask_coil, y_input, x_rec):
    k_sample = mask_coil * y_input
    k_no_sample = (1 - mask_coil) * x_rec
    k_dc = k_sample + k_no_sample
    return k_dc


def torch_dc_csm_SDE(cc_inputs_real, k_inputs, mask, CSM, CSM_conj, lamb_dc):
    x_out = torch_double2complex(cc_inputs_real)  # [nx, 2*ncoil=2, nframe, ny] -> [nx, ncoil=1, nframe, ny]
    x_coil = x_out * CSM  # multi-coil image
    k_coil = torch_fft2c_Fast(x_coil, 0, -1)
    k_sample = mask * (torch.divide((k_inputs + lamb_dc * k_coil), (1 + lamb_dc)))
    k_no_sample = (1 - mask) * k_coil
    k_dc = k_sample + k_no_sample
    im_dc = torch_ifft2c_Fast(k_dc, 0, -1)
    cc_out_complex = torch.sum(im_dc * CSM_conj, dim=1, keepdim=True)  # [nx, ncoil=1, nframe, ny]
    cc_out = torch_complex2double(cc_out_complex)
    return cc_out  # real im: [nx, 2*ncoil=2, nframe, ny]


def torch_graddc_csm_SDE_forupdate(cc_inputs_real, x_inputs_real, mask, CSM, CSM_conj, lamb_dc):
    x_out = torch_double2complex(cc_inputs_real)  # [nx, 2*ncoil=2, nframe, ny] -> [nx, ncoil=1, nframe, ny]
    k_inputs = torch_fft2c_Fast(torch_double2complex(x_inputs_real) * CSM, 0, -1)
    x_coil = x_out * CSM  # multi-coil image
    k_coil = mask * torch_fft2c_Fast(x_coil, 0, -1)
    grad = torch_ifft2c_Fast(mask * (k_inputs - k_coil), 0, -1)
    grad = lamb_dc * torch.sum(grad * CSM_conj, dim=1, keepdim=True)  # [nx, ncoil=1, nframe, ny]
    cc_out_complex = x_out + grad
    cc_out = torch_complex2double(cc_out_complex)
    return cc_out  # real im: [nx, 2*ncoil=2, nframe, ny]


def torch_graddc_csm_SDE(cc_inputs_real, k_inputs, mask, CSM, CSM_conj, lamb_dc):
    x_out = torch_double2complex(cc_inputs_real)  # [nx, 2*ncoil=2, nframe, ny] -> [nx, ncoil=1, nframe, ny]
    x_coil = x_out * CSM  # multi-coil image
    k_coil = mask * torch_fft2c_Fast(x_coil, 0, -1)
    grad = torch_ifft2c_Fast(mask * (k_inputs - k_coil), 0, -1)
    grad = lamb_dc * torch.sum(grad * CSM_conj, dim=1, keepdim=True)  # [nx, ncoil=1, nframe, ny]
    cc_out_complex = x_out + grad
    cc_out = torch_complex2double(cc_out_complex)
    return cc_out  # real im: [nx, 2*ncoil=2, nframe, ny]


def torch_get_zfimage(k_inputs, mask, CSM_conj):
    k_sample = mask * k_inputs  # [nx, ncoil=8, nframe, ny]
    zfimage = torch_ifft2c_Fast(k_sample, 0, -1)
    cc_zfimage_complex = torch.sum(zfimage * CSM_conj, dim=1, keepdim=True)  # [nx, ncoil=1, nframe, ny]
    cc_zfimage = torch_complex2double(cc_zfimage_complex)
    return cc_zfimage  # real im: [nx, 2*ncoil=2, nframe, ny]


def torch_get_mask_conditional(cc_inputs_real, k_inputs, mask, CSM, CSM_conj):
    x_out = torch_double2complex(cc_inputs_real)  # [nx, 2*ncoil=2, nframe, ny] -> [nx, ncoil=1, nframe, ny]
    x_coil = x_out * CSM  # multi-coil image
    k_coil = torch_fft2c_Fast(x_coil, 0, -1)
    k_sample = mask * k_inputs
    k_no_sample = (1 - mask) * k_coil
    k_dc = k_sample + k_no_sample
    im_dc = torch_ifft2c_Fast(k_dc, 0, -1)
    cc_out_complex = torch.sum(im_dc * CSM_conj, dim=1, keepdim=True)  # [nx, ncoil=1, nframe, ny]
    cc_out = torch_complex2double(cc_out_complex)
    return cc_out  # real im: [nx, 2*ncoil=2, nframe, ny]



