# -*- coding: utf-8 -*-
"""
Loaddata_1D_Extendto3DFormat code - pytorch - for SDE - separable save in KFS, Mask, CSM

Created on 2024/09/06

@author: Zi Wang

If you want to use this code, please cite following paper:
Zi Wang et al., Robust cardiac cine MRI reconstruction with spatiotemporal diffusion model, IEEE Transactions on Computational Imaging, 11: 1258-1270, 2025.

Email: Dr. Zi Wang (zi.wang@imperial.ac.uk)
GitHub: https://github.com/wangziblake/STDM
"""

import h5py
import os
import numpy as np
from Tools.Tools_torch import *


def loaddata_1DExtend_DMRI_KFS_Minibatch(dir_file):
    with h5py.File(dir_file, mode='r') as h5file:
        #  label k-space
        #  Complex
        label_k_space = h5file['k_label_1D'][:]  # [ncoil=8, nframe=10, ky=225, nx=1]
        label_k_space = label_k_space['real'] + label_k_space['imag'] * 1j
        KFS = np.transpose(label_k_space.astype(np.complex64), (0, 1, 3, 2))  # [8, 10, 1, 225]
        return KFS


def loaddata_1DExtend_DMRI_Mask_Minibatch(dir_file):
    with h5py.File(dir_file, mode='r') as h5file:
        #  mask coil
        #  Float
        mask_coil = h5file['mask_1D'][:]  # [nframe=10, ky=225, nx=1]
        mask_coil = np.expand_dims(mask_coil, axis=0)  # [1, 10, 225, 1]
        mask_coil = np.transpose(mask_coil.astype(np.complex64), (0, 1, 3, 2))  # [1, 10, 1, 225]
        Mask = mask_coil * (1 + 0j)  # Complex
        return Mask


def loaddata_1DExtend_DMRI_CSM_Minibatch(dir_file):
    with h5py.File(dir_file, mode='r') as h5file:
        #  CSM
        #  Complex
        CSM = h5file['CSM_1D'][:]  # [8, 10, 225, 1]
        CSM = CSM['real'] + CSM['imag'] * 1j
        CSM = np.transpose(CSM.astype(np.complex64), (0, 1, 3, 2))  # [8, 10, 1, 225]
        return CSM


def getdata_Train_1DExtend_DMRI_CSM_imagemaxnorm_Minibatch_SDE(KFS, CSM):
    #  label k-space
    #  Complex
    Training_k_space = KFS
    Training_image = np_ifft1c_DMRI(Training_k_space, -1)

    # Coil-combined image
    CSM_conj = np.conj(CSM)
    Training_ccimage = np.sum(Training_image * CSM_conj, axis=0, keepdims=True)  # complex single-coil [1, 10, 1, 225]
    Training_ccimage = np.squeeze(Training_ccimage, axis=-2)  # complex single-coil [ncoil=1, nframe=10, ny=225]

    #  Normalization
    factor_norm = np.amax(np.abs(Training_ccimage), axis=(-3, -2, -1), keepdims=True)
    # factor_norm = 1  # No normalization

    Training_cc_inputs_Norm = Training_ccimage / factor_norm

    return Training_cc_inputs_Norm, np.squeeze(CSM, axis=-2), np.squeeze(CSM_conj, axis=-2)
    # complex single-coil [ncoil=1, nframe=10, ny=225], nx will be used as batchsize


def loaddata_DMRI_KFS(dataset_dir, number_part):
    dir_file = os.path.join(dataset_dir, r'K_Data_Part{number_part}.mat').format(number_part=number_part)
    with h5py.File(dir_file) as h5file:
        #  label k-space
        #  Complex
        label_k_space = h5file['k_label'][:]  # [ncoil=8, nframe=10, ky=225, kx=224]
        label_k_space = label_k_space['real'] + label_k_space['imag'] * 1j
        KFS = np.transpose(label_k_space.astype(np.complex64), (0, 1, 3, 2))  # [8, 10, 224, 225]
        return KFS


def loaddata_DMRI_Mask(dataset_dir, number_part):
    dir_file = os.path.join(dataset_dir, r'Mask_Data_Part{number_part}.mat').format(number_part=number_part)
    if not os.path.exists(dir_file):
        dir_file = os.path.join(dataset_dir, r'Mask_Data_Part1.mat')
    with h5py.File(dir_file) as h5file:
        #  mask coil
        #  Float
        mask_coil = h5file['mask_coil'][:]  # [nframe=10, ky=225, kx=224]
        mask_coil = np.expand_dims(mask_coil, axis=0)  # [1, 10, 225, 224]
        mask_coil = np.transpose(mask_coil.astype(np.complex64), (0, 1, 3, 2))  # [1, 10, 224, 225]
        Mask = mask_coil * (1 + 0j)  # Complex
        return Mask


def loaddata_DMRI_CSM(dataset_dir, number_part):
    dir_file = os.path.join(dataset_dir, r'CSM_Data_Part{number_part}.mat').format(number_part=number_part)
    with h5py.File(dir_file) as h5file:
        #  CSM
        #  Complex
        CSM = h5file['CSM'][:]  # [8, 10, 225, 224]
        CSM = CSM['real'] + CSM['imag'] * 1j
        CSM = np.transpose(CSM.astype(np.complex64), (0, 1, 3, 2))  # [8, 10, 224, 225]
        return CSM


def getdata_Test_DMRI_CSM_imagemaxnorm_SDEv3(KFS, Mask, CSM):
    #  input k-space
    #  Complex
    Testing_k_space = KFS  # [8, 10, kx=224, ky=225]
    Testing_image = np_ifft2c_DMRI(Testing_k_space, -2, -1)

    # mask coil
    Testing_mask_coil = Mask  # [1, 10, 224, 225]

    # Coil-combined image
    CSM_conj = np.conj(CSM)  # [8, 10, 224, 225]
    Testing_ccimage = np.sum(Testing_image * CSM_conj, axis=0, keepdims=True)  # complex single-coil [1, 10, 224, 225]

    # Transpose nx to the first dimension
    Testing_k_space = np.transpose(Testing_k_space, (2, 0, 1, 3))  # [kx=224, ncoil=8, nframe=10, ky=225]
    Testing_ccimage = np.transpose(Testing_ccimage, (2, 0, 1, 3))  # [nx=224, ncoil=1, nframe=10, ny=225]
    Testing_mask_coil = np.transpose(Testing_mask_coil, (2, 0, 1, 3))  # [kx=224, ncoil=1, nframe=10, ky=225]
    CSM = np.transpose(CSM, (2, 0, 1, 3))  # [nx=224, ncoil=8, nframe=10, ny=225]
    CSM_conj = np.transpose(CSM_conj, (2, 0, 1, 3))  # [nx=224, ncoil=8, nframe=10, ny=225]

    #  Normalization (use FS-for-factor_norm)
    factor_norm = np.amax(np.abs(Testing_ccimage), axis=(-4, -3, -2, -1), keepdims=True)

    Testing_k_inputs_Norm = (Testing_k_space * Testing_mask_coil) / factor_norm

    Testing_cc_inputs_Norm = np.sum(np_ifft2c_DMRI(Testing_k_space * Testing_mask_coil, 0, -1) * CSM_conj, axis=1,
                                    keepdims=True) / factor_norm

    return Testing_cc_inputs_Norm, Testing_k_inputs_Norm, Testing_mask_coil, factor_norm, CSM, CSM_conj


def loaddata_DMRI_KUS(dataset_dir, number_part):
    dir_file = os.path.join(dataset_dir, r'K_Data_Part{number_part}.mat').format(number_part=number_part)
    with h5py.File(dir_file) as h5file:
        #  input k-space
        #  Complex
        input_k_space = h5file['k_input'][:]  # [ncoil=8, nframe=10, ky=225, kx=224]
        input_k_space = input_k_space['real'] + input_k_space['imag'] * 1j
        KUS = np.transpose(input_k_space.astype(np.complex64), (0, 1, 3, 2))  # [8, 10, 224, 225]
        return KUS
