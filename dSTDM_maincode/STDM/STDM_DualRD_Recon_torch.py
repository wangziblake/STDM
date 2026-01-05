# -*- coding: utf-8 -*-
"""
Reconstruction Code for dSTDM (Dual-directional SpatioTemporal Diffusion Model for Dynamic MRI) - Reverse diffusion sampling - torch

Created on 2024/12/01

@author: Zi Wang

If you want to use this code, please cite following paper:
Zi Wang et al., Robust cardiac cine MRI reconstruction with spatiotemporal diffusion model, IEEE Transactions on Computational Imaging, 11: 1258-1270, 2025.

Email: Dr. Zi Wang (zi.wang@imperial.ac.uk)
GitHub: https://github.com/wangziblake/STDM
"""

import os
GPU_used = '0'
os.environ["CUDA_VISIBLE_DEVICES"] = GPU_used
import torch
import sys
import pathlib
sys.path.insert(0, os.path.dirname(pathlib.Path(__file__).parent.absolute()))
import warnings
import time
import scipy.io as sio
from SDE_Models import ddpmv22  # Keep the import below for registering all model definitions
from Tools.Loaddata_STDM_SepSave_SDE_torch import *
from Tools import SDE_Lib as sde_lib
from Tools.Sampling import ReverseDiffusionPredictor, get_sampling_fn_withDC
from Tools.Utils_SDE_torch import restore_checkpoint, lambda_schedule_linear
from SDE_Models.ema import ExponentialMovingAverage
from Tools import Losses as losses
from STDM.STDM_Config import *


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # only show the error information
warnings.filterwarnings("ignore")
print("pyTorch version: " + torch.__version__)  # Code for pyTorch 1.10.0


# --------------------------------------- #
# ---------------For users--------------- #
# Input datasets:
# DMRI_mat:
root_data_dir = pathlib.Path('/media/NAS_R01_P1S1/USER_PATH/wangz/CMRxRecon_XMUA40')  # data root path
data_name = 'Cine_SAX'  # Cine_SAX
rec_data_name = 'Cine_SAX'  # Cine_SAX
data_num_start = 1  # default-1
data_all_num = 50  # Cine_SAX-50
SR = '10'
sampling_pattern = 'VRS'  # VISTA, VRS, UNI, RAD
# UNI SR13% (AF8), VRS SR10% (AF10) for Cine_SAX, Cine_LAX
recon = 'retrospective'  # retrospective, prospective

# Network configs:
Mode = 'DMRI_mat'  # Choose the model mode: DMRI_mat
model_scheme = 'STDM'  # Model scheme
model_scheme_save = 'dSTDM'  # Model scheme for saving

# GPU Configs:
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # cuda:0 use the first gpu in the gpu_list

# Seed for reproducibility
set_seed = 42
# torch.manual_seed(set_seed)  # CPU
# torch.cuda.manual_seed(set_seed)  # GPU
# ------------------End------------------ #
# --------------------------------------- #


# ---------------Not for users--------------- #
# Fixed configs:
config = train_config_stdm()
pth_model_number = 50
num_avesample = 1  # Number of samples for average, 1/2/4/8
predictor = ReverseDiffusionPredictor  # ReverseDiffusionPredictor

initial_conditional = 'RandomNoise'  # RandomNoise, ZeroFilled, NNRecon
config.sampling.N = config.model.num_scales
# Sampling iterations (Smaller --> faster sampling but worse quality), default: 100
real_num_scales = config.sampling.N * num_avesample

config.device = device
config.sampling.method = 'reverse_pdual'  # reverse_pdual
config.sampling.predictor = 'reverse_diffusion'  # reverse_diffusion
if config.sampling.method == 'reverse_pdual':
    schedule = 'linear'  # constant or linear increasing
    start_lamb = 0.0
    end_lamb = 0.0  # x_predict : y_sampled = lamb_dc : 1, default: (100-0.0)
else:
    print('Lambda setting error.')

# ---------------Start Reconstruction--------------- #
print("-----------------------------")
print("Sampling rate: %s%%" % SR)
print("Model: %s" % model_scheme)
print("Data: %s" % data_name)
print("-----------------------------")
print("Reconstruction starts.\n")


# 1 ----- Mode DMRI_mat: For dynamic MRI reconstruction. -----#
# Input format: .mat
# Output format: .mat
if Mode == 'DMRI_mat':
    # Call model
    model = mutils.create_model(config)
    model = model.to(config.device)

    # lamb_dc
    if schedule == 'linear':
        lamb_schedule = lambda_schedule_linear(num_total=config.sampling.N, start_lamb=start_lamb, end_lamb=end_lamb)
    else:
        raise NotImplementedError(f"Given schedule {schedule} not implemented yet!")

    # Setup SDEs
    if config.training.sde.lower() == 'vesde':
        sde = sde_lib.VESDE(config, sigma_min=config.model.sigma_min, sigma_max=config.model.sigma_max, N=config.model.num_scales)
        sampling_eps = 1e-5
    else:
        raise NotImplementedError(f"SDE {config.training.sde} unknown.")

    optimizer = losses.get_optimizer(config, model.parameters())
    ema = ExponentialMovingAverage(model.parameters(), decay=config.model.ema_rate)
    state = dict(optimizer=optimizer, model=model, ema=ema, step=0)

    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(base_dir, '%s_%s' % (model_scheme, data_name))
    ckpt_filename = os.path.join(base_dir, model_dir, 'Saved_Model_%d.pth' % (pth_model_number))
    state = restore_checkpoint(ckpt_filename, state, config.device, skip_sigma=True)
    ema.copy_to(model.parameters())

    # Save Data
    data_rec_dir = root_data_dir / 'Data' / 'Result' / 'DMRI' / rec_data_name / f'{model_scheme_save}_{data_name}_{real_num_scales}' / sampling_pattern / f'SR{SR}'
    os.makedirs(data_rec_dir)

    # Model evaluation (testing)
    for k in range(data_num_start, data_num_start + data_all_num):
        if recon == 'retrospective':
            data_KFS_dir = root_data_dir / 'Data' / 'TestData' / 'DMRI' / rec_data_name / 'KFS'
            load_K = loaddata_DMRI_KFS(data_KFS_dir, k)
        elif recon == 'prospective':
            data_KUS_dir = root_data_dir / 'Data' / 'TestData' / 'DMRI' / rec_data_name / 'KUS'
            load_K = loaddata_DMRI_KUS(data_KUS_dir, k)
        else:
            print('Error.')

        data_Mask_dir = root_data_dir / 'Data' / 'TestData' / 'DMRI' / rec_data_name / sampling_pattern / f'SR{SR}' / 'Mask'
        data_CSM_dir = root_data_dir / 'Data' / 'TestData' / 'DMRI' / rec_data_name / 'CSM'
        load_Mask = loaddata_DMRI_Mask(data_Mask_dir, k)
        load_CSM = loaddata_DMRI_CSM(data_CSM_dir, k)

        if config.data.norm == 'imagemax':
                cc_inputs, k_inputs, mask, factor_norm, CSM, CSM_conj = getdata_Test_DMRI_CSM_imagemaxnorm_SDEv3(load_K, load_Mask, load_CSM)
        else:
            print('Data error.')

        cc_inputs = torch.tensor(cc_inputs, dtype=torch.complex64).to(config.device)
        k_inputs = torch.tensor(k_inputs, dtype=torch.complex64).to(config.device)
        mask = torch.tensor(mask, dtype=torch.complex64).to(config.device)
        CSM = torch.tensor(CSM, dtype=torch.complex64).to(config.device)
        CSM_conj = torch.tensor(CSM_conj, dtype=torch.complex64).to(config.device)
        tfactor_norm = torch.tensor(factor_norm, dtype=torch.float32).to(config.device)

        sampling_withDC = get_sampling_fn_withDC(config, sde, eps=sampling_eps)

        output_complex = 0
        tic = time.time()
        for i in range(num_avesample):
            if config.sampling.method == 'reverse_pdual':
                output_real = sampling_withDC(model, cc_inputs, k_inputs, mask, CSM, CSM_conj, tfactor_norm, lamb_schedule, initial_conditional)
            else:
                print('Sampling error.')
            output_complex = output_complex + torch_double2complex(output_real)
        output_ave = output_complex / num_avesample
        output = output_ave.cpu().data.numpy()  # [batchsize=nx, ncoil=1, nframe, ny]

        image_rec_de_norm = output * factor_norm
        image_rec_de_norm = np.transpose(image_rec_de_norm, (0, 3, 2, 1))  # -> [nx, ny, nframe, ncoil]
        image_rec_de_norm = np.expand_dims(image_rec_de_norm, axis=0)  # -> [nslice=1, nx, ny, nframe, ncoil]
        toc = time.time()

        print('No.%3d: Testing duration is %.4f s.' % (k - data_num_start + 1, toc - tic))

        # Save as .mat
        mdict = {'dSTDM_image_rec': image_rec_de_norm}
        savefile_name = os.path.join(data_rec_dir, r'dSTDM_' + str(k) + '.mat')
        sio.savemat(savefile_name, mdict)
# 1 ----- End Mode DMRI_mat: For dynamic MRI reconstruction. -----#


# ---------------End Reconstruction--------------- #
print("\nReconstruction finishes.")
print("-----------------------------")
print("Sampling rate: %s%%" % SR)
print("Model: %s" % model_scheme)
print("Rec_Data: %s" % rec_data_name)
print("Number of samples: %d" % data_all_num)
print("-----------------------------")

