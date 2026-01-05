# -*- coding: utf-8 -*-
"""
Training Code for dSTDM (Dual-directional SpatioTemporal Diffusion Model for Dynamic MRI) - torch load mini-batch data
- diffusion with CSM - in order

Created on 2024/07/30

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
import scipy.io as sio
import time
import platform
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from SDE_Models import ddpmv22  # Keep the import below for registering all model definitions
from Tools.Loaddata_STDM_SepSave_SDE_torch import *
from Tools.Utils_SDE_torch import restore_checkpoint, save_checkpoint
from SDE_Models.ema import ExponentialMovingAverage
from Tools import SDE_Lib as sde_lib
from Tools import Losses as losses
from STDM.STDM_Config import *

model_scheme = 'STDM'  # Model scheme

trainnum = 191424  # 997*192 = 191424
valnum = 19200  # 100*192 = 19200

# root_data_dir = pathlib.Path('/media/NAS_R01_P1S1/USER_PATH/wangz/CMRxRecon_XMUA40')  # data root path
root_data_dir = pathlib.Path('/media/ssd/wangzi/CMRxRecon_XMUA40')  # data root path
data_name = 'Cine_SAX'  # Cine_SAX
batch_size = 64
epoch = 50

# Only for continue training from a checkpoint
pth_model_number = 1

print_flag = 0  # print parameter number

# GPU Configs
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # cuda:0 use the first gpu in the gpu_list

# Seed for reproducibility
set_seed = 42
# torch.manual_seed(set_seed)  # CPU
# torch.cuda.manual_seed(set_seed)  # GPU

# ---------------Start Running/Model Configuration---------------- #
config = train_config_stdm()
config.device = device
# ---------------Start Running/Model Configuration---------------- #


# ---------------Start Minibatch and Read Data Size---------------- #
class RandomDataset_Minibatch_SepSave_SDE(Dataset):
    def __init__(self, samples_KFS: list, samples_CSM: list, config):
        self.KFS = samples_KFS
        self.CSM = samples_CSM
        self.norm_type = config.data.norm

    def __getitem__(self, index):
        load_KFS = loaddata_1DExtend_DMRI_KFS_Minibatch(self.KFS[index])
        load_CSM = loaddata_1DExtend_DMRI_CSM_Minibatch(self.CSM[index])
        if self.norm_type == 'imagemax':
            # Note: Use 8 TIs for T1 mapping, network does not support 9 TIs !!!
            cc_inputs, CSM, CSM_conj = getdata_Train_1DExtend_DMRI_CSM_imagemaxnorm_Minibatch_SDE(load_KFS, load_CSM)
        else:
            print('Error.')
        cc_inputs = torch.tensor(cc_inputs, dtype=torch.complex64)  # Fully-sampled coil-combined image
        CSM = torch.tensor(CSM, dtype=torch.complex64)
        CSM_conj = torch.tensor(CSM_conj, dtype=torch.complex64)

        return cc_inputs, CSM, CSM_conj

    def __len__(self):
        return len(self.KFS)
# ---------------End Minibatch and Read Data Size--------------- #


# ---------------Start Training and Save--------------- #
print("---------------")
print("Start training.")

# Build model
model = mutils.create_model(config)
model = model.to(config.device)
# Print parameters
if print_flag:
    num_count = 0
    for para in model.parameters():
        num_count += 1
        print('Layer %d' % num_count)
        print(para.size())
# Setup SDEs
if config.training.sde.lower() == 'vesde':
    sde = sde_lib.VESDE(config, sigma_min=config.model.sigma_min, sigma_max=config.model.sigma_max, N=config.model.num_scales)
    sampling_eps = 1e-5
else:
    raise NotImplementedError(f"SDE {config.training.sde} unknown.")
# Create data normalizer and its inverse
scaler = mutils.get_data_scaler(config)
# Optimizer
ema = ExponentialMovingAverage(model.parameters(), decay=config.model.ema_rate)
optimizer = losses.get_optimizer(config, model.parameters())
# State
state = dict(optimizer=optimizer, model=model, ema=ema, step=0)
# Build one-step training and validation functions
optimize_fn = losses.optimization_manager(config)
continuous = config.training.continuous
reduce_mean = config.training.reduce_mean
likelihood_weighting = config.training.likelihood_weighting
train_step_fn = losses.get_step_fn(sde, train=True, optimize_fn=optimize_fn, reduce_mean=reduce_mean, continuous=continuous,
                                   likelihood_weighting=likelihood_weighting)
val_step_fn = losses.get_step_fn(sde, train=False, optimize_fn=optimize_fn, reduce_mean=reduce_mean, continuous=continuous,
                                  likelihood_weighting=likelihood_weighting)


# Training and validation loop
base_dir = os.path.dirname(os.path.abspath(__file__))
model_dir = os.path.join(base_dir, '%s_%s' % (model_scheme, data_name))
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
    start_epoch = 1
else:
    pth_ckpt = os.path.join(base_dir, model_dir, 'Saved_Model_%d.pth' % (pth_model_number))
    state = restore_checkpoint(pth_ckpt, state, config.device)
    start_epoch = pth_model_number + 1
    print("Continue learning from checkpoint %d." % pth_model_number)
os.chdir(model_dir)

cruve_dir = 'Train_cruve'
if not os.path.exists(cruve_dir):
    os.makedirs(cruve_dir)
else:
    pass

output_file_name = ("Log_output_%s_%s.txt" % (model_scheme, data_name))
val_file_name = ("Log_val_%s_%s.txt" % (model_scheme, data_name))
loss_file_name = ("Loss_%s_%s.txt" % (model_scheme, data_name))
valloss_file_name = ("Val_Loss_%s_%s.txt" % (model_scheme, data_name))

print("Training dataset is %s." % data_name)

# --------------------------------------------------------------------------------------
# Path of the training dataset (Change filepath due to os.chdir(model_dir))
filepathtrain_KFS = root_data_dir / 'Data' / 'TrainData_1D' / 'DMRI' / data_name / 'KFS'
filepathtrain_CSM = root_data_dir / 'Data' / 'TrainData_1D' / 'DMRI' / data_name / 'CSM'
filepathval_KFS = root_data_dir / 'Data' / 'ValData_1D' / 'DMRI' / data_name / 'KFS'
filepathval_CSM = root_data_dir / 'Data' / 'ValData_1D' / 'DMRI' / data_name / 'CSM'

samples_list_train_KFS, samples_list_train_Mask, samples_list_train_CSM = [], [], []
samples_list_val_KFS, samples_list_val_Mask, samples_list_val_CSM = [], [], []

for k in range(1, trainnum + 1):  # in-order
    train_dir_file_KFS = os.path.join(filepathtrain_KFS, r'K_Data_Part{number_part}.mat').format(number_part=k)
    train_dir_file_CSM = os.path.join(filepathtrain_CSM, r'CSM_Data_Part{number_part}.mat').format(number_part=k)
    samples_list_train_KFS.append(train_dir_file_KFS)
    samples_list_train_CSM.append(train_dir_file_CSM)

for k in range(1, valnum + 1):  # in-order
    val_dir_file_KFS = os.path.join(filepathval_KFS, r'K_Data_Part{number_part}.mat').format(number_part=k)
    val_dir_file_CSM = os.path.join(filepathval_CSM, r'CSM_Data_Part{number_part}.mat').format(number_part=k)
    samples_list_val_KFS.append(val_dir_file_KFS)
    samples_list_val_CSM.append(val_dir_file_CSM)

train_dataset = RandomDataset_Minibatch_SepSave_SDE(samples_list_train_KFS[0:trainnum], samples_list_train_CSM[0:trainnum], config)
val_dataset = RandomDataset_Minibatch_SepSave_SDE(samples_list_val_KFS[0:valnum], samples_list_val_CSM[0:valnum], config)

# Dataloader
nw = min([batch_size, os.cpu_count() // 4, 4])  # type: ignore # num_workers
print(f'Number of worker {nw}')

if platform.system() == "Windows":
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, num_workers=0, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, num_workers=0, shuffle=True)
else:
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, num_workers=nw, shuffle=True, pin_memory=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, num_workers=nw, shuffle=True, pin_memory=True)
# --------------------------------------------------------------------------------------

# Model training and validation, save model checkpoint
train_total_aveloss = []
val_total_aveloss = []
loss_min = 100.0

for i in range(start_epoch, epoch + 1):  # training
    tic_i = time.time()
    totalloss = []

    train_loader = tqdm(train_loader, desc=f'[epoch {i}]')
    for step, batch in enumerate(train_loader, start=1):
        b_cc_inputs, b_CSM, b_CSM_conj = batch
        b_cc_inputs = b_cc_inputs.to(config.device)
        b_cc_inputs_real = scaler(torch_complex2double(b_cc_inputs))  # [batchsize, 2*ncoil, nframe, ny]
        b_CSM = b_CSM.to(config.device)
        b_CSM_conj = b_CSM_conj.to(config.device)

        # loss: Zero gradients, perform a backward pass, and update the weights
        loss = train_step_fn(state, b_cc_inputs_real, b_CSM, b_CSM_conj)

        train_loader.desc = f'[epoch {i}] loss={round(loss.item(),5)}'
        totalloss.append(loss.item())
        # torch.cuda.empty_cache()
        del b_cc_inputs, b_CSM, b_CSM_conj
    avgtrainloss = np.mean(totalloss)
    toc_i = time.time()

    output = ('After %4d epoch(s), loss is %.5f, and duration is %.2f s.' %
              (i, avgtrainloss, toc_i - tic_i))

    print(output)
    output_file = open(output_file_name, 'a')
    output_file.write(output + '\n')
    output_file.close()

    if i % 1 == 0:  # validation
        tic_v = time.time()
        totalloss_val = []

        val_loader = tqdm(val_loader, desc=f'[valepoch {i}]')
        for val_step, batchv in enumerate(val_loader, start=1):
            bv_cc_inputs, bv_CSM, bv_CSM_conj = batchv
            bv_cc_inputs = bv_cc_inputs.to(config.device)
            bv_cc_inputs_real = scaler(torch_complex2double(bv_cc_inputs))  # [batchsize, 2*ncoil, nframe, ny]
            bv_CSM = bv_CSM.to(config.device)
            bv_CSM_conj = bv_CSM_conj.to(config.device)

            # loss: No_grad, do not update the weights
            lossv = val_step_fn(state, bv_cc_inputs_real, bv_CSM, bv_CSM_conj)

            val_loader.desc = f'[valepoch {i}] loss={round(lossv.item(), 5)}'
            totalloss_val.append(lossv.item())
            # torch.cuda.empty_cache()
            del bv_cc_inputs, bv_CSM, bv_CSM_conj
        avgtrainloss_val = np.mean(totalloss_val)
        toc_v = time.time()
        output_v = ('---After %4d epoch(s), val_loss is %.5f, and duration is %.2f s.' %
                    (i, avgtrainloss_val, toc_v - tic_v))

        print(output_v)
        output_v_file = open(val_file_name, 'a')
        output_v_file.write(output_v + '\n')
        output_v_file.close()

    # Save model at a specific checkpoint
    # if i <= epoch:
    #     loss_val = avgtrainloss_val
    #     if loss_val < loss_min:
    #         loss_min = loss_val
    #         save_checkpoint(state, name='Saved_Model_%d.pth' % i)
    save_checkpoint(state, name='Saved_Model_%d.pth' % i)

    # Training/validation curves
    train_total_aveloss.append(avgtrainloss)
    val_total_aveloss.append(avgtrainloss_val)

# Save training/validation curves as .mat
mdict = {'trainloss': train_total_aveloss, 'valloss': val_total_aveloss}
savefile_name = os.path.join(cruve_dir, r'train_cruve.mat')
sio.savemat(savefile_name, mdict)

print("Training finished.")
print("------------------")
