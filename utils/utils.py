import os
os.environ['OPENBLAS_WARNINGS'] = '0'
os.environ["OMP_NUM_THREADS"] = "1" 
os.environ["MKL_NUM_THREADS"] = "1" 
import cv2
cv2.setNumThreads(0)
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as tdist
from torch.utils.data import Dataset, DataLoader
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler
from torch.optim.lr_scheduler import *
import glob
import matplotlib
# matplotlib.use('AGG')
import matplotlib.pyplot as plt
import numpy as np
import gc
from PIL import Image
import time
import socket
import scipy
import scipy.io as sio
from scipy import stats
import argparse
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
from multiprocessing import Pool
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from torch.nn.parallel import DistributedDataParallel as DDP
import threading
from functools import wraps
from tqdm import tqdm
import exifread
import rawpy
import math
import random
import yaml
import pickle
import warnings
import h5py
import pickle
import pickle as pkl
from natsort import natsort
import scipy.io
from scipy.stats import poisson, norm
from scipy.signal import convolve
from scipy.interpolate import interp1d
import warnings
import kornia.filters as kf

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
setup_seed(1997)

fn_time = {}

def timestamp(time_points, n):
    time_points[n] = time.time()
    return time_points[n] - time_points[n-1]

def fn_timer(function, print_log=False):
  @wraps(function)
  def function_timer(*args, **kwargs):
    global fn_timer
    t0 = time.time()
    result = function(*args, **kwargs)
    t1 = time.time()
    if print_log:
        print ("Total time running %s: %.6f seconds" %
            (function.__name__, t1-t0))
    if function.__name__ in fn_time :
        fn_time[function.__name__] += t1-t0
    else:
        fn_time[function.__name__] = t1-t0
    return result
  return function_timer

def log(string, log=None, str=False, end='\n', notime=False):
    log_string = f'{time.strftime("%Y-%m-%d %H:%M:%S")} >>  {string}' if not notime else string
    print(log_string)
    if log is not None:
        with open(log,'a+') as f:
            f.write(log_string+'\n')
    else:
        pass
        # os.makedirs('worklog', exist_ok=True)
        # log = f'worklog/worklog-{time.strftime("%Y-%m-%d")}.txt'
        # with open(log,'a+') as f:
        #     f.write(log_string+'\n')
    if str:
        return string+end
    
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f', log=True, last_epoch=0):
        self.name = name
        self.fmt = fmt
        self.log = log
        self.history = []
        self.last_epoch = last_epoch
        self.history_init_flag = False
        self.reset()

    def reset(self):
        if self.log:
            try:
                if self.avg>0: self.history.append(self.avg)
            except:
                pass#print(f'Start log {self.name}!')
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
    
    def plot_history(self, savefile='log.jpg', logfile='log.pkl'):
        # 读取老log
        if os.path.exists(logfile) and not self.history_init_flag:
            self.history_init_flag = True
            with open(logfile, 'rb') as f:
                history_old = pickle.load(f)
                if self.last_epoch: # 为0则重置
                    self.history = history_old + self.history[:self.last_epoch]
        # 记录log
        with open(logfile, 'wb') as f:
            pickle.dump(self.history, f)
        # 画图
        plt.figure(figsize=(12,9))
        plt.title(f'{self.name} log')
        x = list(range(len(self.history)))
        plt.plot(x, self.history)
        plt.xlabel('Epoch')
        plt.ylabel(self.name)
        plt.savefig(savefile, bbox_inches='tight')
        plt.close()

    def __str__(self):
        fmtstr = '{name}:{val' + self.fmt + '}({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

def pkl_convert(param):
    return {
        k.replace("module.", ""): v
        for k, v in param.items()
        if "module." in k
    }

def load_weights(model, pretrained_dict, multi_gpu=False, by_name=False):
    model_dict = model.module.state_dict() if multi_gpu else model.state_dict()
    # 1. filter out unnecessary keys
    tsm_replace = []
    for k in pretrained_dict:
        if 'tsm_shift' in k:
            k_new = k.replace('tsm_shift', 'tsm_buffer')
            tsm_replace.append((k, k_new))
    for k, k_new in tsm_replace:
        pretrained_dict[k_new] = pretrained_dict[k]
    if by_name:
        del_list = []
        for k, v in pretrained_dict.items():
            if k in model_dict:
                if model_dict[k].shape != pretrained_dict[k].shape:
                    # 1. Delete values not in key
                    del_list.append(k)
                    # 2. Cat it to the end
                    # diff = model_dict[k].size()[1] - pretrained_dict[k].size()[1]
                    # v = torch.cat((v, v[:,:diff]), dim=1)
                    # 3. Repeat it to same
                    # nframe = model_dict[k].shape[1] // pretrained_dict[k].shape[1]
                    # v = torch.repeat_interleave(v, nframe, dim=1)
                    # 4. Clip it to same
                    # c_model = model_dict[k].shape[1]
                    # c_save = pretrained_dict[k].shape[1]
                    # c_diff = c_model - c_save
                    # if c_model > c_save:
                    #     v = torch.cat((v, torch.empty(b_model, c_diff, h_model, w_model).cuda()), dim=1)
                    # else:
                    #     v = v[:,:c_diff]
                    log(f'Warning:  "{k}":{pretrained_dict[k].shape}->{model_dict[k].shape}')
                pretrained_dict[k] = v
            else:
                del_list.append(k)
                log(f'Warning:  "{k}" is not exist and has been deleted!!')
        for k in del_list:
            del pretrained_dict[k]
    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict)
    if multi_gpu:
        model.module.load_state_dict(model_dict)
    else:
        model.load_state_dict(model_dict)
    # for name, parameter in model.named_parameters():
    #     if name not in ["input_blocks.0.0.weight", "out.2.weight", "out.2.bias"]:
    #     # if name not in del_list:
    #         parameter.requires_grad = False
    #         log(f'Warning: layer except "{name}" is freezed...')
    return model

def tensor_dimxto4(tensor):
    c, h, w = tensor.shape[-3:]
    tensor = tensor.reshape(-1, c, h, w)
    return tensor

def tensor_dimxto5(tensor):
    t, c, h, w = tensor.shape[-4:]
    tensor = tensor.reshape(-1, t, c, h, w)
    return tensor

def tensor_dim5to4(tensor):
    batchsize, crops, c, h, w = tensor.shape
    tensor = tensor.reshape(batchsize*crops, c, h, w)
    return tensor

def tensor_dim6to5(tensor):
    batchsize, crops, t, c, h, w = tensor.shape
    tensor = tensor.reshape(batchsize*crops, t, c, h, w)
    return tensor

def get_host_with_dir(dataset_name=''):
    multi_gpu = False
    hostname = socket.gethostname()
    log(f"User's hostname is '{hostname}'")
    if hostname == 'ubun':
        host = '/data/fenghansen/datasets'
    elif hostname == 'DESKTOP-LGD8S6F': # BIT-816
        host = 'E:/datasets'
    elif hostname == 'MoreTiming':
        host = 'F:/datasets'
    else:
        host = '/data'
        multi_gpu = True if torch.cuda.device_count() > 1 else False
    return hostname, host + dataset_name, multi_gpu

def get_p2d(shape, base=16):
    xb, xc, xh, xw = shape
    yh, yw = ((xh-1)//base+1)*base, ((xw-1)//base+1)*base
    diffY = yh - xh
    diffX = yw - xw
    p2d = (diffX // 2, diffX - diffX//2, diffY // 2, diffY - diffY//2)
    return p2d

def big_image_split(data, n=2, pad=64):
    # 把大图分割成小图
    p2d = get_p2d(data.shape, base=pad)
    data = F.pad(data, p2d, mode='reflect')
    data = torch.cat(torch.chunk(data, n, dim=2), dim=0)
    data = torch.cat(torch.chunk(data, n, dim=3), dim=0)
    return data, p2d

def big_image_merge(data, n=2, p2d=[0,0,0,0]):
    # 把小图合并成大图
    data = torch.cat(torch.chunk(data, n, dim=0), dim=3)
    data = torch.cat(torch.chunk(data, n, dim=0), dim=2)
    H, W = data.shape[-2:]
    data = data[..., p2d[-2]:H-p2d[-1], p2d[0]:W-p2d[1]]
    return data

def tensor2numpy(data, eval=True, transpose=True):
    if eval: data = data[0]
    data = data.detach().cpu().numpy()
    if transpose: 
        length = len(data.shape)
        if length == 3:
            data = data.transpose(1,2,0)
        elif length == 4:
            data = data.transpose(0,2,3,1)
    return data

def numpy2tensor(data, device='cpu', eval=True, transpose=True, clone=False):
    if clone: data = data.copy()
    data = torch.from_numpy(np.ascontiguousarray(data)).float().to(device)
    length = len(data.shape)
    if transpose:
        if length == 3: 
            data = data.permute(2,0,1)
        elif length == 2:
            data = data[None,:]
        elif length == 4:
            return data.permute(0,3,1,2)
    if eval:
        data = data[None,:]
    return data

def read_paired_fns(filename):
    with open(filename) as f:
        fns = f.readlines()
        fns = [tuple(fn.strip().split(' ')) for fn in fns]
    return fns

def metrics_recorder(file, names, psnrs, ssims):
    if os.path.exists(file):
        with open(file, 'rb') as f:
            metrics = pkl.load(f)
    else:
        metrics = {}
    for name, psnr, ssim in zip(names, psnrs, ssims):
        metrics[name] = [psnr, ssim]
    with open(file, 'wb') as f:
        pkl.dump(metrics, f)
    return metrics

def mpop(func, idx, *args, **kwargs):
    data = func(*args, **kwargs)
    log(f'Finish task No.{idx}...')
    return idx, func(*args, **kwargs)

def dataload(path):
    suffix = path[-4:].lower()
    if suffix in ['.arw','.dng','.nef','.cr2']:
        data = rawpy.imread(path).raw_image_visible
    elif suffix in ['.raw']:
        data = np.fromfile(path, np.uint16).reshape(1440, 2560)
    elif suffix in ['.npy']:
        data = np.load(path)
    elif suffix in ['.mat']:
        if 'metadata' in path.lower():
            data = scipy.io.loadmat(path)
        else:
            with h5py.File(path, 'r') as f:
                data = np.array(f['x'])
    elif suffix in ['.jpg', '.png', '.bmp', 'tiff']:
        data = cv2.imread(path)[:,:,::-1]
    return data

# 把ELD模型中的Unet权重单独提取出来
def pth_transfer(src_path='/data/ELD/checkpoints/sid-ours-inc4/model_200_00257600.pt',
                dst_path='checkpoints/SonyA7S2_Official.pth',
                reverse=False):
    model_src = torch.load(src_path, map_location='cpu')
    if reverse:
        model_dst = torch.load(dst_path, map_location='cpu')
        model_src['netG'] = model_dst
        save_dir = os.path.join('pth_transfer', os.path.basename(dst_path)[9:-15])
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, os.path.basename(src_path))
        torch.save(model_src, save_path)
    else:
        model_src = model_src['netG']
        torch.save(model_src, dst_path)
