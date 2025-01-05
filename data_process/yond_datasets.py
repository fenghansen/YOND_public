import pickle
import torch
import numpy as np
import cv2
import os
import h5py
import rawpy
from matplotlib import pyplot as plt
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from .unprocess import mosaic, unprocess, random_gains
from .process import *
from utils import *

def bayer_aug(rggb, k=0):
    bayer = rggb2bayer(rggb)
    bayer = np.rot90(bayer, k=k, axes=(-2,-1))
    rggb = bayer2rggb(bayer)
    return rggb

# def get_threshold(data, step=1, mode='score2', print=False):
#     if mode == 'naive':
#         quants = list(range(step, 100, step))
#         ths = np.percentile(data.reshape(-1), quants, method='linear')
#         ths = np.array(ths.tolist() + [data.max()])
#         th = ths[0]
#         diffs = ths[1:] - ths[:-1]
#         gap = data.max() / (100/step)**2
#         for i in range(len(diffs)):
#             if diffs[i] > gap:
#                 if print: log(f'Adaptive percent: {quants[i]}% - th: {ths[i]*959:.2f}')
#                 th = ths[i]
#                 break
#     elif mode == 'score':
#         quants = np.linspace(step, 100, 100//step)
#         ths = np.percentile(data.reshape(-1), quants, method='linear')
#         diffs = ths[1:] - ths[:-1]
#         quants = quants[:-1]
#         score = diffs/quants
#         i = np.argmin(score)
#         th = ths[i]
#         if print: log(f'Adaptive percent: {quants[i]}% - th: {ths[i]*959:.2f}')
#     return th

# Raw AWGN Dataset(Raw->Raw)
class SID_Raw_Dataset(Dataset):
    def __init__(self, args=None):
        super().__init__()
        self.default_args()
        if args is not None:
            for key in args:
                self.args[key] = args[key]
        self.initialization()

    def default_args(self):
        self.args = {}
        self.args['root_dir'] = 'SID'
        self.args['crop_per_image'] = 8
        self.args['crop_size'] = 512
        self.args['ori'] = False
        self.args['iso'] = None
        self.args['dgain'] = None
        self.args['params'] = None
        self.args['lock_wb'] = False
        self.args['gpu_preprocess'] = False
        self.args['dstname'] = 'SID'
        self.args['mode'] = 'train'
        self.args['wp'] = 16383
        self.args['bl'] = 512
        self.args['command'] = ''

    def initialization(self):
        # 获取数据地址
        self.suffix = 'npy'
        self.root_dir = self.args['root_dir']
        self.mode = self.args['mode']
        self.dataset_file = f'SID_{self.args["mode"]}.info' 
        with open(f"infos/{self.dataset_file}", 'rb') as info_file:
            self.infos = pkl.load(info_file)
            print(f'>> Successfully load "{self.dataset_file}" (Length: {len(self.infos)})')
        self.datapath = [info['long'] for info in self.infos]
        self.names = [info['name'] for info in self.infos]
        self.buffer = [None] * len(self.infos)
        if 'cache' in self.args['command']:
            log(f'Loading {len(self.infos)} crops!!!')
            self.buffer = [dataload(path) for path in tqdm(self.datapath)]
        self.length = len(self.infos)
        self.sigma = -1
        self.get_shape()

    def __len__(self):
        return self.length

    def get_shape(self):
        self.H, self.W = self.args['H'], self.args['W'] 
        self.C = 3
        self.h = self.H // 2
        self.w = self.W // 2
        self.c = 4
    
    def data_aug(self, data, mode=0):
        if mode == 0: return data
        rot = mode % 4
        flip = mode // 4
        data = np.rot90(data, k=rot, axes=(-2, -1))
        if flip:
            data = data[..., ::-1]
        return data
    
    def init_random_crop_point(self, mode='non-overlapped', raw_crop=False):
        self.h_start = []
        self.w_start = []
        self.h_end = []
        self.w_end = []
        self.aug = np.random.randint(8, size=self.args['crop_per_image'])
        h, w = self.h, self.w
        if raw_crop:
            h, w = self.H, self.W
        if mode == 'non-overlapped':
            nh = h // self.args["patch_size"]
            nw = w // self.args["patch_size"]
            h_start = np.random.randint(0, h - nh*self.args["patch_size"] + 1)
            w_start = np.random.randint(0, w - nw*self.args["patch_size"] + 1)
            for i in range(nh):
                for j in range(nw):
                    self.h_start.append(h_start + i * self.args["patch_size"])
                    self.w_start.append(w_start + j * self.args["patch_size"])
                    self.h_end.append(h_start + (i+1) * self.args["patch_size"])
                    self.w_end.append(w_start + (j+1) * self.args["patch_size"])

        else: # random_crop
            for i in range(self.args['crop_per_image']):
                h_start = np.random.randint(0, h - self.args["patch_size"] + 1)
                w_start = np.random.randint(0, w - self.args["patch_size"] + 1)
                self.h_start.append(h_start)
                self.w_start.append(w_start)
                self.h_end.append(h_start + self.args["patch_size"])
                self.w_end.append(w_start + self.args["patch_size"])

    def random_crop(self, img):
        # 本函数用于将numpy随机裁剪成以crop_size为边长的方形crop_per_image等份
        c, h, w = img.shape
        # 创建空numpy做画布, [crops, h, w]
        crops = np.empty((self.args["crop_per_image"], c, self.args["patch_size"], self.args["patch_size"]), dtype=np.float32)
        # 往空tensor的通道上贴patchs
        for i in range(self.args["crop_per_image"]):
            crop = img[:, self.h_start[i]:self.h_end[i], self.w_start[i]:self.w_end[i]]
            # crop = self.data_aug(crop, mode=self.aug[i]) # 会导致格子纹
            crops[i] = crop

        return crops

    def __getitem__(self, idx):
        data = {}
        # 读取数据
        data['wb'] = self.infos[idx]['wb']
        data['ccm'] = self.infos[idx]['ccm']
        data['name'] = self.infos[idx]['name']

        if self.buffer[idx] is None:
            self.buffer[idx] = dataload(self.datapath[idx])
        hr_raw = (self.buffer[idx].astype(np.float32) - self.args['bl']) / (self.args['wp'] - self.args['bl'])
        # BayerAug, 这个旋转是用来改变bayer模式的
        data['pattern'] = np.random.randint(4) if self.args["mode"] == 'train' else idx%4
        hr_raw = np.rot90(hr_raw, k=data['pattern'], axes=(-2,-1))
        hr_raw = bayer2rggb(hr_raw).clip(0, 1).transpose(2,0,1)
        # 模拟VST后的gt数值范围 y=sqrt(x+3/8)≈sqrt(x)
        data['vst_aug'] = True if np.random.randint(2) else False
        hr_raw = hr_raw ** 0.5 if data['vst_aug'] else hr_raw

        if self.args["mode"] == 'train':
            # 随机裁剪成crop_per_image份
            self.init_random_crop_point(mode=self.args['croptype'], raw_crop=False)
            if data['pattern'] % 2:
                self.h_start, self.h_end, self.w_start, self.w_end = self.w_start, self.w_end, self.h_start, self.h_end
            hr_crops = self.random_crop(hr_raw)
        else:
            setup_seed(idx)
            hr_crops = hr_raw[None,:]

        lr_shape = hr_crops.shape
        
        if self.args["lock_wb"] is False and np.random.randint(2):
            rgb_gain, red_gain, blue_gain = random_gains()
            red_gain = data['wb'][0] / red_gain.numpy()
            blue_gain = data['wb'][2] / blue_gain.numpy()
            hr_crops *= rgb_gain.numpy()
            hr_crops[:,0] = hr_crops[:,0] * red_gain
            hr_crops[:,2] = hr_crops[:,2] * blue_gain
            data['wb'][0] = red_gain
            data['wb'][2] = blue_gain
        lr_crops = hr_crops.copy()

        # 人工加噪声
        if self.args['gpu_preprocess'] is False:
            if self.args['mode'] == 'train':
                lower, upper = np.log(self.args['sigma_min']), np.log(self.args['sigma_max'])
                data['sigma'] = np.exp(np.random.rand()*(upper-lower)+lower) / 255.
            else:
                data['sigma'] = self.sigma
                setup_seed(idx)
            noise = np.random.randn(*lr_crops.shape) * data['sigma']
            lr_crops += noise
        
        data["lr"] = np.ascontiguousarray(lr_crops)
        data["hr"] = np.ascontiguousarray(hr_crops)

        if self.args['clip']:
            data["lr"] = lr_crops.clip(0, 1)
            data["hr"] = hr_crops.clip(0, 1)
        
        return data

# Unprocess Synthetic Dataset(sRGB->Raw)
class RGB_Img2Raw_Dataset(Dataset):
    def __init__(self, args=None):
        super().__init__()
        self.default_args()
        if args is not None:
            for key in args:
                self.args[key] = args[key]
        self.initialization()

    def default_args(self):
        self.args = {}
        self.args['root_dir'] = 'YOND'
        self.args['crop_size'] = 256
        self.args['ori'] = False
        self.args['iso'] = None
        self.args['dgain'] = None
        self.args['params'] = None
        self.args['lock_wb'] = False
        self.args['gpu_preprocess'] = False
        self.args['dstname'] = 'YOND'
        self.args['mode'] = 'train'
        self.args['command'] = ''

    def initialization(self):
        # 获取数据地址
        self.suffix = 'npy'
        self.root_dir = self.args['root_dir']
        self.mode = self.args['mode']
        self.data_dir = f"{self.root_dir}/{self.mode}"
        if self.mode == 'train':
            self.data_dir += f'_{self.args["subname"]}'
        self.datapath = sorted(glob.glob(f'{self.data_dir}/*.{self.suffix}'))
        self.names = [os.path.basename(path)[:-4] for path in self.datapath]
        self.infos = [{'name':name, 'path':path} for name, path in zip(self.names, self.datapath)]
        self.buffer = [None] * len(self.infos)
        if 'cache' in self.args['command']:
            log(f'Loading {len(self.infos)} crops!!!')
            self.buffer = [dataload(path) for path in tqdm(self.datapath)]
        self.length = len(self.infos)
        self.sigma = -1
        self.get_shape()
        log(f'Successfully cache {self.length} npy data!!!')

    def __len__(self):
        return self.length

    def get_shape(self):
        self.H, self.W = self.args['H'], self.args['W'] 
        self.C = 3
        self.h = self.H // 2
        self.w = self.W // 2
        self.c = 4
    
    def data_aug(self, data, mode=0):
        if mode == 0: return data
        rot = mode % 4
        flip = mode // 4
        data = np.rot90(data, k=rot, axes=(-2, -1))
        if flip:
            data = data[..., ::-1]
        return data
    
    def __getitem__(self, idx):
        data = {}
        # 读取数据
        data['name'] = self.infos[idx]['name']
        if self.buffer[idx] is None:
            self.buffer[idx] = dataload(self.datapath[idx])
        max_val = 255. if self.buffer[idx].dtype == np.uint8 else 65535.
        hr_imgs = self.buffer[idx].astype(np.float32) / max_val
        if self.args["mode"] == 'train':
            data['aug_id1'] = np.random.randint(8)
            self.data_aug(hr_imgs, data['aug_id1'])
        else:
            setup_seed(idx)
        hr_crops = hr_imgs
    
        # RAW需要复杂的unproces
        lr_shape = hr_crops.shape
        hr_crops = torch.from_numpy(hr_crops)

        hr_crops, metadata = unprocess(hr_crops, lock_wb=self.args["lock_wb"], use_gpu=self.args['gpu_preprocess'])
        data['wb'] = np.array([metadata['red_gain'].item(), 1., metadata['blue_gain'].item()])
        data['ccm'] = metadata['cam2rgb'].numpy()
        hr_crops = mosaic(hr_crops).numpy() # rgbg
        # 这个旋转是用来改变bayer模式的
        if 'no_bayeraug' in self.args["command"]:
            data['pattern'] = 0
        else:
            if self.args["mode"] == 'train':
                data['pattern'] = np.random.randint(4)
            else:
                data['pattern'] = idx%4
        hr_crops = bayer_aug(hr_crops, k=data['pattern'])
        data['vst_aug'] = False
        hr_crops = hr_crops ** 0.5 if data['vst_aug'] else hr_crops
        # [crops,h,w,c] -> [crops,c,h,w]
        hr_crops = hr_crops.transpose(2,0,1)
        lr_crops = hr_crops.copy()

        # 人工加噪声
        if self.args['gpu_preprocess'] is False:
            if self.args['mode'] == 'train':
                lower, upper = np.log(self.args['sigma_min']), np.log(self.args['sigma_max'])
                data['sigma'] = np.exp(np.random.rand()*(upper-lower)+lower) / 255.
            else:
                data['sigma'] = self.sigma
                setup_seed(idx)
            noise = np.random.randn(*lr_crops.shape) * data['sigma']
            lr_crops += noise
        
        data["lr"] = np.ascontiguousarray(lr_crops)
        data["hr"] = np.ascontiguousarray(hr_crops)

        if self.args['clip']:
            data["lr"] = lr_crops.clip(0, 1)
            data["hr"] = hr_crops.clip(0, 1)
        
        return data

# Synthetic Dataset(sRGB)
class RGB_Img_Dataset(Dataset):
    def __init__(self, args=None):
        super().__init__()
        self.default_args()
        if args is not None:
            for key in args:
                self.args[key] = args[key]
        self.initialization()

    def default_args(self):
        self.args = {}
        self.args['root_dir'] = 'YOND'
        self.args['crop_size'] = 256
        self.args['ori'] = False
        self.args['iso'] = None
        self.args['dgain'] = None
        self.args['params'] = None
        self.args['lock_wb'] = False
        self.args['gpu_preprocess'] = False
        self.args['dstname'] = 'YOND'
        self.args['mode'] = 'train'
        self.args['command'] = ''

    def initialization(self):
        # 获取数据地址
        self.suffix = 'npy'
        self.root_dir = self.args['root_dir']
        self.mode = self.args['mode']
        self.data_dir = f"{self.root_dir}/{self.mode}"
        if self.mode == 'train':
            self.data_dir += f'_{self.args["subname"]}'
        self.datapath = sorted(glob.glob(f'{self.data_dir}/*.{self.suffix}'))
        self.names = [os.path.basename(path)[:-4] for path in self.datapath]
        self.infos = [{'name':name, 'path':path} for name, path in zip(self.names, self.datapath)]
        self.buffer = [None] * len(self.infos)
        if 'cache' in self.args['command']:
            log(f'Loading {len(self.infos)} crops!!!')
            self.buffer = [dataload(path) for path in tqdm(self.datapath)]
        self.length = len(self.infos)
        self.sigma = -1
        self.get_shape()
        log(f'Successfully cache {self.length} npy data!!!')

    def __len__(self):
        return self.length

    def get_shape(self):
        self.H, self.W = self.args['H'], self.args['W'] 
        self.C = 3
        self.h = self.H // 2
        self.w = self.W // 2
        self.c = 4
    
    def data_aug(self, data, mode=0):
        if mode == 0: return data
        rot = mode % 4
        flip = mode // 4
        data = np.rot90(data, k=rot, axes=(-2, -1))
        if flip:
            data = data[..., ::-1]
        return data
    
    def __getitem__(self, idx):
        data = {}
        # 读取数据
        data['name'] = self.infos[idx]['name']
        if self.buffer[idx] is None:
            self.buffer[idx] = dataload(self.datapath[idx])
        max_val = 255. if self.buffer[idx].dtype == np.uint8 else 65535.
        hr_imgs = self.buffer[idx].astype(np.float32) / max_val
        if self.args["mode"] == 'train':
            data['aug_id1'] = np.random.randint(8)
            self.data_aug(hr_imgs, data['aug_id1'])
        else:
            setup_seed(idx)
        hr_crops = hr_imgs.transpose(2,0,1)
        lr_crops = hr_crops.copy()

        # 人工加噪声
        if self.args['gpu_preprocess'] is False:
            if self.args['mode'] == 'train':
                # lower, upper = np.log(self.args['sigma_min']), np.log(self.args['sigma_max'])
                # data['sigma'] = np.exp(np.random.rand()*(upper-lower)+lower) / 255.
                lower, upper = self.args['sigma_min'], self.args['sigma_max']
                data['sigma'] = (np.random.rand()*(upper-lower)+lower) / 255.
            else:
                data['sigma'] = self.sigma
                setup_seed(idx)
            noise = np.random.randn(*lr_crops.shape) * data['sigma']
            lr_crops += noise
        
        data["lr"] = np.ascontiguousarray(lr_crops)
        data["hr"] = np.ascontiguousarray(hr_crops)

        if self.args['clip']:
            data["lr"] = lr_crops.clip(0, 1)
            data["hr"] = hr_crops.clip(0, 1)
        
        return data

# Unprocess Synthetic Dataset(sRGB->Raw)
class DIV2K_Img2Raw_Dataset(Dataset):
    def __init__(self, args=None):
        super().__init__()
        self.default_args()
        if args is not None:
            for key in args:
                self.args[key] = args[key]
        self.initialization()

    def default_args(self):
        self.args = {}
        self.args['root_dir'] = 'DIV2K'
        self.args['crop_size'] = 256
        self.args['ori'] = False
        self.args['iso'] = None
        self.args['dgain'] = None
        self.args['params'] = None
        self.args['lock_wb'] = False
        self.args['gpu_preprocess'] = False
        self.args['dstname'] = 'DIV2K'
        self.args['mode'] = 'train'
        self.args['command'] = ''

    def initialization(self):
        # 获取数据地址
        self.suffix = 'npy'
        self.root_dir = self.args['root_dir']
        self.mode = self.args['mode']
        self.data_dir = f"{self.root_dir}/npy/{self.mode}"
        self.datapath = sorted(glob.glob(f'{self.data_dir}/*.{self.suffix}'))
        self.names = [os.path.basename(path)[:-4] for path in self.datapath]
        self.infos = [{'name':name, 'path':path} for name, path in zip(self.names, self.datapath)]
        self.buffer = [None] * len(self.infos)
        if 'cache' in self.args['command']:
            log(f'Loading {len(self.infos)} crops!!!')
            self.buffer = [dataload(path) for path in tqdm(self.datapath)]
        self.length = len(self.infos)
        self.sigma = -1
        self.get_shape()
        log(f'Successfully cache {self.length} npy data!!!')

    def __len__(self):
        return self.length

    def get_shape(self):
        self.H, self.W = self.args['H'], self.args['W'] 
        self.C = 3
        self.h = self.H // 2
        self.w = self.W // 2
        self.c = 4
    
    def data_aug(self, data, mode=0):
        if mode == 0: return data
        rot = mode % 4
        flip = mode // 4
        data = np.rot90(data, k=rot, axes=(-2, -1))
        if flip:
            data = data[..., ::-1]
        return data
    
    def __getitem__(self, idx):
        data = {}
        # 读取数据
        data['name'] = self.infos[idx]['name']
        if self.buffer[idx] is None:
            self.buffer[idx] = dataload(self.datapath[idx])
        hr_imgs = self.buffer[idx].astype(np.float32) / 255.
        if self.args["mode"] == 'train':
            data['aug_id1'] = np.random.randint(8)
            self.data_aug(hr_imgs, data['aug_id1'])
        else:
            setup_seed(idx)
        hr_crops = hr_imgs
    
        # RAW需要复杂的unproces
        lr_shape = hr_crops.shape
        hr_crops = torch.from_numpy(hr_crops)

        hr_crops, metadata = unprocess(hr_crops, lock_wb=self.args["lock_wb"], use_gpu=self.args['gpu_preprocess'])
        data['wb'] = np.array([metadata['red_gain'].item(), 1., metadata['blue_gain'].item()])
        data['ccm'] = metadata['cam2rgb'].numpy()
        hr_crops = mosaic(hr_crops).numpy() # rgbg
        # 这个旋转是用来改变bayer模式的
        # if self.args["mode"] == 'train':
        data['pattern'] = np.random.randint(4) if self.args["mode"] == 'train' else idx%4
        hr_crops = bayer_aug(hr_crops, k=data['pattern'])
        data['vst_aug'] = False
        hr_crops = hr_crops ** 0.5 if data['vst_aug'] else hr_crops
        # [crops,h,w,c] -> [crops,c,h,w]
        hr_crops = hr_crops.transpose(2,0,1)
        lr_crops = hr_crops.copy()

        # 人工加噪声
        if self.args['gpu_preprocess'] is False:
            if self.args['mode'] == 'train':
                lower, upper = np.log(self.args['sigma_min']), np.log(self.args['sigma_max'])
                data['sigma'] = np.exp(np.random.rand()*(upper-lower)+lower) / 255.
            else:
                data['sigma'] = self.sigma
                setup_seed(idx)
            noise = np.random.randn(*lr_crops.shape) * data['sigma']
            lr_crops += noise
        
        data["lr"] = np.ascontiguousarray(lr_crops)
        data["hr"] = np.ascontiguousarray(hr_crops)

        if self.args['clip']:
            data["lr"] = lr_crops.clip(0, 1)
            data["hr"] = hr_crops.clip(0, 1)
        
        return data

# Unprocess Synthetic Dataset(sRGB->Raw3c1n)
class RGB_Img2Raw3c1n_Dataset(Dataset):
    def __init__(self, args=None):
        super().__init__()
        self.default_args()
        if args is not None:
            for key in args:
                self.args[key] = args[key]
        self.initialization()

    def default_args(self):
        self.args = {}
        self.args['root_dir'] = 'YOND'
        self.args['crop_size'] = 256
        self.args['ori'] = False
        self.args['iso'] = None
        self.args['dgain'] = None
        self.args['params'] = None
        self.args['lock_wb'] = False
        self.args['gpu_preprocess'] = False
        self.args['dstname'] = 'YOND'
        self.args['mode'] = 'train'
        self.args['command'] = ''

    def initialization(self):
        # 获取数据地址
        self.suffix = 'npy'
        self.root_dir = self.args['root_dir']
        self.mode = self.args['mode']
        self.data_dir = f"{self.root_dir}/{self.mode}"
        if self.mode == 'train':
            self.data_dir += f'_{self.args["subname"]}'
        self.datapath = sorted(glob.glob(f'{self.data_dir}/*.{self.suffix}'))
        self.names = [os.path.basename(path)[:-4] for path in self.datapath]
        self.infos = [{'name':name, 'path':path} for name, path in zip(self.names, self.datapath)]
        self.buffer = [None] * len(self.infos)
        if 'cache' in self.args['command']:
            log(f'Loading {len(self.infos)} crops!!!')
            self.buffer = [dataload(path) for path in tqdm(self.datapath)]
        self.length = len(self.infos)
        self.sigma = -1
        self.get_shape()
        log(f'Successfully cache {self.length} npy data!!!')

    def __len__(self):
        return self.length

    def get_shape(self):
        self.H, self.W = self.args['H'], self.args['W'] 
        self.C = 3
        self.h = self.H // 2
        self.w = self.W // 2
        self.c = 4
    
    def data_aug(self, data, mode=0):
        if mode == 0: return data
        rot = mode % 4
        flip = mode // 4
        data = np.rot90(data, k=rot, axes=(-2, -1))
        if flip:
            data = data[..., ::-1]
        return data
    
    def __getitem__(self, idx):
        data = {}
        # 读取数据
        data['name'] = self.infos[idx]['name']
        if self.buffer[idx] is None:
            self.buffer[idx] = dataload(self.datapath[idx])
        max_val = 255. if self.buffer[idx].dtype == np.uint8 else 65535.
        hr_imgs = self.buffer[idx].astype(np.float32) / max_val
        if self.args["mode"] == 'train':
            data['aug_id1'] = np.random.randint(8)
            self.data_aug(hr_imgs, data['aug_id1'])
        else:
            setup_seed(idx)
        hr_crops = hr_imgs
    
        # RAW需要复杂的unproces
        lr_shape = hr_crops.shape
        hr_crops = torch.from_numpy(hr_crops)

        hr_crops, metadata = unprocess(hr_crops, lock_wb=self.args["lock_wb"], use_gpu=self.args['gpu_preprocess'])
        data['wb'] = np.array([metadata['red_gain'].item(), 1., metadata['blue_gain'].item()])
        data['ccm'] = metadata['cam2rgb'].numpy()
        hr_crops = mosaic(hr_crops).numpy() # rggb
        # [crops,h,w,c] -> [crops,c,h,w]
        hr_crops = hr_crops.transpose(2,0,1)
        lr_crops = hr_crops.copy()

        # 人工加噪声
        if self.args['gpu_preprocess'] is False:
            if self.args['mode'] == 'train':
                lower, upper = np.log(self.args['sigma_min']), np.log(self.args['sigma_max'])
                data['sigma'] = np.exp(np.random.rand()*(upper-lower)+lower) / 255.
            else:
                data['sigma'] = self.sigma
                setup_seed(idx)
            noise = np.random.randn(*lr_crops.shape[-2:]) * data['sigma']
            lr_crops[2] += noise
        
        data["lr"] = np.ascontiguousarray(lr_crops)
        data["hr"] = np.ascontiguousarray(hr_crops)

        if self.args['clip']:
            data["lr"] = lr_crops.clip(0, 1)
            data["hr"] = hr_crops.clip(0, 1)
        
        return data


# Unprocess Synthetic Dataset(sRGB->Raw)
class DIV2K_PG_Dataset(DIV2K_Img2Raw_Dataset):
    def __init__(self, args=None):
        super().__init__(args)
        self.noise_params= {
            'Kmin': -2.5, 'Kmax': 3.5, 'lam': 0.102, 'q': 1/(2**10), 'wp': 1023, 'bl': 64,
            'sigTLk': 0.85187, 'sigTLb': 0.07991, 'sigTLsig': 0.02921,
            'sigRk': 0.87611,  'sigRb': -2.11455, 'sigRsig': 0.03274,
            'sigGsk': 0.85187, 'sigGsb': 0.67991, 'sigGssig': 0.02921,
        }
        self.p = self.get_noise_params()

    def get_noise_params(self):
        p = self.noise_params
        log_K = np.random.uniform(low=p['Kmin'], high=p['Kmax'])
        mu_Gs = (p['sigGsk'] + np.random.uniform(-0.2, 0.2))*log_K + (p['sigGsb'] + np.random.uniform(-1, 1))
        log_sigGs = np.random.normal(loc=mu_Gs, scale=p['sigGssig'])
        K = np.exp(log_K)
        sigma = np.exp(log_sigGs)
        scale = p['wp'] - p['bl']
        self.p = {'K':K, 'sigma':sigma, 'beta1':K/scale, 'beta2':(sigma/scale)**2, 
                'wp':p['wp'], 'bl':p['bl'], 'scale':p['wp']-p['bl']}
        return self.p
    
    def __getitem__(self, idx):
        data = {}
        # 读取数据
        data['name'] = self.infos[idx]['name']
        if self.buffer[idx] is None:
            self.buffer[idx] = dataload(self.datapath[idx])
        hr_imgs = self.buffer[idx].astype(np.float32) / 255.
        if self.args["mode"] == 'train':
            data['aug_id1'] = np.random.randint(8)
            # self.data_aug(hr_imgs, data['aug_id1'])
        else:
            setup_seed(idx)
        hr_crops = hr_imgs
    
        # RAW需要复杂的unproces
        lr_shape = hr_crops.shape
        hr_crops = torch.from_numpy(hr_crops)

        hr_crops, metadata = unprocess(hr_crops, lock_wb=self.args["lock_wb"], use_gpu=False)
        data['wb'] = np.array([metadata['red_gain'].item(), 1., metadata['blue_gain'].item()])
        data['ccm'] = metadata['cam2rgb'].numpy()
        hr_crops = mosaic(hr_crops).numpy() # rgbg
        # 这个旋转是用来改变bayer模式的
        # if self.args["mode"] == 'train':
        data['pattern'] = np.random.randint(4) if self.args["mode"] == 'train' else idx%4
        hr_crops = bayer_aug(hr_crops, k=data['pattern'])
        lr_crops = hr_crops.copy()

        # 人工加噪声
        if self.args['mode'] == 'train':
            p = self.get_noise_params()
            data.update(p)
        else:
            data.update(self.p)

        if self.args['gpu_preprocess'] is False:
            lr_crops = np.random.poisson(lr_crops/p['beta1'])*p['beta1'] + np.random.randn(*lr_crops.shape) * data['beta2']**0.5
            if 'est' in self.args['command']:
                # 对噪图求blur
                k = 19
                lr_crops = lr_crops
                hr_crops = hr_crops
                lr_blur = cv2.blur(lr_crops, (k, k))
                hr_blur = cv2.blur(hr_crops, (k, k))
                lr_std = stdfilt(lr_crops, k)
                hr_std = stdfilt(hr_crops, k)
                hr_target = (p['beta1'] * hr_blur + p['beta2']) ** 0.5

                var = lr_std**2
                mean = lr_blur
                th, percent = get_threshold(hr_std)
                mask = (hr_std <= th)
                # 按阈值分割图像平滑区域与非平滑区域
                if var[mask].size > 0:
                    var, mean = var[mask], mean[mask]
                else:
                    mask = (hr_std <= hr_std.max())

                data['th'] = th
                data['hr_mask'] = mask
                data['lr_rggb'] = lr_crops
                data['hr_rggb'] = hr_crops
                data['lr_std'] = lr_std
                data['hr_std'] = hr_std
                data['lr_blur'] = lr_blur
                data['hr_blur'] = hr_blur
                data['lr'] = np.concatenate([lr_std, lr_blur, lr_crops], axis=-1)
                data['hr'] = hr_target
        else:
            data["lr"] = np.ascontiguousarray(lr_crops)
            data["hr"] = np.ascontiguousarray(hr_crops)

        for key in data:
            if 'lr' in key or 'hr' in key:
                data[key] = data[key].transpose(2,0,1)

        if self.args['clip']:
            data['lr'] = lr_crops.clip(0, 1)
            data['hr'] = hr_crops.clip(0, 1)
        
        return data

# SIDD Paired Real Data
class SIDD_Dataset(Dataset):
    def __init__(self, args=None):
        super().__init__()
        self.default_args()
        if args is not None:
            for key in args:
                self.args[key] = args[key]
        self.initialization()

    def default_args(self):
        self.args = {}
        self.args['root_dir'] = '/data/fenghansen/datasets/SIDD'
        self.args['params'] = None
        self.args['lock_wb'] = False
        self.args['gpu_preprocess'] = False
        self.args['dstname'] = 'SIDD'
        self.args['mode'] = 'eval'
        self.args['clip'] = 'True'
        self.args['wp'] = 1023
        self.args['bl'] = 64
        self.args['patch_size'] = 256
        self.args['H'] = 256
        self.args['W'] = 256
        self.args['command'] = ''

    def initialization(self):
        # 获取数据地址
        self.suffix = 'npy'
        self.root_dir = self.args['root_dir']
        self.mode = self.args['mode']
        if self.mode == 'train':
            self.data_dir = f'{self.root_dir}/SIDD_Medium_Raw/data'
        else:
            self.data_dir = f'{self.root_dir}/SIDD_Benchmark_Data'
            if self.mode == 'eval':
                self.lr_data = sio.loadmat(f'{self.root_dir}/SIDD_Validation_Raw/ValidationNoisyBlocksRaw.mat')['ValidationNoisyBlocksRaw']
                self.hr_data = sio.loadmat(f'{self.root_dir}/SIDD_Validation_Raw/ValidationGtBlocksRaw.mat')['ValidationGtBlocksRaw']
            else:
                self.lr_data = sio.loadmat(f'{self.root_dir}/SIDD_Validation_Raw/BenchmarkNoisyBlocksRaw.mat')['BenchmarkNoisyBlocksRaw']
                self.hr_data = None
            self.pos = sio.loadmat(f'{self.root_dir}/SIDD_Validation_Raw/BenchmarkBlocks32.mat')['BenchmarkBlocks32']
        self.names = os.listdir(self.data_dir)
        self.datapaths = sorted(glob.glob(f'{self.data_dir}//*/*_010.MAT'))
        self.metapaths = sorted([path for path in self.datapaths if 'META' in path])
        self.lr_paths = sorted([path for path in self.datapaths if 'NOISY' in path])
        self.hr_paths = sorted([path for path in self.datapaths if 'GT' in path])
        self.length = len(self.names)
        self.infos = [None] * self.length
        for i in range(self.length):
            metadata = read_metadata(dataload(self.metapaths[i]))
            self.infos[i] = {
                'name': self.names[i],
                'lr_path': self.lr_paths[i],
                'hr_path': self.hr_paths[i] if len(self.hr_paths)>0 else None,
                'metadata': metadata,
            }
        self.sigma = -1
        self.get_shape()
        log(f'Successfully load {self.length} data!!! ({self.mode})')

    def __len__(self):
        return self.length

    def get_shape(self):
        self.H, self.W = self.args['H'], self.args['W'] 
        self.C = 3
        self.h = self.H // 2
        self.w = self.W // 2
        self.c = 4
    
    def data_aug(self, data, mode=0):
        if mode == 0: return data
        rot = mode % 4
        flip = mode // 4
        data = np.rot90(data, k=rot, axes=(-2, -1))
        if flip:
            data = data[..., ::-1]
        return data
    
    def __getitem__(self, idx):
        data = {}
        # 读取数据
        data['name'] = self.infos[idx]['name']
        data['meta'] = self.infos[idx]['metadata']
        data['lr_path_full'] = self.infos[idx]['lr_path']
        data['hr_path_full'] = self.infos[idx]['hr_path']
        data['wb'] = data['meta']['wb']
        data['cfa'] = data['meta']['bayer_2by2']
        data['ccm'] = data['meta']['cst2']
        data['iso'] = data['meta']['iso']
        data['reg'] = (data['meta']['beta1'], data['meta']['beta2'])

        if self.args["mode"] == 'train':
            data['lr'] = dataload(data['lr_path_full'])
            data['hr'] = dataload(data['hr_path_full'])
            # raise NotImplementedError
        else:
            data['lr'] = self.lr_data[idx]
            if self.mode == 'eval':
                data['hr'] = self.hr_data[idx]
        
        return data
    
class LRID_Dataset(Dataset):
    def __init__(self, args=None):
        super().__init__()
        self.default_args()
        if args is not None:
            for key in args:
                self.args[key] = args[key]
        self.initialization()
    
    def default_args(self):
        self.args = {}
        self.args['root_dir'] = 'LRID/'
        self.args['suffix'] = 'dng'
        self.args['dgain'] = 1
        self.args['dstname'] = 'indoor_x5'
        self.args['camera_type'] = 'IMX686'
        self.args['params'] = None
        self.args['mode'] = 'eval'
        self.args['GT_type'] = 'GT_align_ours'
        self.args['command'] = ''
        self.args['H'] = 3472
        self.args['W'] = 4624
        self.args['wp'] = 1023
        self.args['bl'] = 64
        self.args['clip'] = False

    def initialization(self):
        # 获取数据地址
        self.suffix = 'dng'
        self.change_eval_ratio(ratio=1)
        self.iso = 6400

    def __len__(self):
        return self.length
    
    def get_shape(self):
        self.shape = self.templet_raw.raw_image_visible.shape
        self.H, self.W = self.shape
        self.bl_all = np.array(self.templet_raw.black_level_per_channel)
        if np.mean(self.bl_all - self.bl_all[0]) != 0:
            warnings.warn(f'4 channel have different black level!!! ({self.bl_all})')
        self.bl = self.bl_all[0]
        self.wp = self.templet_raw.white_level

    def change_eval_ratio(self, ratio):
        self.ratio = ratio
        self.infos_gt = []
        self.infos_short = []
        for dstname in self.args['dstname']:
            with open(f"infos/{dstname}_{self.args['GT_type']}.info", 'rb') as info_file:
                info = pkl.load(info_file)
                eval_id = self.get_eval_id(dstname)
                for idx in eval_id:
                    self.infos_gt.append(info[idx])
            with open(f'infos/{dstname}_short.info', 'rb') as info_file:
                info = pkl.load(info_file)[ratio]
                eval_id = self.get_eval_id(dstname)
                for idx in eval_id:
                    self.infos_short.append(info[idx])
        self.infos = self.infos_gt
        for i in range(len(self.infos)):
            self.infos[i]['hr'] = self.infos[i]['data']
            self.infos[i]['lr'] = self.infos_short[i]
            del self.infos[i]['data']
        print(f'>> Successfully load infos.pkl (Length: {len(self.infos)})')
        self.iso = 6400
        self.length = len(self.infos)
        self.templet_raw_path = self.infos[0]['lr']['data'][0]
        self.templet_raw = rawpy.imread(self.templet_raw_path)
        self.get_shape()
    
    def get_eval_id(self, dstname='indoor_x5'):
        if dstname == 'indoor_x5':
            eval_ids = [4,14,25,41,44,51,52,53,58]
        elif dstname == 'indoor_x3':
            eval_ids = []#[0,6,15]
        elif dstname == 'outdoor_x5':
            eval_ids = [1,2,5]
        elif dstname == 'outdoor_x3':
            eval_ids = [9,21,22,32,44,51]
        else:
            eval_ids = []
        return eval_ids

    def __getitem__(self, idx):
        data = {}
        # dataload
        hr_raw = np.array(dataload(self.infos[idx]['hr'])).reshape(self.H,self.W)
        lr_raw = np.array(dataload(self.infos[idx]['lr']['data'][0])).reshape(self.H,self.W)
        data["hr"] = (hr_raw.astype(np.float32) - self.bl) / (self.wp - self.bl)
        # lr_raw = bayer2rggb(lr_raw.astype(np.float32)) - self.bl_all.reshape(1,1,-1)
        data["lr"] = (lr_raw.astype(np.float32) - self.bl) * self.ratio / (self.wp - self.bl)

        data['name'] = f"{self.infos[idx]['name']}_x{self.ratio:02d}"
        data['ratio'] = self.ratio
        data['ccm'] = self.infos[idx]['ccm']
        data['wb'] = self.infos[idx]['wb']
        data['cfa'] = 'rggb'
        data['ISO'] = self.iso
        data['ExposureTime'] = self.infos[idx]['lr']['metadata'][0]['ExposureTime'] * 1000

        if self.args['clip']:
            data["hr"] = data["hr"].clip(0,1)
            data["lr"] = data["lr"].clip(0,1)

        return data

class ELD_Full_Dataset(Dataset):
    def __init__(self, args=None):
        super().__init__()
        self.default_args()
        if args is not None:
            for key in args:
                self.args[key] = args[key]
        self.initialization()
    
    def default_args(self):
        self.args = {}
        self.args['root_dir'] = 'ELD/'
        self.args['ratio'] = 1
        self.args['dstname'] = 'ELD'
        self.args['params'] = None
        self.args['mode'] = 'eval'
        self.args['command'] = ''
        self.args['wp'] = 16383
        self.args['bl'] = 512
        self.args['clip'] = False

    def initialization(self):
        # 获取数据地址
        self.suffix = {'CanonEOS70D':'CR2', 'CanonEOS700D':'CR2', 'NikonD850':'nef', 'SonyA7S2':'ARW'}
        self.infos_all = {'CanonEOS70D':[], 'CanonEOS700D':[], 'NikonD850':[], 'SonyA7S2':[]}
        iso_list = [800, 1600, 3200]
        ratio_list = [1,10,100,200]
        hr_ids = np.array([1, 6, 11, 16])
        for camera_type in self.infos_all:
            sub_dir = f'{self.args["root_dir"]}/{camera_type}'
            for scene in range(1,11):
                for iso_id, iso in enumerate(iso_list):
                    for ratio_id, ratio in enumerate(ratio_list):
                        lr_id = iso_id*5 + ratio_id + 2
                        ind = np.argmin(np.abs(lr_id - hr_ids))
                        hr_id = hr_ids[ind]
                        name = f'IMG_{lr_id:04d}.{self.suffix[camera_type]}'
                        hr_name = f'IMG_{hr_id:04d}.{self.suffix[camera_type]}'
                        self.infos_all[camera_type].append({
                            'cam': camera_type,
                            'name': f'{camera_type}_{scene:02d}_{name[:-4]}',
                            'hr': f'{sub_dir}/scene-{scene}/{hr_name}',
                            'lr': f'{sub_dir}/scene-{scene}/{name}',
                            'iso': iso,
                            'ratio': ratio,
                        })
        self.change_eval_ratio('SonyA7S2', ratio=1)

    def __len__(self):
        return self.length
    
    def change_eval_ratio(self, cam='SonyA7S2', ratio=1, iso_list=None):
        if iso_list is None:
            iso_list = [800, 1600, 3200]
        self.infos = []
        for i in range(len(self.infos_all[cam])):
            if self.infos_all[cam][i]['iso'] in iso_list and self.infos_all[cam][i]['ratio'] == ratio:
                self.infos.append(self.infos_all[cam][i])
        self.length = len(self.infos)
        self.ratio = ratio
        self.templet_raw = rawpy.imread(self.infos[0]['lr'])
        self.get_shape()
        log(f'Eval change to {cam} (length:{self.length}): ratio={ratio}, iso_list={iso_list}')

    def get_shape(self):
        self.shape = self.templet_raw.raw_image_visible.shape
        self.H, self.W = self.shape
        self.bl = np.array(self.templet_raw.black_level_per_channel)
        if np.mean(self.bl - self.bl[0]) != 0:
            warnings.warn(f'4 channel have different black level!!! ({self.bl})')
        self.bl = self.bl[0]
        self.wp = self.templet_raw.white_level

    def __getitem__(self, idx):
        data = {}
        # dataload
        hr_raw = np.array(dataload(self.infos[idx]['hr'])).reshape(self.H,self.W)
        lr_raw = np.array(dataload(self.infos[idx]['lr'])).reshape(self.H,self.W)
        data["hr"] = (hr_raw.astype(np.float32) - self.bl) / (self.wp - self.bl)
        data["lr"] = (lr_raw.astype(np.float32) - self.bl) * self.infos[idx]['ratio'] / (self.wp - self.bl)

        data['name'] = self.infos[idx]['name']
        data['wb'], data['ccm'] = read_wb_ccm(rawpy.imread(self.infos[idx]['hr']))
        data['ratio'] = self.infos[idx]['ratio']
        data['ISO'] = self.infos[idx]['iso']

        if self.args['clip']:
            data["hr"] = data["hr"].clip(0,1)
            data["lr"] = data["lr"].clip(0,1)

        return data

'''
ds = np.load('E:/datasets/LRID/resources/darkshading-iso-6400.npy')
    lr_paths = glob.glob()
    raw_hr = rawpy.imread('F:/datasets/SELD/indoor_x5/100/004/000071_exp-512000000_iso-100_2022_08_17_06_21_43_034_12562490542059_orientation_0_camera-0.dng')
    raw_lr = rawpy.imread('E:/datasets/LRID/indoor_x5/6400/1/004/000115_exp-8000000_iso-6400_2022_08_17_06_21_58_288_12578251173465_orientation_0_camera-0.dng')#.raw_image_visible
    p = get_ISO_ExposureTime('F:/datasets/SELD/indoor_x5/100/004/000071_exp-512000000_iso-100_2022_08_17_06_21_43_034_12562490542059_orientation_0_camera-0.dng')
    p['name'] = 'IMX686'
    bl = raw_lr.black_level_per_channel[0]
    print(raw_lr.black_level_per_channel)
    p['bl'] = raw_lr.black_level_per_channel[0]
    p['wp'] = 1023
    p['ratio'] = 1
    p['scale'] = (p['wp']-p['bl']) / p['ratio']
    print(p)
    lr_raw = (raw_lr.raw_image_visible.astype(np.float32) - p['bl']) / (p['wp'] - p['bl'])
    hr_raw = (raw_hr.raw_image_visible.astype(np.float32) - p['bl']) / (p['wp'] - p['bl'])
    hr_raw = np.load('E:/datasets/LRID/indoor_x5/npy/GT_align_ours/004.npy')
    hr_raw = (hr_raw.astype(np.float32) - p['bl']) / (p['wp'] - p['bl'])
    print(hr_raw.shape, hr_raw.min(), hr_raw.max())
    print(lr_raw.shape, lr_raw.min(), lr_raw.max())

    raw_hr.raw_image_visible[:] = hr_raw * (p['wp'] - p['bl']) + p['bl']
    img_hr = raw_hr.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=8)
    plt.imsave(f"{p['name']}.png", img_hr)

    data = {'lr': lr_raw, 'hr':hr_raw.clip(0,1), 'name':p['name']}
    '''