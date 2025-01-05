import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import time
from torch.optim import Adam, lr_scheduler
from data_process import *
from utils import *
from archs import *
from losses import *
from trainer_base import *
from bm3d import bm3d
import scipy.signal as sg

def get_threshold(data, step=1, mode='score2', print_log=False, scale=1023-64):
    if mode == 'score2':
        quants = np.linspace(step, 100, 100//step)
        ths = np.percentile(data.reshape(-1), quants, method='linear')
        score = ths/quants
        start_pos = np.where(score>0)[0][0] + 5
        i = np.argmin(score[start_pos:]) + start_pos
        th = ths[i]
        if print_log: log(f'Adaptive percent: {quants[i]}% - th: {ths[i]*scale:.2f}')
    elif mode == 'score3':
        data, mean = data
        nbins = 1000
        quants = np.linspace(step, 100, 100//step, endpoint=True)
        ths = np.percentile(data.reshape(-1), quants, method='linear')
        npeaks = np.ones_like(ths)
        for i in range(len(ths)):
            # 思路1：统计直方图的峰值数量（思路错误）
            # hist, bin_edges = np.histogram(mean[img_lap<=ths[i]]*1, bins=nbins, range=(0, 1))
            # hist_diff = np.array([0,0,] + list(hist[1:] - hist[:-1]))
            # cross = (hist_diff[1:] * hist_diff[:-1]) <= 0
            # sum(cross)//2 + 1
            # 思路2：统计直方图的有效“宽度”
            # 将数据映射到桶中
            # 将数据归一化到[0, nbins-1]的范围内，然后取整数索引
            bucket_indices = (mean[data <= ths[i]].clip(0,1) * nbins).astype(int)
            # 使用 np.bincount 来统计每个桶的元素数量
            # np.bincount 返回的是长度为 nbins 的数组，表示每个桶的计数
            bucket_counts = np.bincount(bucket_indices, minlength=nbins+1)
            # 计算未被映射到的桶的数量，即计数为0的桶
            # 由于 bucket_counts 是从索引0开始的，我们需要加上1来得到实际的桶索引
            npeaks[i] = np.sum(bucket_counts > 0)
            # npeaks[i] = len(np.unique(np.floor(mean[data<=ths[i]]*nbins)))
        score = ths/(quants*npeaks)
        start_pos = 1
        i = np.argmin(score[start_pos:]) + start_pos
        th = ths[i]
        if print_log: log(f'Adaptive percent: {quants[i]}% - th: {ths[i]*scale:.2f}')
    else:
        raise NotImplementedError
    return th, quants[i]

'''
plt.scatter(quants, score/score.max())
plt.plot(quants, score/score.max())
plt.plot(quants, ths[:-1]/ths.max())
plt.grid('on')
plt.show()
'''

def SelfNLF(lr_rggb, k=29, kwargs={}):
    # SIDD特有的小图拼接
    if 'SIDD_256' in kwargs and kwargs['SIDD_256']:
        lr_rggb = np.concatenate(np.split(lr_rggb, 32, axis=-2), axis=-1)
    # lr_rggb = np.clip(lr_rggb * gain, 0, 1)
    lr_rggb_k = stdfilt(lr_rggb, k)
    # 噪图的均值要靠模糊获得，边缘一定不准，但没啥好办法，只能寄希望于分割得好
    mean = cv2.blur(lr_rggb, (k, k))
    img_lap = stdfilt(cv2.blur(lr_rggb, (k//3*2+1, k//3*2+1)), k)
    var = lr_rggb_k**2
    # 基于分布分点位的自适应阈值
    print_log = kwargs['print_log'] if 'print_log' in kwargs else False
    # th, percent = get_threshold(img_lap, step=1, mode='score2', print_log=print_log)
    th, percent = get_threshold((img_lap, mean), step=5, mode='score3', print_log=print_log)
    # 按阈值分割图像平滑区域与非平滑区域
    if len(var[img_lap<th].reshape(-1)) > 0:
        var, mean = var[img_lap<th], mean[img_lap<th]
    else:
        log('Warning: no flat area, default 25\% area')
        th_backup = np.percentile(img_lap.reshape(-1), 25, method='linear')
        if th != th_backup:
            th = th_backup
            var, mean = var[img_lap<th], mean[img_lap<th]
    # 可以用RANSAC拟合，但巨慢且收益不大
    reg = polyfit(mean.reshape(-1), var.reshape(-1), ransac=False, clip=False)
    return reg

def CollabNLF(lr_rggb, hr_rggb, k=29, kwargs={}):
    # SIDD特有的小图拼接
    if 'SIDD_256' in kwargs and kwargs['SIDD_256']:
        lr_rggb = np.concatenate(np.split(lr_rggb, 32, axis=-2), axis=-1)
        hr_rggb = np.concatenate(np.split(hr_rggb, 32, axis=-2), axis=-1)
    lr_rggb_k = stdfilt(lr_rggb, k)
    hr_rggb_k = stdfilt(hr_rggb, k)
    var = lr_rggb_k**2 - hr_rggb_k**2
    mean = cv2.blur(hr_rggb, (k, k))
    img_lap = hr_rggb_k
    
    # 基于分布分点位的自适应阈值
    print_log = kwargs['print_log'] if 'print_log' in kwargs else False
    # th, percent = get_threshold(img_lap, step=1, mode='score2', print_log=print_log)
    th, percent = get_threshold((img_lap, mean), step=5, mode='score3', print_log=print_log)
    # 按阈值分割图像平滑区域与非平滑区域
    if len(var[img_lap<th].reshape(-1)) > 0:
        var, mean = var[img_lap<th], mean[img_lap<th]
    else:
        log('Warning: no flat area, default 25% area')
        th_backup = np.percentile(img_lap.reshape(-1), 25, method='linear')
        if th != th_backup:
            th = th_backup
            var, mean = var[img_lap<th], mean[img_lap<th]
    # 可以用RANSAC拟合，但巨慢且收益不大
    reg = polyfit(mean.reshape(-1), var.reshape(-1), ransac=False, clip=False)
    return reg

def SimpleNLF(lr_raw, hr_raw=None, k=29, setting={'mode':'self'}):
    lr_rggb = bayer2rggb(lr_raw)
    if setting['mode']=='self':
        reg = SelfNLF(lr_rggb, k, setting)
    elif setting['mode']=='collab':
        hr_rggb = bayer2rggb(hr_raw)
        reg = CollabNLF(lr_rggb, hr_rggb, k, setting)
    return reg

# 早期实验，有效但会拉低泛化能力，不再使用
def NeuralNLF(lr_raw, hr_raw=None, k=19, setting={'mode':'self'}):
    lr_rggb = bayer2rggb(lr_raw)
    if setting['mode']=='self':
        reg = SelfNLF(lr_rggb, k, setting)
    elif setting['mode']=='collab':
        hr_rggb = bayer2rggb(hr_raw)
        reg = CollabNLF(lr_rggb, hr_rggb, k, setting)
    return reg

class YOND_SIDD():
    def __init__(self):
        # 初始化
        parser = YONDParser()
        self.parser = parser.parse()
        self.initialization()
    
    def initialization(self):
        os.environ["CUDA_VISIBLE_DEVICES"] = self.parser.gpu
        with open(self.parser.runfile, 'r', encoding="utf-8") as f:
            self.args = yaml.load(f.read(), Loader=yaml.FullLoader)
        self.mode = self.args['mode'] if self.parser.mode is None else self.parser.mode
        if self.parser.debug is True:
            self.args['num_workers'] = 0
            warnings.warn('You are using debug mode, only main worker(cpu) is used!!!')
        if 'clip' not in self.args['dst']: 
            self.args['dst']['clip'] = False
        self.save_plot = False if self.parser.nofig else True
        self.args['dst']['mode'] = self.mode
        self.hostname, self.hostpath, self.multi_gpu = get_host_with_dir()
        self.model_dir = self.args['checkpoint']
        if not self.parser.nohost:
            for key in self.args:
                if 'dst' in key:
                    self.args[key]['root_dir'] = f"{self.hostpath}/{self.args[key]['root_dir']}"
        self.dst = self.args['dst']
        self.arch = self.args['arch']
        self.pipe = self.args['pipeline']
        if self.pipe['bias_corr'] == 'none':
            self.pipe['bias_corr'] = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_name = self.args['model_name']
        self.method_name = self.args['method_name']
        self.fast_ckpt = self.args['fast_ckpt']
        self.sample_dir = os.path.join(self.args['result_dir'] ,f"{self.method_name}")
        self.biaslut = BiasLUT() if os.path.exists('checkpoints/bias_lut_2d.npy') else None
        os.makedirs(self.sample_dir, exist_ok=True)
        os.makedirs('./logs', exist_ok=True)
        os.makedirs('./metrics', exist_ok=True)
        
        # 模型加载
        self.net = globals()[self.arch['name']](self.arch)
        model_path = os.path.join(f'{self.fast_ckpt}/{self.model_name}_best_model.pth')
        if not os.path.exists(model_path):
            model_path = os.path.join(f'{self.fast_ckpt}/{self.model_name}_last_model.pth')
            if not os.path.exists(model_path):
                model_path = os.path.join(f'{self.fast_ckpt}/{self.model_name}.pth')
        model = torch.load(model_path, map_location=self.device)
        self.net = load_weights(self.net, model, by_name=False)
        self.net = self.net.to(self.device)

        # 加载噪声参数估计模块
        self.est_args = {key: self.args[key] for key in self.args if 'est_' in key}
        self.est_net = {key: None for key in self.args if 'est_' in key}
        for est_name in self.est_args:
            est = self.est_args[est_name]
            self.est_net[est_name] = globals()[est['name']](est)
            model_path = os.path.join(est["weights"])
            model = torch.load(model_path, map_location=self.device)
            self.est_net[est_name] = load_weights(self.est_net[est_name], model, by_name=False)
            self.est_net[est_name] = self.est_net[est_name].to(self.device)
        
        self.change_eval_dst(mode='eval')
        self.print_model_log()
    
    def print_model_log(self):
        pytorch_total_params = sum([p.numel() for p in self.net.parameters()])
        self.eval_psnrs = [AverageMeter('PSNR', ':2f') for _ in range(5)]
        self.eval_ssims = [AverageMeter('SSIM', ':4f') for _ in range(5)]
        self.eval_psnrs_rgb = [AverageMeter('PSNR', ':2f') for _ in range(5)]
        self.eval_ssims_rgb = [AverageMeter('SSIM', ':4f') for _ in range(5)]
        self.logfile = f'./logs/log_{self.method_name}.log'
        log(f'Method Name:\t{self.method_name}', log=self.logfile, notime=True)
        log(f'Model Name:\t{self.model_name}', log=self.logfile, notime=True)
        log(f'Architecture:\t{self.arch["name"]}', log=self.logfile, notime=True)
        log(f'Parameters:\t{pytorch_total_params/1e6:.2f}M', log=self.logfile, notime=True)
        log(f'EvalDataset:\t{self.args["dst_eval"]["dataset"]}', log=self.logfile, notime=True)
        log(f'num_channels:\t{self.arch["nf"]}', log=self.logfile, notime=True)
        # log(f'PatchSize:\t{self.dst["patch_size"]}', log=self.logfile, notime=True)
        log(f'num_workers:\t{self.args["num_workers"]}', log=self.logfile, notime=True)
        log(f'Command:\t{self.dst["command"]}', log=self.logfile, notime=True)
        log(f"Let's use {torch.cuda.device_count()} GPUs!", log=self.logfile, notime=True)
        # self.device != torch.device(type='cpu') 
        if 'gpu_preprocess' in self.dst and self.dst['gpu_preprocess']:
            log("Using PyTorch's GPU Preprocess...")
            self.use_gpu = True
        else:
            log(f"Using Numpy's CPU Preprocess")
            self.use_gpu = False 

        if torch.cuda.device_count() > 1:
            log("Using PyTorch's nn.DataParallel for multi-gpu...")
            self.multi_gpu = True
            self.net = nn.DataParallel(self.net)
        else:
            self.multi_gpu = False
    
    def change_eval_dst(self, mode='eval'):
        self.dst = self.args[f'dst_{mode}']
        self.dstname = self.dst['dstname']
        self.dst_eval = globals()[self.dst['dataset']](self.dst)
    
    def Simple_Denoiser(self, lr_raw, denoiser='unet', p=None, show=False):
        lr_rggb = bayer2rggb(lr_raw)#pack_raw(lr_raw)
        with torch.no_grad():
            lr_rggb = torch.from_numpy(lr_rggb).float().to(self.device).permute(2,0,1)[None,]
            p2d = get_p2d(lr_rggb.shape, base=32)
            lr_rggb = F.pad(lr_rggb, p2d, mode='reflect')
            dn_rggb = self.net(lr_rggb.clamp(0,1)).clamp(0,1)
            _, _, H, W = dn_rggb.shape
            dn_rggb = dn_rggb[..., p2d[-2]:H-p2d[-1], p2d[0]:W-p2d[1]]
            dn_rggb = dn_rggb[0].permute(1,2,0).detach().cpu().numpy()
        return rggb2bayer(dn_rggb)#unpack_raw(dn_rggb)

    def VST_Denoiser(self, lr_raw, hr_raw=None, bias_corr='pre', bias_func=None, denoiser='bm3d', p=None, show=False):   
        lr_rggb = bayer2rggb(lr_raw) * p['scale']
        bias_base = np.maximum(lr_rggb, 0)
        if bias_corr is not None:
            if self.biaslut is None:
                if bias_func is None:
                    bias_func = get_bias(lr_rggb.max(), p['sigma'], p['gain'], post=(bias_corr=='post'), show=show)
                bias = bias_func(bias_base)
            else:
                bias = self.biaslut.get_lut(bias_base, K=p['gain'], sigGs=p['sigma'])
        raw_vst = VST(lr_rggb, p['sigma'], gain=p['gain'])
        if bias_corr == 'pre':
            raw_vst = raw_vst - bias
        # Denoise
        lower = VST(0, p['sigma'], gain=p['gain'])
        upper = VST(p['scale'], p['sigma'], gain=p['gain'])
        if denoiser == 'fbi' or denoiser == 'bm3d':
            lower, upper = raw_vst.min(), raw_vst.max()
        nsr = 1 / (upper - lower)
        raw_vst = (raw_vst - lower) / (upper - lower)
        if denoiser == 'bm3d':
            # raw_vst = bm3d(raw_vst, 1.0)
            raw_vst = bm3d(raw_vst, 1.0 * nsr)
        else:
            with torch.no_grad():
                raw_vst = torch.from_numpy(raw_vst).float().to(self.device).permute(2,0,1)[None,]
                if denoiser == 'fbi':
                    raw_vst = rggb2bayers(raw_vst.permute(0,2,3,1))[:,None]
                    raw_vst = self.net(raw_vst.clamp(0,1)).clamp(0,1)
                    raw_vst = bayer2rggbs(raw_vst[:,0]).permute(0,3,1,2)
                else:
                    p2d = get_p2d(raw_vst.shape, base=32)
                    raw_vst = F.pad(raw_vst, p2d, mode='reflect')
                    if 'guided' in self.arch:
                        sigma_corr = 1.03 if bias_corr == 'pre' else 1.00
                        t = torch.tensor(nsr*sigma_corr, dtype=raw_vst.dtype, device=raw_vst.device)
                        raw_vst = self.net(raw_vst.clamp(0,1), t).clamp(0,1)
                    else:
                        raw_vst = self.net(raw_vst.clamp(0,1)).clamp(0,1)
                    _, _, H, W = raw_vst.shape
                    raw_vst = raw_vst[..., p2d[-2]:H-p2d[-1], p2d[0]:W-p2d[1]]
                raw_vst = raw_vst[0].permute(1,2,0).detach().cpu().numpy()
        raw_vst = raw_vst * (upper - lower) + lower
        
        # if bias_corr == 'post':
        #     raw_vst = raw_vst - bias
        exact_inverse = True if bias_corr is None and self.pipe['vst_type'] == 'exact' else False
        raw_vst = inverse_VST(raw_vst, p['sigma'], gain=p['gain'], exact=exact_inverse)
        raw_dn = rggb2bayer(raw_vst) / p['scale']
        return raw_dn

    def IterDenoise(self, data, params):
        results = {}
        lr_path_full = data['lr_path_full']
        lr_raw = data['lr']
        hr_raw = data['hr'] if 'hr' in data else None
        meta = data['meta']
        name = data['name']
        img_id = params['img_id']
        p = params['p']
        regs = []

        ###### Round 1：Self-Calibration + VST_Denoising
        # 噪声参数估计
        if self.pipe['full_est']:
            lr_raw = np.concatenate(lr_raw, axis=-1)
            if 'cal_est' in self.pipe:
                with open(self.pipe['cal_est'], 'rb') as f:
                    record = pkl.load(f)
                ct, iso = name.split('_')[2], int(name.split('_')[3])
                if f'{ct}_{iso:05d}' not in record['sfrn']:
                    reg = np.poly1d(record['beta1'][ct])(iso), np.poly1d(record['beta2'][ct])(iso)
                else:
                    reg = record['sfrn'][f'{ct}_{iso:05d}']
            elif 'foi' in self.pipe['est_type']:
                reg = sio.loadmat(f'{self.dst["root_dir"]}/SIDD_Validation_Raw/FoiEst_fullPict.mat')['return_params'][img_id]
            elif 'liu' in self.pipe['est_type']:
                reg = sio.loadmat(f'{self.dst["root_dir"]}/SIDD_Validation_Raw/LiuEst_fullPict.mat')['return_params'][img_id]
            elif 'zou' in self.pipe['est_type']:
                reg = np.load(f'{self.dst["root_dir"]}/SIDD_Validation_Raw/Zou_fullPict.npy')[img_id]
            elif 'pge' in self.pipe['est_type']:
                if 'est_net' not in self.args:
                    reg = np.load(f'{self.dst["root_dir"]}/SIDD_Validation_Raw/PGE_fullPict.npy')[img_id]
                else:
                    raw_inp = torch.from_numpy(lr_raw.copy()).float().to(self.device)[None,None,:]
                    reg = self.est_net['est_net'](raw_inp)
                    reg = reg.detach().cpu().numpy()
                reg[1] = reg[1] ** 2
            elif 'simple' in self.pipe['est_type']:
                lr_full = dataload(lr_path_full)
                raw4est = lr_raw if lr_path_full is None else lr_full
                reg = SimpleNLF(raw4est, k=self.pipe['k'], setting={'mode':'self', 'print_log':True})
            elif 'ours' in self.pipe['est_type']:
                k = self.est_args['est_self']['k']
                lr_full = dataload(lr_path_full)
                raw4est = lr_raw if lr_path_full is None else lr_full
                # reg = SimpleNLF(raw4est, k=self.pipe['k'], setting={'mode':'self', 'print_log':True})
                setting = {'mode':'self', 'print_log':True, 'net':self.est_net['est_self'], 'SIDD_256':False}
                reg = NeuralNLF(raw4est, None, k=k, setting=setting)
            elif 'manual' in self.pipe['est_type']:
                p['gain'], p['sigma'] = 14, 20
                reg = p['gain']/(p['wp']-p['bl']), (p['sigma']/(p['wp']-p['bl']))**2
            else:
                raise NotImplementedError
            lr_raw = np.array(np.split(lr_raw, 32, axis=-1))
            # show Est results
            p['gain'], p['sigma'] = reg[0]*(p['wp']-p['bl']), np.sqrt(max(reg[1], 0))*(p['wp']-p['bl'])
            log(f"Self Est: K={p['gain']:.4f}, b={p['sigma']:.4f} (beta1={reg[0]:.3e}, beta2={reg[1]:.3e})", log=self.logfile)
        else:
            if 'pge' in self.pipe['est_type']:
                if 'est_net' not in self.args:
                    reg = np.load(f'{self.dst["root_dir"]}/SIDD_Validation_Raw/PGE.npy')[img_id]
                else:
                    raw_inp = torch.from_numpy(lr_raw.copy()).float().to(self.device)[:,None,:,:]
                    reg = self.est_net['est_net'](raw_inp)
                    reg = reg.detach().cpu().numpy()
                reg[:,1] = reg[:,1] ** 2
            else:
                raw_dn = np.empty((32, 256, 256), np.float32)
                for num in range(32):
                    raw_dn[num] = self.Simple_Denoiser(lr_raw[num])
                    # raw_dn[num] = rot_bayer(self.Simple_Denoiser(rot_bayer(lr_raw[num], p['cfa'])).clip(0,1), p['cfa'], rev=True)
                raw_dn = np.concatenate(raw_dn, axis=-1)
                raw_dns = [raw_dn.copy()]
                results['raw_dns'] = raw_dns
                results['lr_raw'] = np.concatenate(data['lr'], axis=-1)
                results['hr_raw'] = np.concatenate(data['hr'], axis=-1) if 'hr' in data else None
                results['regs'] = (0,0)
                return results
                # raise NotImplementedError
            p['gain'], p['sigma'] = reg[:,0].mean()*(p['wp']-p['bl']), np.sqrt(max(reg[:,1].mean(), 0))*(p['wp']-p['bl'])
            log(f"Self Est (mean): K={p['gain']:.4f}, b={p['sigma']:.4f} (beta1={reg[:,0].mean():.3e}, beta2={reg[:,1].mean():.3e})", log=self.logfile)
        regs.append(reg)
        # Denoise
        bias_corr = self.pipe['bias_corr']#None if self.pipe['iter']!='once' else self.pipe['bias_corr']
        bias_func = None
        p['stage'] = 'self'
        if self.pipe['full_dn']:
            lr_raw = np.concatenate(lr_raw, axis=-1)
            raw_dn = self.VST_Denoiser(lr_raw, None, bias_corr, denoiser=self.pipe['denoiser_type'], p=p).clip(0,1)
        else:
            raw_dn = np.empty((32, 256, 256), np.float32)
            if bias_corr is not None:
                upper_bound = lr_raw.max()*(p['wp']-p['bl'])
                if self.biaslut is None:
                    bias_func = get_bias(upper_bound, p['sigma'], p['gain'])
                else:
                    bias_func = None
            for num in range(32):
                if self.pipe['est_type'] == 'pge':
                    p['gain'], p['sigma'] = reg[num, 0]*(p['wp']-p['bl']), np.sqrt(max(reg[num, 1], 0))*(p['wp']-p['bl'])
                # raw_dn[num] = self.Simple_Denoiser(lr_raw[num])
                if 'rot_cfa' in p:
                    raw_dn[num] = rot_bayer(self.VST_Denoiser(rot_bayer(lr_raw[num], p['cfa']), None, bias_corr, 
                        bias_func=bias_func, denoiser=self.pipe['denoiser_type'], p=p).clip(0,1), p['cfa'], rev=True)
                else:
                    raw_dn[num] = self.VST_Denoiser(lr_raw[num], None, bias_corr, 
                        bias_func=bias_func, denoiser=self.pipe['denoiser_type'], p=p).clip(0,1)
            raw_dn = np.concatenate(raw_dn, axis=-1)

        raw_dns = [raw_dn.copy()]

        ###### Full Image Pre-Denoising Trick on SIDD
        if lr_path_full is not None:
            # raw4est_dn = self.VST_Denoiser(raw4est, None, bias_corr, denoiser=self.pipe['denoiser_type'], p=p).clip(0,1)
            raw4est_dn = raw_dn
        else:
            raw4est_dn = raw_dn
        ###### Round 2：Iterative-Calibration + VST_Denoising
        if self.pipe['iter'] == 'iter':
            p['stage'] = 'collab'
            for epoch in range(1, self.pipe['max_iter']+1):
                # try:
                    if self.pipe['full_est']:
                        if not self.pipe['full_dn']: lr_raw = np.concatenate(lr_raw, axis=-1)
                        if 'ours' in self.pipe['est_type']:
                            k = self.est_args['est_collab']['k']
                            # reg = SimpleNLF(lr_raw, raw_dn, k=self.pipe['k'], setting={'mode':'collab', 'print_log':True, 'SIDD_256':True})
                            setting={'mode':'collab', 'print_log':True, 'net':self.est_net['est_collab'], 'SIDD_256':True}
                            reg = NeuralNLF(lr_raw, raw_dn, k=k, setting=setting)
                        else:
                            reg = SimpleNLF(lr_raw, raw_dn, k=self.pipe['k'], setting={'mode':'collab', 'print_log':True, 'SIDD_256':True})
                            # reg = SimpleNLF(raw4est, raw4est_dn, k=self.pipe['k'], setting={'mode':'collab', 'print_log':True, 'SIDD_256':False})
                        if not self.pipe['full_dn']: lr_raw = np.array(np.split(lr_raw, 32, axis=-1))
                    else:
                        pass#raise NotImplementedError

                    bias_corr = self.pipe['bias_corr']
                    if reg[1] < 0:
                        log(f'Warning!!! b={reg[1]:.4f} is backup to {reg[0]**2:.4f}', log=self.logfile)
                        reg = (reg[0], reg[0]**2)

                    p['gain'], p['sigma'] = reg[0]*(p['wp']-p['bl']), np.sqrt(reg[1])*(p['wp']-p['bl'])
                    log(f"Iter {epoch} Est: K={p['gain']:.4f}, sigma={p['sigma']:.4f} (beta1={reg[0]:.3e}, beta2={reg[1]:.3e})", log=self.logfile)

                    if reg[0] < 0:
                        log(f'Warning!!! Wrong noise level! Backup to iter_0 result.', log=self.logfile)
                        break
                    
                    # VST + Denoise
                    upper_bound = lr_raw.max()*(p['wp']-p['bl'])
                    if self.biaslut is None:
                        bias_func = get_bias(upper_bound, p['sigma'], p['gain'], post=(bias_corr=='post'))
                    else:
                        bias_func = None
                    
                    if self.pipe['full_dn']:
                        raw_dn = self.VST_Denoiser(lr_raw, raw_dn, bias_corr=bias_corr, bias_func=bias_func, 
                                               denoiser=self.pipe['denoiser_type'], p=p).clip(0,1)
                    else:
                        raw_dn = np.array(np.split(raw_dn, 32, axis=-1))
                        for num in range(32):
                            if 'rot_cfa' in p:
                                raw_dn[num] = rot_bayer(self.VST_Denoiser(rot_bayer(lr_raw[num], p['cfa']), rot_bayer(raw_dn[num], p['cfa']), bias_corr, 
                                    bias_func=bias_func, denoiser=self.pipe['denoiser_type'], p=p).clip(0,1), p['cfa'], rev=True)
                            else:
                                raw_dn[num] = self.VST_Denoiser(lr_raw[num], raw_dn[num], bias_corr=bias_corr, bias_func=bias_func, denoiser=self.pipe['denoiser_type'], p=p).clip(0,1)
                        raw_dn = np.concatenate(raw_dn, axis=-1)

                    if 'ddim' in self.pipe:
                        p['ddim'] = {'eta':0.85, 'at':1/2**epoch}
                    raw_dns.append(raw_dn.copy())
                    regs.append(reg)
                # except Exception as e:
                #     print(e)
                #     log(f'Warning!!! Pleach check log at "{name[:4]}_{num}_{epoch}". Backup to iter_0 result.', log=self.logfile)
                #     break

        if not self.pipe['full_dn']: lr_raw = np.concatenate(lr_raw, axis=-1)
        results['raw_dns'] = raw_dns
        results['lr_raw'] = np.concatenate(data['lr'], axis=-1)
        results['hr_raw'] = np.concatenate(data['hr'], axis=-1) if 'hr' in data else None
        results['regs'] = regs
        return results

    def eval(self, epoch=-1):
        self.net.eval()
        for i in range(5):
            self.eval_psnrs[i].reset()
            self.eval_ssims[i].reset()
            self.eval_psnrs_rgb[i].reset()
            self.eval_ssims_rgb[i].reset()
        pool = []
        # record every metric
        self.metrics = {}
        metrics_path = f'./metrics/{self.method_name}_metrics.pkl'
        if os.path.exists(metrics_path):
            with open(metrics_path, 'rb') as f:
                self.metrics = pkl.load(f)
                
        t = tqdm(total=len(self.dst_eval))
        bench_init = np.empty((40, 32, 256, 256), np.float32)
        bench_results = np.empty((40, 32, 256, 256), np.float32)
        p = self.pipe
        p.update({'K':8.74253, 'sigGs':12.81, 'wp': 1023, 'bl': 64, 'ratio': 1, 'gain': 1, 'sigma': 0})
        p['scale'] = (p['wp']-p['bl']) / p['ratio']

        for k, data in enumerate(self.dst_eval):# img.shape=[32,H,W]
            meta = data['meta']
            name = data['name']
            p['cfa'] = data['cfa']
            self.metrics[name] = {'psnr':[], 'ssim':[], 'psnr_rgb':[], 'ssim_rgb':[], 'reg':[]}
            outputs = np.zeros((self.pipe['max_iter']+1, 256, 256*32), np.float32)
            # 迭代去噪
            results = self.IterDenoise(data, {'p':p, 'img_id':k})
            raw_dns = results['raw_dns']
            lr_raw, hr_raw = results['lr_raw'], results['hr_raw']
            self.metrics[name]['reg'] = results['regs']
            for it in range(len(raw_dns)):
                outputs[it] = raw_dns[it]

            pool.append(threading.Thread(target=self.multiprocess_plot, args=(lr_raw, outputs, hr_raw, meta, name)))
            # self.multiprocess_plot(lr_raw, outputs, hr_raw, meta, name)
            # pool.append(threading.Thread(target=self.nothing))
            pool[-1].start()

            t.set_description(f'{name}')
            t.set_postfix({'PSNR(Raw)':f"{self.eval_psnrs[-1].avg:.2f}", 'SSIM(Raw)':f"{self.eval_ssims[-1].avg:.4f}",
                           'PSNR(sRGB)':f"{self.eval_psnrs_rgb[-1].avg:.2f}", 'SSIM(sRGB)':f"{self.eval_ssims_rgb[-1].avg:.4f}"})
            t.update(1)
                
            bench_init[k] = np.array(np.split(raw_dns[0], 32, axis=-1))
            bench_results[k] = np.array(np.split(raw_dns[-1], 32, axis=-1))

            # 缓存
            os.makedirs(f'npy/{self.method_name}', exist_ok=True)
            np.save(f'npy/{self.method_name}/{k:03d}.npy', outputs)
        
        for i in range(len(pool)):
            pool[i].join()
        del pool

        # # save denoised images in a .mat file with dictionary key "results"
        # res_fn = f"submits/{self.mode}/{self.method_name}"
        # res_key = 'results'  # Note: do not change this key, the evaluation code will look for this key
        # os.makedirs(res_fn, exist_ok=True)
        # sio.savemat(res_fn+'/SubmitRaw.mat', {res_key: bench_results})

        # os.makedirs(res_fn.replace('iter', 'once'), exist_ok=True)
        # sio.savemat(res_fn.replace('iter', 'once')+'/SubmitRaw.mat', {res_key: bench_init})

        log(f'{self.method_name}:', log=self.logfile)
        for it in range(self.pipe['max_iter']+1):
            psnr, ssim = self.eval_psnrs[it].avg, self.eval_ssims[it].avg
            psnr_rgb, ssim_rgb = self.eval_psnrs_rgb[it].avg, self.eval_ssims_rgb[it].avg
            log(f"Iter{it}: PSNR={psnr:.2f}, SSIM={ssim:.4f}\n" +
                f"PSNR(sRGB)={psnr_rgb:.2f}, SSIM(sRGB)={ssim_rgb:.4f}",
                log=self.logfile)
        
        psnr, ssim = self.eval_psnrs[-1].avg, self.eval_ssims[-1].avg
        psnr_rgb, ssim_rgb = self.eval_psnrs_rgb[-1].avg, self.eval_ssims_rgb[-1].avg
        log(f"Iter_last: PSNR={psnr:.2f}, SSIM={ssim:.4f}\n" +
            f"PSNR(sRGB)={psnr_rgb:.2f}, SSIM(sRGB)={ssim_rgb:.4f}",
            log=self.logfile)
        
        with open(metrics_path, 'wb') as f:
            pkl.dump(self.metrics, f)

        plt.close('all')
        gc.collect()
        return self.metrics
    
    def benchmark(self):
        self.net.eval()
        self.metrics = {}
        metrics_path = f'./metrics/{self.method_name}_metrics.pkl'
        if os.path.exists(metrics_path):
            with open(metrics_path, 'rb') as f:
                self.metrics = pkl.load(f)
        os.makedirs(f'{self.sample_dir}/benchmark', exist_ok=True)
                
        t = tqdm(total=len(self.dst_eval))
        bench_init = np.empty((40, 32, 256, 256), np.float32)
        bench_results = np.empty((40, 32, 256, 256), np.float32)
        p = self.pipe
        p.update({'K':8.74253, 'sigGs':12.81, 'wp': 1023, 'bl': 64, 'ratio': 1, 'gain': 1, 'sigma': 0})
        p['scale'] = (p['wp']-p['bl']) / p['ratio']

        for k, data in enumerate(self.dst_eval):# img.shape=[32,H,W]
            meta = data['meta']
            name = data['name']
            outputs = np.zeros((self.pipe['max_iter']+1, 256, 256*32), np.float32)
            # 迭代去噪
            results = self.IterDenoise(data, {'p':p, 'img_id':k})
            raw_dns = results['raw_dns']
            lr_raw = results['lr_raw']
            self.metrics[name]['reg_test'] = results['regs']
            for it in range(len(raw_dns)):
                outputs[it] = raw_dns[it]

            if self.save_plot:
                img_lr = process_sidd_image(lr_raw, meta['bayer_2by2'], meta['wb'], meta['cst2'])
                cv2.imwrite(f'{self.sample_dir}/benchmark/{name[:4]}_noisy.png', img_lr)

            for it in range(self.pipe['max_iter']+1):
                if outputs[it].max() <= 0:   # 有的场景迭代挂掉了，没数据
                    continue
                img_dn = process_sidd_image(outputs[it], meta['bayer_2by2'], meta['wb'], meta['cst2'])
                cv2.imwrite(f'{self.sample_dir}/benchmark/{name[:4]}_{it}.png', img_dn)

            t.set_description(f'{name}')
            t.update(1)
                
            bench_init[k] = np.array(np.split(raw_dns[0], 32, axis=-1))
            bench_results[k] = np.array(np.split(raw_dns[-1], 32, axis=-1))

            # 缓存
            # os.makedirs(f'npy/{self.method_name}', exist_ok=True)
            # np.save(f'npy/{self.method_name}/{k:03d}.npy', outputs)

        # save denoised images in a .mat file with dictionary key "results"
        # res_fn = f"submits/{self.mode}/{self.method_name}"
        # res_key = 'results'  # Note: do not change this key, the evaluation code will look for this key
        # os.makedirs(res_fn, exist_ok=True)
        # sio.savemat(res_fn+'/SubmitRaw.mat', {res_key: bench_results})

        # os.makedirs(res_fn.replace('iter', 'once'), exist_ok=True)
        # sio.savemat(res_fn.replace('iter', 'once')+'/SubmitRaw.mat', {res_key: bench_init})
            
        plt.close('all')
        gc.collect()

    def nothing(self):
        return

    def multiprocess_plot(self, lr_raw, outputs, hr_raw, meta, name):
        if self.save_plot:
            img_lr = process_sidd_image(lr_raw, meta['bayer_2by2'], meta['wb'], meta['cst2'])
            cv2.imwrite(f'{self.sample_dir}/{name[:4]}_noisy.png', img_lr)
            if hr_raw is not None:
                img_hr = process_sidd_image(hr_raw, meta['bayer_2by2'], meta['wb'], meta['cst2'])
                cv2.imwrite(f'{self.sample_dir}/{name[:4]}_gt.png', img_hr)
        
        for it in range(self.pipe['max_iter']+1):
            if outputs[it].max() <= 0:
                self.eval_psnrs[it].update(-1)
                self.eval_ssims[it].update(-1)
                continue
            raw_dn_ = np.array(np.split(outputs[it], 32, axis=-1))
            raw_hr_ = np.array(np.split(hr_raw, 32, axis=-1))
            # Raw/Raw指标
            psnr = np.mean([compare_psnr(dn, hr, data_range=1) for dn, hr in zip(raw_dn_, raw_hr_)])
            ssim = np.mean([calculate_ssim(dn*255, hr*255) for dn, hr in zip(raw_dn_, raw_hr_)])
            self.eval_psnrs[it].update(psnr)
            self.eval_ssims[it].update(ssim)
            self.metrics[name]['psnr'].append(psnr)
            self.metrics[name]['ssim'].append(ssim)
            # Raw/RGB指标
            if self.save_plot:
                img_dn = process_sidd_image(outputs[it], meta['bayer_2by2'], meta['wb'], meta['cst2'])
                cv2.imwrite(f'{self.sample_dir}/{name[:4]}_{it}.png', img_dn)
                img_dn_ = np.array(np.split(img_dn, 32, axis=-2))
                img_hr_ = np.array(np.split(img_hr, 32, axis=-2))
                # Raw/Raw指标
                psnr_rgb = np.mean([compare_psnr(dn, hr, data_range=255) for dn, hr in zip(img_dn_, img_hr_)])
                ssim_rgb = np.mean([calculate_ssim(dn, hr) for dn, hr in zip(img_dn_, img_hr_)])
                self.eval_psnrs_rgb[it].update(psnr_rgb)
                self.eval_ssims_rgb[it].update(ssim_rgb)
                self.metrics[name]['psnr_rgb'].append(psnr_rgb)
                self.metrics[name]['ssim_rgb'].append(ssim_rgb)

        self.eval_psnrs[-1].update(psnr)
        self.eval_ssims[-1].update(ssim)
        log(f"{name}: PSNR={psnr:.2f}, SSIM={ssim:.4f}",log=self.logfile)
        if self.save_plot:
            self.eval_psnrs_rgb[-1].update(psnr_rgb)
            self.eval_ssims_rgb[-1].update(ssim_rgb)
            log(f"PSNR(sRGB)={psnr_rgb:.2f}, SSIM(sRGB)={ssim_rgb:.4f}",log=self.logfile)

def ssim(prediction, target):
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2
    img1 = prediction.astype(np.float64)
    img2 = target.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())
    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2
    ssim_map = ((2 * mu1_mu2 + C1) *
                (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                       (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def calculate_ssim(target, ref):
    '''
    calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    '''
    img1 = np.array(target, dtype=np.float64)
    img2 = np.array(ref, dtype=np.float64)
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1[:, :, i], img2[:, :, i]))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')

class YONDParser():
    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    def parse(self):
        self.parser.add_argument('--runfile', '-f', default="runfiles/YOND/SIDD_simple+full_pre_grumix.yml", type=Path, help="path to config")
        self.parser.add_argument('--mode', '-m', default='eval', type=str, help="train or test")
        self.parser.add_argument('--debug', action='store_true', default=False, help="debug or not")
        self.parser.add_argument('--nofig', action='store_true', default=True, help="don't save_plot")
        self.parser.add_argument('--nohost', action='store_true', default=False, help="don't save_plot")
        self.parser.add_argument('--gpu', default="0", help="os.environ['CUDA_VISIBLE_DEVICES']")
        return self.parser.parse_args()

if __name__ == '__main__':
    trainer = YOND_SIDD()
    if 'eval' in trainer.mode:
        trainer.change_eval_dst('eval')
        metrics = trainer.eval(-1)
        log(f'Metrics have been saved in ./metrics/{trainer.method_name}_metrics.pkl')
    if 'test' in trainer.mode:
        trainer.change_eval_dst('test')
        trainer.benchmark()