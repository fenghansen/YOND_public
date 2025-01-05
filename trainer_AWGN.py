import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import time
import copy
from torch.optim import Adam, AdamW, lr_scheduler
from data_process import *
from utils import *
from archs import *
from losses import *
from trainer_base import *

class AWGN_Trainer(Base_Trainer):
    def __init__(self):
        # 初始化
        parser = AWGN_Parser()
        self.parser = parser.parse()
        self.initialization()
        # model
        self.net = globals()[self.arch['name']](self.arch)
        # load weight
        if self.hyper['last_epoch']:
            # 不是初始化
            try:
                model_path = os.path.join(f'{self.fast_ckpt}/{self.model_name}_best_model.pth')
                if not os.path.exists(model_path):
                    model_path = os.path.join(f'{self.fast_ckpt}/{self.model_name}_last_model.pth')
                model = torch.load(model_path, map_location='cpu')
                self.net = load_weights(self.net, model, by_name=True)
            except:
                log('No checkpoint file!!!')
        else:
            log(f'Initializing {self.arch["name"]}...')
            initialize_weights(self.net)

        self.optimizer = Adam(self.net.parameters(), lr=self.hyper['learning_rate'])
        
        # Print Model Log
        self.print_model_log()

        self.infos = None
        if self.mode=='train':
            self.dst_train = globals()[self.args['dst_train']['dataset']](self.args['dst_train'])
            if self.multi_gpu:
                train_sampler = DistributedSampler(self.dst_train, shuffle=True)
                self.dataloader_train = DataLoader(self.dst_train, batch_size=self.hyper['batch_size'], worker_init_fn=worker_init_fn,
                                        shuffle=False, num_workers=self.args['num_workers'], pin_memory=False, drop_last=True,
                                        sampler=train_sampler)
            else:
                self.dataloader_train = DataLoader(self.dst_train, batch_size=self.hyper['batch_size'], worker_init_fn=worker_init_fn,
                                        shuffle=True, num_workers=self.args['num_workers'], pin_memory=False, drop_last=True)
            self.change_eval_dst('eval')
            self.dataloader_eval = DataLoader(self.dst_eval, batch_size=1, shuffle=False, 
                                    num_workers=self.args['num_workers'], pin_memory=False)

        # Choose Learning Rate
        self.lr_lambda = self.get_lr_lambda_func()
        self.scheduler = LambdaScheduler(self.optimizer, self.lr_lambda)

        if self.multi_gpu:
            self.net = DDP(self.net.cuda(self.local_rank), device_ids=[self.local_rank], output_device=self.local_rank)
            log('MultiGPU @DDP Mode...')
        else:
            self.net = self.net.to(self.device)
            
        self.loss = Unet_Loss()
        if 'gamma' in self.dst['command']:
            self.loss = Unet_Loss(charbonnier=True, use_gamma=True)
            log('Enable Gamma & Charbonnier Loss Mode...')
        torch.backends.cudnn.benchmark = True
    
    def change_eval_dst(self, mode='eval'):
        self.dst = self.args[f'dst_{mode}']
        self.dstname = self.dst['dstname']
        self.dst_eval = globals()[self.dst['dataset']](self.dst)
        self.dataloader_eval = DataLoader(self.dst_eval, batch_size=1, shuffle=False, 
                                    num_workers=self.args['num_workers'], pin_memory=False)

    def train(self):
        pf = self.hyper['plot_freq']
        # self.scheduler.step()
        lr = self.scheduler.get_last_lr()[0]
        for epoch in range(self.hyper['last_epoch']+1, self.hyper['stop_epoch']+1):
            # log init
            if self.multi_gpu: torch.distributed.barrier()
            self.net.train()
            self.train_psnr.reset()
            runtime = {'preprocess':0, 'dataloader':0, 'net':0, 'bp':0, 'metric':0, 'total':1e-9}
            time_points = [0] * 10
            time_points[0] = time.time()
            if 'consistency' in self.dst['command']:
                ema_net = copy.deepcopy(self.net).requires_grad_(False)
            with tqdm(total=len(self.dataloader_train)) as t:                
                for k, data in enumerate(self.dataloader_train):
                    runtime['dataloader'] += timestamp(time_points, 1)
                    # Preprocess
                    imgs_lr, imgs_hr, sigma = self.preprocess(data, mode='train', preprocess=True)
                    runtime['preprocess'] += timestamp(time_points, 2)
                    
                    # 训练
                    self.optimizer.zero_grad()
                    if 'guided' in self.args['arch']:
                        pred = self.net(imgs_lr, sigma)
                        if 'consistency' in self.dst['command'] and epoch>100:
                            sigma_t = np.random.rand() * 0.25 + 0.7
                            noise = torch.randn_like(imgs_hr, device=self.device) * sigma
                            imgs_lr2 = imgs_hr + noise * sigma_t
                            with torch.no_grad():
                                pred2 = ema_net(imgs_lr2, sigma * sigma_t)
                    else:
                        pred = self.net(imgs_lr)

                    runtime['net'] += timestamp(time_points, 3)
                    loss = self.loss(pred, imgs_hr)
                    if 'consistency' in self.dst['command'] and epoch>100:
                        loss += 0.1 * torch.mean(torch.abs(pred - pred2))
                    loss.backward()
                    self.optimizer.step()
                    runtime['bp'] += timestamp(time_points, 4)

                    # 更新tqdm的参数
                    with torch.no_grad():
                        pred = torch.clamp(pred, 0, 1)
                        imgs_hr = torch.clamp(imgs_hr, 0, 1)
                        psnr = PSNR_Loss(pred, imgs_hr)
                        self.train_psnr.update(psnr.item())
                    
                    runtime['total'] = runtime['preprocess']+runtime['dataloader']+runtime['net']+runtime['bp']
                    if self.local_rank == 0:
                        t.set_description(f'Epoch {epoch}')
                        t.set_postfix({'lr':f"{lr:.2e}", 'PSNR':f"{self.train_psnr.avg:.2f}",
                                        'loader':f"{100*runtime['dataloader']/runtime['total']:.1f}%",
                                        'process':f"{100*runtime['preprocess']/runtime['total']:.1f}%",
                                        'net':f"{100*runtime['net']/runtime['total']:.1f}%",
                                        'bp':f"{100*runtime['bp']/runtime['total']:.1f}%",})
                        t.update(1)
                        if k % 100 == 0:
                            inputs = imgs_lr[0].detach().cpu().numpy().clip(0,1)
                            output = pred[0].detach().cpu().numpy()
                            target = imgs_hr[0].detach().cpu().numpy()
                            pck = (4-data['pattern'][0].item())%4
                            inputs = bayer_aug(inputs.transpose(1,2,0), k=pck)
                            output = bayer_aug(output.transpose(1,2,0), k=pck)
                            target = bayer_aug(target.transpose(1,2,0), k=pck)
                            temp_img = np.concatenate((inputs, output, target),axis=1)
                            wb = data['wb'][0].numpy().reshape(-1)
                            ccm = data['ccm'][0].numpy().reshape(3,3)
                            temp_img = FastISP(temp_img, wb=wb, ccm=ccm)
                            filename = os.path.join(self.sample_dir, 'temp', f'temp_{epoch//pf*pf:04d}.png')
                            cv2.imwrite(filename, np.uint8(temp_img[:,:,::-1]*255))
                    time_points[0] = time.time()

            # 更新学习率
            self.scheduler.step()
            lr = self.scheduler.get_last_lr()[0]
            if self.multi_gpu: torch.distributed.barrier()

            if self.local_rank == 0:
                # 存储模型
                if epoch % self.hyper['save_freq'] == 0:
                    model_dict = self.net.module.state_dict() if self.multi_gpu else self.net.state_dict()
                    epoch_id = epoch // pf * pf
                    save_path = os.path.join(self.model_dir, '%s_e%04d.pth'% (self.model_name, epoch_id))
                    torch.save(model_dict, save_path)
                    torch.save(model_dict, f'{self.fast_ckpt}/{self.model_name}_last_model.pth')
                
                # 输出过程量，随时看
                savefile = os.path.join(self.sample_dir, f'{self.model_name}_train_psnr.jpg')
                logfile = os.path.join(self.sample_dir, f'{self.model_name}_train_psnr.pkl')
                self.train_psnr.plot_history(savefile=savefile, logfile=logfile)
                # if epoch % self.hyper['plot_freq'] == 0:
                
                if self.save_plot:
                    inputs = imgs_lr[0].detach().cpu().numpy().clip(0,1)
                    output = pred[0].detach().cpu().numpy()
                    target = imgs_hr[0].detach().cpu().numpy()
                    pck = (4-data['pattern'][0].item())%4
                    inputs = bayer_aug(inputs.transpose(1,2,0), k=pck)
                    output = bayer_aug(output.transpose(1,2,0), k=pck)
                    target = bayer_aug(target.transpose(1,2,0), k=pck)
                    temp_img = np.concatenate((inputs, output, target),axis=1)
                    wb = data['wb'][0].numpy().reshape(-1)
                    ccm = data['ccm'][0].numpy().reshape(3,3)
                    temp_img = FastISP(temp_img, wb=wb, ccm=ccm)
                    filename = os.path.join(self.sample_dir, 'temp', f'temp_{epoch//pf*pf:04d}.png')
                    cv2.imwrite(filename, np.uint8(temp_img[:,:,::-1]*255))

                # fast eval
                if epoch % self.hyper['plot_freq'] == 0:
                    log(f"learning_rate: {lr:.3e}")
                    self.dst_eval.sigma = self.args['dst_eval']['sigma_list'][1] / 255.
                    self.eval(epoch=epoch)
                    model_dict = self.net.module.state_dict() if self.multi_gpu else self.net.state_dict()
                    torch.save(model_dict, f'{self.fast_ckpt}/{self.model_name}_last_model.pth')

    def eval(self, epoch=-1):
        self.net.eval()
        self.metrics_reset()
        # record every metric
        metrics = {}
        metrics_path = f'./metrics/{self.model_name}_metrics.pkl'
        if os.path.exists(metrics_path):
            with open(metrics_path, 'rb') as f:
                metrics = pkl.load(f)
        # multiprocess
        if epoch > 0:
            pool = []
        else:
            pool = ProcessPoolExecutor(max_workers=max(4, self.args['num_workers']))
        task_list = []
        save_plot = self.save_plot
        with tqdm(total=len(self.dataloader_eval)) as t:
            for k, data in enumerate(self.dataloader_eval):
                # 由于crops的存在，Dataloader会把数据变成5维，需要view回4维
                imgs_lr, imgs_hr, sigma = self.preprocess(data, mode='eval', preprocess=False)
                wb = data['wb'][0].numpy().reshape(-1)
                ccm = data['ccm'][0].numpy().reshape(3,3)
                name = data['name'][0] + f'_sig{int(sigma.item()*255)}'

                with torch.no_grad():
                    # 扛得住就pad再crop
                    if imgs_lr.shape[-1] % 16 != 0:
                        p2d = (4,4,4,4)
                        imgs_lr = F.pad(imgs_lr, p2d, mode='reflect')
                        imgs_dn = self.net(imgs_lr)
                        imgs_lr = imgs_lr[..., 4:-4, 4:-4]
                        imgs_dn = imgs_dn[..., 4:-4, 4:-4]
                    else:
                        if 'guided' in self.args['arch']:
                            imgs_dn = self.net(imgs_lr, sigma)
                        else:
                            imgs_dn = self.net(imgs_lr)
                    
                    imgs_lr = torch.clamp(imgs_lr, 0, 1)
                    imgs_dn = torch.clamp(imgs_dn, 0, 1)

                    # PSNR & SSIM (Raw domain)
                    output = tensor2im(imgs_dn)
                    target = tensor2im(imgs_hr)
                    res = quality_assess(output, target, data_range=255)
                    raw_metrics = [res['PSNR'], res['SSIM']]
                    self.eval_psnr.update(res['PSNR'])
                    self.eval_ssim.update(res['SSIM'])
                    metrics[name] = raw_metrics
                    # convert raw to rgb
                    if save_plot:
                        inputs = tensor2im(imgs_lr)
                        res = quality_assess(inputs, target, data_range=255)
                        raw_metrics = [res['PSNR'], res['SSIM']] + raw_metrics
                        pck = (4-data['pattern'][0].item())%4
                        imgs_lr = bayer_aug(imgs_lr[0].detach().cpu().numpy().transpose(1,2,0), k=pck)
                        imgs_dn = bayer_aug(imgs_dn[0].detach().cpu().numpy().transpose(1,2,0), k=pck)
                        imgs_hr = bayer_aug(imgs_hr[0].detach().cpu().numpy().transpose(1,2,0), k=pck)
                        if epoch > 0:
                            pool.append(threading.Thread(target=self.multiprocess_plot, args=(imgs_lr, imgs_dn, imgs_hr, 
                                    wb, ccm, name, save_plot, epoch, raw_metrics, k)))
                            pool[k].start()
                        else:
                            inputs = FastISP(imgs_lr, wb, ccm)
                            output = FastISP(imgs_dn, wb, ccm)
                            target = FastISP(imgs_hr, wb, ccm)
                            # raw_metrics = None # 用RGB metrics
                            task_list.append(
                                pool.submit(plot_sample, inputs, output, target, 
                                    filename=name, save_plot=save_plot, epoch=epoch,
                                    model_name=self.model_name, save_path=self.sample_dir,
                                    res=raw_metrics
                                    )
                                )

                    t.set_description(f'{name}')
                    t.set_postfix({'PSNR':f"{self.eval_psnr.avg:.2f}"})
                    t.update(1)

        if save_plot:
            if epoch > 0:
                for i in range(len(pool)):
                    pool[i].join()
            else:
                pool.shutdown(wait=True)
                for task in as_completed(task_list):
                    psnr, ssim, name = task.result()
                    metrics[name] = (psnr[1], ssim[1])
                    self.eval_psnr_lr.update(psnr[0])
                    self.eval_psnr_dn.update(psnr[1])
                    self.eval_ssim_lr.update(ssim[0])
                    self.eval_ssim_dn.update(ssim[1])
        else:
            self.eval_psnr_dn = self.eval_psnr
            self.eval_ssim_dn = self.eval_ssim

        # 超过最好记录才保存
        if self.eval_psnr.avg >= self.best_psnr and epoch > 0:
            self.best_psnr = self.eval_psnr.avg
            log(f"Best PSNR is {self.best_psnr} now!!")
            model_dict = self.net.module.state_dict() if self.multi_gpu else self.net.state_dict()
            torch.save(model_dict, f'{self.fast_ckpt}/{self.model_name}_best_model.pth')

        log(f"Epoch {epoch}: PSNR={self.eval_psnr.avg:.2f}, SSIM={self.eval_ssim.avg:.4f}\n"
            +f"psnrs_lr={self.eval_psnr_lr.avg:.2f}, psnrs_dn={self.eval_psnr_dn.avg:.2f}"
            +f"\nssims_lr={self.eval_ssim_lr.avg:.4f}, ssims_dn={self.eval_ssim_dn.avg:.4f}",
            log=f'./logs/log_{self.model_name}.log')
        if epoch < 0:
            with open(metrics_path, 'wb') as f:
                pkl.dump(metrics, f)
        savefile = os.path.join(self.sample_dir, f'{self.model_name}_eval_psnr.jpg')
        logfile = os.path.join(self.sample_dir, f'{self.model_name}_eval_psnr.pkl')
        if epoch > 0:
            self.eval_psnr.plot_history(savefile=savefile, logfile=logfile)
        del pool
        plt.close('all')
        gc.collect()
        return metrics
    
    def multiprocess_plot(self, imgs_lr, imgs_dn, imgs_hr, wb, ccm, name, save_plot, epoch, raw_metrics, k):
        # if self.infos is None:
        inputs = FastISP(imgs_lr, wb, ccm)
        output = FastISP(imgs_dn, wb, ccm)
        target = FastISP(imgs_hr, wb, ccm)
        
        psnr, ssim, _ = plot_sample(inputs, output, target, 
                        filename=name, 
                        save_plot=save_plot, epoch=epoch,
                        model_name=self.model_name,
                        save_path=self.sample_dir,
                        res=raw_metrics)
        self.eval_psnr_lr.update(psnr[0])
        self.eval_psnr_dn.update(psnr[1])
        self.eval_ssim_lr.update(ssim[0])
        self.eval_ssim_dn.update(ssim[1])

    def predict(self, raw, name='ds'):
        self.net.eval()
        img_lr = raw2bayer(raw+self.dst["bl"])[None, ...]
        img_lr = torch.from_numpy(img_lr)
        img_lr = img_lr.type(torch.FloatTensor).to(self.device)
        with torch.no_grad():
            croped_imgs_lr = self.dst_eval.eval_crop(img_lr)
            croped_imgs_dn = []
            for img_lr in tqdm(croped_imgs_lr):
                img_dn = self.net(img_lr)
                croped_imgs_dn.append(img_dn)
            croped_imgs_dn = torch.cat(croped_imgs_dn)
            img_dn = self.dst_eval.eval_merge(croped_imgs_dn)
            img_dn = img_dn
            img_dn = img_dn[0].detach().cpu().numpy()
        np.save(f'{name}.npy', img_dn)
    
    def preprocess(self, data, mode='train', preprocess=True):
        imgs_hr = data['hr'].type(torch.FloatTensor).to(self.device)
        imgs_lr = data['lr'].type(torch.FloatTensor).to(self.device)
        imgs_hr = tensor_dimxto4(imgs_hr)
        imgs_lr = tensor_dimxto4(imgs_lr)
        # self.use_gpu = True
        dst = self.dst_train if mode=='train' else self.dst_eval

        if self.use_gpu and mode=='train' and preprocess:
            b = imgs_lr.shape[0]
            raise NotImplementedError
        else: # mode == 'eval'
            pass
        
        sigma = data['sigma'].type(torch.FloatTensor).to(self.device)
        sigma = sigma.view(-1,1,1,1)
        
        if self.dst['clip']:
            imgs_lr = imgs_lr.clamp(0, 1)
            imgs_hr = imgs_hr.clamp(0, 1)
        return imgs_lr, imgs_hr, sigma

class AWGN_Parser():
    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    def parse(self):
        self.parser.add_argument('--runfile', '-f', default="runfiles/Gaussian/DiffusionUnet_5to100.yml", type=Path, help="path to config")
        self.parser.add_argument('--mode', '-m', default='train', type=str, help="train or test")
        self.parser.add_argument('--debug', action='store_true', default=False, help="debug or not")
        self.parser.add_argument('--nofig', action='store_true', default=False, help="don't save_plot")
        self.parser.add_argument('--nohost', action='store_true', default=False, help="don't save_plot")
        self.parser.add_argument('--gpu', default="0", help="os.environ['CUDA_VISIBLE_DEVICES']")
        self.parser.add_argument("--local_rank", type=int, default=0,) 
        return self.parser.parse_args()

if __name__ == '__main__':
    trainer = AWGN_Trainer()
    if trainer.mode == 'train':
        trainer.train()
        savefile = os.path.join(trainer.sample_dir, f'{trainer.model_name}_train_psnr.jpg')
        logfile = os.path.join(trainer.sample_dir, f'{trainer.model_name}_train_psnr.pkl')
        trainer.train_psnr.plot_history(savefile=savefile, logfile=logfile)
        trainer.eval_psnr.plot_history(savefile=os.path.join(trainer.sample_dir, f'{trainer.model_name}_eval_psnr.jpg'))
        trainer.mode = 'evaltest'
    # best_model
    best_model_path = os.path.join(f'{trainer.fast_ckpt}', f'{trainer.model_name}_best_model.pth')
    if os.path.exists(best_model_path) is False: 
        best_model_path = os.path.join(f'{trainer.fast_ckpt}',f'{trainer.model_name}_last_model.pth')
    best_model = torch.load(best_model_path, map_location=trainer.device)
    trainer.net = load_weights(trainer.net, best_model, multi_gpu=trainer.multi_gpu, by_name=True)

    if 'eval' in trainer.mode and trainer.local_rank==0:
        trainer.change_eval_dst('eval')
        for sigma in trainer.args['dst_test']['sigma_list']:
            log(f'AWGN Datasets: sigma={sigma}',log=f'./logs/log_{trainer.model_name}.log')
            trainer.dst_eval.sigma = sigma / 255.
            metrics = trainer.eval(-1)
    log(f'Metrics have been saved in ./metrics/{trainer.model_name}_metrics.pkl')