from .isp_ops import *
import sklearn.linear_model as lm

# sigma是σ_read, gain是K
def VST(x, sigma, mu=0, gain=1.0):
    # 无增益时，y = 2 * np.sqrt(x + 3.0 / 8.0 + sigma ** 2)
    fz = gain*x + (3/8)*gain**2 + sigma**2 - gain*mu
    fz = torch.maximum(fz, torch.zeros_like(fz)) if torch.is_tensor(fz) else np.maximum(fz,0)
    fz = 2/gain * fz**0.5
    # y = x * wp
    # y = gain * x + (gain ** 2) * 3.0 / 8.0 + sigma ** 2 - gain * mu
    # y = np.sqrt(np.maximum(y, np.zeros_like(y)))
    # y = (2.0 / gain) * y / wp
    return fz

# sigma是σ_read, gain是K
def inverse_VST(z, sigma, gain=1, exact=False):
    # 如果未提供alpha参数，则假定alpha等于1
    sigma = sigma / gain
    if exact:
        if z.min() <= 0:
            fz = z
            fz[fz<=0] = 0
            z = z[z>0]
            fz[fz>0] = (z/2)**2 + (1/4)*((3/2)**0.5)*z**(-1) - (11/8)*z**(-2) + (5/8)*((3/2)**0.5)*z**(-3) - 1/8 - sigma**2
        else:
            fz = (z/2)**2 + (1/4)*((3/2)**0.5)*z**(-1) - (11/8)*z**(-2) + (5/8)*((3/2)**0.5)*z**(-3) - 1/8 - sigma**2  # 闭式精确无偏反变换的近似
    else:
        fz = (z/2)**2 - 3.0/8.0 - sigma**2
    fz = torch.maximum(fz, torch.zeros_like(fz)) if torch.is_tensor(fz) else np.maximum(fz,0)
    # 恢复初始变量变换
    fz = fz * gain
    return fz

# sigma是σ_read, gain是K
def inverse_VST_torch(z, sigma, gain=1, exact=False):
    # 如果未提供alpha参数，则假定alpha等于1
    sigma = sigma / gain
    if exact:
        fz = (z/2)**2 + (1/4)*np.sqrt(3/2)*z**(-1) - (11/8)*z**(-2) + (5/8)*np.sqrt(3/2)*z**(-3) - 1/8 - sigma**2  # 闭式精确无偏反变换的近似
    else:
        fz = (z/2)**2 - 3.0/8.0 - sigma**2
    fz = np.maximum(0, fz)  # 确保结果非负
    # 恢复初始变量变换
    fz = fz * gain

    return fz

def getGsP(lam, K, sigGs, r=5, pho=1, clip=False, show=False):
    l = 2 * pho * r + 1
    x = np.linspace(-r, r, l)
    x_conv = np.linspace(-2*r, 2*r, 2*l-1)
    Ps_pmf = poisson.pmf(x, lam/K)
    if sigGs > 0:
        Gs_pmf = norm.pdf(x, loc=0, scale=sigGs/K) #* bw
        Conv_pdf = convolve(Ps_pmf, Gs_pmf, mode='same')
    else:
        Gs_pmf = np.zeros_like(x)
        Gs_pmf[r] = 1
        Conv_pdf = poisson.pmf(x, lam/K)
        # Conv_pdf = convolve(Ps_pmf, Gs_pmf, mode='full')
    Conv_pdf[Conv_pdf<0] = 0
    if clip:
        Conv_pdf[r*pho] += Conv_pdf[:r*pho].sum()
        Conv_pdf[:r*pho] = 0
    Conv_pdf = Conv_pdf / (Conv_pdf.sum() / pho)
    # print(refill)
    # if lam - int(lam) < 0.01: 
    #     show=True
    # show = True
    if show:
        print(lam, np.sum(Conv_pdf*x*K)/pho)
        # plt.plot(x_conv*K, Conv_pdf, 'C0', zorder=10)
        # Conv_pdf = convolve(Ps_pmf, Gs_pmf, mode='same')
        plt.plot(x*K, Conv_pdf, 'C0', zorder=11)
        plt.plot(x*K, Ps_pmf, 'C1')
        plt.plot(x*K, Gs_pmf, 'C2')
        # plt.scatter(lams, Ex)
        plt.grid('on')
        plt.xlim(0,r*pho)
        plt.show()
    return x, Conv_pdf

def close_form_bias(x, sigGs=25.853043, K=24.48128):
    # z = z / alpha
    # sigma = sigma / alpha
    # fz = 2 * np.sqrt(np.maximum(0, z + (3/8) + sigma**2))
    y = x / K
    sigma = sigGs / K
    y_hat = y + 3/8 + sigma**2
    m1 = (y+sigma**2)/y_hat**2
    m2 = (y)/y_hat**3
    m3 = ((y)+3*(y+sigma**2)**2)/y_hat**4
    bias = 2*y_hat**0.5*(-1/8*m1+1/16*m2-5/128*m3)

    return bias

def get_bias(img=None, sigGs=25.853043, K=24.48128, pho_min=1, post=False, clip=False, close_form=True, show=False):
    # H, W, C = img.shape
    # lb, ub = np.floor(img.min()), np.ceil(img.max())+1
    lb, ub = 0, np.ceil(img.max())+1
    # 精度分段下降，提速
    if ub < 50:
        lams = np.linspace(lb, ub, int((ub-lb)/0.1)+2)
    elif ub < 500:
        lams = np.concatenate((np.linspace(lb, 50, int((50-lb)/0.1)+1), np.linspace(50, ub, int(ub-50)+2)))
    else:
        lams = np.concatenate((np.linspace(lb, 50, int((50-lb)/0.1)+1), np.linspace(50, 500, 451), np.linspace(500, ub, int(ub-500)//10+2)))
    n = len(lams)
    bias = np.zeros(n, np.float32)
    # 单位间隔的采样率
    pho = np.maximum(int(K**0.5), pho_min)
    # 闭式解加速
    if close_form:
        th = 50 * K if K<1 else 50 * K**0.5 # 经验阈值
        bias[lams>th] = close_form_bias(lams[lams>th], sigGs, K) # Foi, TIP-13
    else:
        th = lams.max()+1   # 无效化
    # 蒙特卡洛方法，计算泊松高斯分布的概率密度函数与采样点，然后数值积分计算偏移量
    for i, lam in enumerate(lams[lams<=th]):
        # 蒙特卡洛方法，计算泊松高斯分布的概率密度函数与采样点
        # r:卷积半径，pho单位间隔的采样率，clip是否零裁剪
        x, p = getGsP(lam, K, sigGs, r=int(lam*(1/K)*2+sigGs*2+lam+10), pho=pho, clip=clip)
        # 概率密度随x增大而收敛于0，所以可以有限区域的数值积分
        bias[i] = np.sum(p * VST(K*x,sigGs,gain=K)/pho) - VST(lam,sigGs,gain=K)
    # 制作LUT表
    # func = interp1d(lams+bias, bias) if post else interp1d(lams, bias)
    func = interp1d(lams, bias)
    # # 将图像映射为bias矫正map
    # M_map = func(img).astype(np.float32)
    # if show:
    #     plt.plot(lams, bias, label='bias')
    #     plt.plot(lams+bias, bias, alpha=0.5, label='post bias (-)')
    #     plt.legend()
    #     plt.show()
    #     plt.imshow(M_map.mean(-1))
    #     plt.colorbar()
    #     plt.show()
    # M_map = torch.from_numpy(M_map).to('cuda:0')
    return func

def get_bias_points(lams, K, sigGs, pho_min=100, close_form=False, clip=False):
    bias = np.zeros_like(lams)
    # 单位间隔的采样率
    pho = np.maximum(int(K**0.5), pho_min)
    # 闭式解加速
    if close_form:
        th = 50 * K if K<1 else 50 * K**0.5 # 经验阈值
        bias[lams>th] = close_form_bias(lams[lams>th], sigGs, K) # Foi, TIP-13
    else:
        th = lams.max()+1   # 无效化
    # 蒙特卡洛方法，计算泊松高斯分布的概率密度函数与采样点，然后数值积分计算偏移量
    lams = lams[lams<=th]
    for i, lam in enumerate(lams):
        # 蒙特卡洛方法，计算泊松高斯分布的概率密度函数与采样点
        # r:卷积半径，pho单位间隔的采样率，clip是否零裁剪
        x, p = getGsP(lam, K, sigGs, r=int(lam*(1/K)*2+sigGs*2+lam+10), pho=pho, clip=clip)
        # 概率密度随x增大而收敛于0，所以可以有限区域的数值积分
        bias[i] = np.sum(p * VST(K*x,sigGs,gain=K)/pho) - VST(lam,sigGs,gain=K)
    return bias

class BiasLUT:
    def __init__(self, lut_path='checkpoints/bias_lut_2d.npy'):
        # 加载预先计算好的bias_lut数据
        self.bias_lut = np.load(lut_path)
        
        # 初始化 x_lut 和 sg_lut
        sp=128
        self.x_lut = np.concatenate((
            np.linspace(0, 2**-4, sp, endpoint=False),
            np.exp(np.linspace(np.log(2**(-4)), np.log(2**10), 14*sp+1))
        ))
        
        self.sg_lut = np.concatenate((
            np.linspace(0, 1, 200, endpoint=False),
            np.linspace(1, 10, 901)
        ))
    
    def pos_interp(self, data, x):
        data = np.concatenate(([-np.inf,], data))
        idx = np.searchsorted(data, x).clip(0,len(data)-1)
        w = data[idx] - x
        diff = data[idx] - data[idx-1]
        delta = w / diff
        idx_interp = idx - delta
        return idx_interp - 1

    def data_merge(self, data, pos):
        pos = pos.clip(0, len(self.x_lut)-1)
        l = np.int32(np.floor(pos))
        r = np.int32(np.ceil(pos))
        weight_r = pos - l
        weight_l = 1 - weight_r
        return data[..., l] * weight_l + data[..., r] * weight_r

    def get_lut(self, x, K=1, sigGs=2, func=False):
        xe = x / K # 光电子数，单位(e-)
        sg = sigGs / K # 电子意义上的读出噪声，单位(e-)
        sg_pos = self.pos_interp(self.sg_lut, sg)
        sg_len = len(self.sg_lut)
        x_len = len(self.x_lut)

        # 检查sg_pos是否超出查找表范围，并处理超出范围的sg_pos
        if sg_pos >= sg_len:
            if func:
                return get_bias(x, K=K, sigGs=sigGs, close_form=True)
            else:
                if x.size > 1000: # 超过1k个点，搞成函数更快
                    bias_func = get_bias(x, K=K, sigGs=sigGs, close_form=True)
                    return bias_func(x)
                else:
                    return get_bias_points(x.reshape(-1), K, sigGs, close_form=True).reshape(*x.shape)
            
        if func:
            if sg_pos >= sg_len or np.any(x_pos >= x_len):
                return get_bias(x, K=K, sigGs=sigGs, close_form=True)
            else:
                data = self.data_merge(self.bias_lut.reshape(-1,sg_len), sg_pos)
                return interp1d(self.x_lut, data, kind='linear', fill_value=0)
        else:
            x_pos = self.pos_interp(self.x_lut, xe)
            if sg_pos >= sg_len:
                return get_bias_points(x, K, sigGs, close_form=True)
            else:
                data = self.data_merge(self.bias_lut.reshape(-1,sg_len), sg_pos)
                bias = self.data_merge(data[None], x_pos)[0]
                # 处理x_pos超出查找表范围的情况
                if np.any(x_pos>=x_len):
                    if len(bias.shape) == 0: bias = np.array([bias])
                    bias[x_pos>=x_len] = get_bias_points(x[x_pos>=x_len], K, sigGs, close_form=True)
                return bias

# 快速计算空域标准差
def stdfilt(img, k=5):
    # 计算均值图像和均值图像的平方图像
    img_blur = cv2.blur(img, (k, k))
    result_1 = img_blur ** 2
    # 计算图像的平方和平方后的均值
    img_2 = img ** 2
    result_2 = cv2.blur(img_2, (k, k))
    result = np.sqrt(np.maximum(result_2 - result_1, 0))
    return result

# 快速计算空域方差
def varfilt(img, k=5):
    # 计算均值图像和均值图像的平方图像
    img_blur = cv2.blur(img, (k, k))
    result_1 = img_blur ** 2
    # 计算图像的平方和平方后的均值
    img_2 = img ** 2
    result_2 = cv2.blur(img_2, (k, k))
    result = result_2 - result_1
    return result

# 存在图像内容时，空域平方计算的是平方和，我们拟合噪声曲线需要的是和的平方，存在比例差异，需要矫正
def var_corr(img, k=5):
    # 计算均值图像和均值图像的平方图像
    img_blur = cv2.blur(img, (k, k))
    result_1 = img_blur ** 2
    # 计算图像的平方和平方后的均值
    img_2 = img ** 2
    result_2 = cv2.blur(img_2, (k, k))
    ratio = result_1 / result_2
    return ratio

def Blur1D(data, c=0.5, log=True):
    l = len(data)
    if log:
        data = np.log2(data)
    temp = data.copy()
    for i in range(1, l-1):
        data[i] = temp[i] * c + (temp[i-1] + temp[i+1]) * (1-c)/2
    if log:
        data = 2 ** data 
    return data

def FastGuidedFilter(p,I,d=7,eps=1):
    p_lr = cv2.resize(p, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
    I_lr = cv2.resize(I, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
    mu_p = cv2.boxFilter(p_lr, -1, (d, d)) 
    mu_I = cv2.boxFilter(I_lr,-1, (d, d)) 
    
    II = cv2.boxFilter(np.multiply(I_lr,I_lr), -1, (d, d)) 
    Ip = cv2.boxFilter(np.multiply(I_lr,p_lr), -1, (d, d))
    
    var = II-np.multiply(mu_I,mu_I)
    cov = Ip-np.multiply(mu_I,mu_p)
    
    a = cov / (var + eps)
    
    b = mu_p - np.multiply(a,mu_I)
    mu_a = cv2.resize(cv2.boxFilter(a, -1, (d, d)), None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR) 
    mu_b = cv2.resize(cv2.boxFilter(b, -1, (d, d)), None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR) 
    
    dstImg = np.multiply(mu_a, I) + mu_b
    
    return dstImg

def GuidedFilter(p,I,d=7,eps=1):
    mu_p = cv2.boxFilter(p, -1, (d, d), borderType=cv2.BORDER_REPLICATE) 
    mu_I = cv2.boxFilter(I,-1, (d, d), borderType=cv2.BORDER_REPLICATE) 
    
    II = cv2.boxFilter(np.multiply(I,I), -1, (d, d), borderType=cv2.BORDER_REPLICATE) 
    Ip = cv2.boxFilter(np.multiply(I,p), -1, (d, d), borderType=cv2.BORDER_REPLICATE)
    
    var = II-np.multiply(mu_I,mu_I)
    cov = Ip-np.multiply(mu_I,mu_p)
    
    a = cov / (var + eps)
    
    b = mu_p - np.multiply(a,mu_I)
    mu_a = cv2.boxFilter(a, -1, (d, d), borderType=cv2.BORDER_REPLICATE)
    mu_b = cv2.boxFilter(b, -1, (d, d), borderType=cv2.BORDER_REPLICATE)
    
    dstImg = np.multiply(mu_a, I) + mu_b
    
    return dstImg

def row_denoise(path, iso, data=None):
    if data is None:
        raw = dataload(path)
    else:
        raw = data
    raw = bayer2rows(raw)
    raw_denoised = raw.copy()
    for i in range(len(raw)):
        rows = raw[i].mean(axis=1)
        rows2 = rows.reshape(1, -1)
        rows2 = cv2.bilateralFilter(rows2, 25, sigmaColor=10, sigmaSpace=1+iso/200, borderType=cv2.BORDER_REPLICATE)[0]
        row_diff = rows-rows2
        raw_denoised[i] = raw[i] - row_diff.reshape(-1, 1)
    raw = rows2bayer(raw)
    raw_denoised = rows2bayer(raw_denoised)
    return raw_denoised

# def is_model_valid(model, X, y):
#     # 检查模型的预测结果是否全为正数
#     if model.coef_[0] <= 0:
#         return False
#     if model.intercept_ <= 0:
#         model.intercept_ = model.coef_[0] ** 2
#         warnings.warn('intercept_ <= 0, set intercept_ = coef_[0]^2')
#     return True

def polyfit(x, y, ransac=False,clip=False):
    setup_seed(2024)
    # 不用饱和点
    nonsat = np.logical_and(x>1e-4, x<0.8)
    if len(x[nonsat==True]) > 0.01 * len(x.reshape(-1)):
        x, y = x[nonsat==True], y[nonsat==True]
    # 最小二乘法，np的有bug，scipy的没bug
    X = np.vstack([x, np.ones(len(x))]).T
    if ransac: # 基于RANSAC的线性模型拟合
        ransac = lm.RANSACRegressor(min_samples=int(np.sqrt(len(x))))
        ransac.fit(X, y)
        # 提取拟合结果
        # inliers = ransac.inlier_mask_
        # outliers = np.logical_not(inliers)
        # 提取最佳模型参数
        best_slope = ransac.estimator_.coef_[0]
        best_intercept = ransac.estimator_.intercept_
        res = (best_slope, best_intercept)
    else: # 最小二乘
        res, loss, rank, s = scipy.linalg.lstsq(X, y)
    return res