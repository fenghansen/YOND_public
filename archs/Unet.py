from .modules import *

# SID Unet
class UNetSeeInDark(nn.Module):
    def __init__(self, args=None):
        super().__init__()
        self.args = args
        self.nframes = args['nframes']
        self.cf = 0
        self.res = args['res']
        self.norm = args['norm'] if 'norm' in args else False
        nframes = self.args['nframes'] if 'nframes' in args else 1
        nf = args['nf']
        in_nc = args['in_nc']
        out_nc = args['out_nc']

        self.conv1_1 = nn.Conv2d(in_nc*nframes, nf, kernel_size=3, stride=1, padding=1)
        self.conv1_2 = nn.Conv2d(nf, nf, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        
        self.conv2_1 = nn.Conv2d(nf, nf*2, kernel_size=3, stride=1, padding=1)
        self.conv2_2 = nn.Conv2d(nf*2, nf*2, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        
        self.conv3_1 = nn.Conv2d(nf*2, nf*4, kernel_size=3, stride=1, padding=1)
        self.conv3_2 = nn.Conv2d(nf*4, nf*4, kernel_size=3, stride=1, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2)
        
        self.conv4_1 = nn.Conv2d(nf*4, nf*8, kernel_size=3, stride=1, padding=1)
        self.conv4_2 = nn.Conv2d(nf*8, nf*8, kernel_size=3, stride=1, padding=1)
        self.pool4 = nn.MaxPool2d(kernel_size=2)
        
        self.conv5_1 = nn.Conv2d(nf*8, nf*16, kernel_size=3, stride=1, padding=1)
        self.conv5_2 = nn.Conv2d(nf*16, nf*16, kernel_size=3, stride=1, padding=1)
        
        self.upv6 = nn.ConvTranspose2d(nf*16, nf*8, 2, stride=2)
        self.conv6_1 = nn.Conv2d(nf*16, nf*8, kernel_size=3, stride=1, padding=1)
        self.conv6_2 = nn.Conv2d(nf*8, nf*8, kernel_size=3, stride=1, padding=1)
        
        self.upv7 = nn.ConvTranspose2d(nf*8, nf*4, 2, stride=2)
        self.conv7_1 = nn.Conv2d(nf*8, nf*4, kernel_size=3, stride=1, padding=1)
        self.conv7_2 = nn.Conv2d(nf*4, nf*4, kernel_size=3, stride=1, padding=1)
        
        self.upv8 = nn.ConvTranspose2d(nf*4, nf*2, 2, stride=2)
        self.conv8_1 = nn.Conv2d(nf*4, nf*2, kernel_size=3, stride=1, padding=1)
        self.conv8_2 = nn.Conv2d(nf*2, nf*2, kernel_size=3, stride=1, padding=1)
        
        self.upv9 = nn.ConvTranspose2d(nf*2, nf, 2, stride=2)
        self.conv9_1 = nn.Conv2d(nf*2, nf, kernel_size=3, stride=1, padding=1)
        self.conv9_2 = nn.Conv2d(nf, nf, kernel_size=3, stride=1, padding=1)
        
        self.conv10_1 = nn.Conv2d(nf, out_nc, kernel_size=1, stride=1)
        self.relu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        if self.norm:
            x, lb, ub = data_normalize(x)
        conv1 = self.relu(self.conv1_1(x))
        conv1 = self.relu(self.conv1_2(conv1))
        pool1 = self.pool1(conv1)
        
        conv2 = self.relu(self.conv2_1(pool1))
        conv2 = self.relu(self.conv2_2(conv2))
        pool2 = self.pool1(conv2)
        
        conv3 = self.relu(self.conv3_1(pool2))
        conv3 = self.relu(self.conv3_2(conv3))
        pool3 = self.pool1(conv3)
        
        conv4 = self.relu(self.conv4_1(pool3))
        conv4 = self.relu(self.conv4_2(conv4))
        pool4 = self.pool1(conv4)
        
        conv5 = self.relu(self.conv5_1(pool4))
        conv5 = self.relu(self.conv5_2(conv5))
        
        up6 = self.upv6(conv5)
        up6 = torch.cat([up6, conv4], 1)
        conv6 = self.relu(self.conv6_1(up6))
        conv6 = self.relu(self.conv6_2(conv6))
        
        up7 = self.upv7(conv6)
        up7 = torch.cat([up7, conv3], 1)
        conv7 = self.relu(self.conv7_1(up7))
        conv7 = self.relu(self.conv7_2(conv7))
        
        up8 = self.upv8(conv7)
        up8 = torch.cat([up8, conv2], 1)
        conv8 = self.relu(self.conv8_1(up8))
        conv8 = self.relu(self.conv8_2(conv8))
        
        up9 = self.upv9(conv8)
        up9 = torch.cat([up9, conv1], 1)
        conv9 = self.relu(self.conv9_1(up9))
        conv9 = self.relu(self.conv9_2(conv9))
        
        out = self.conv10_1(conv9)
        if self.res:
            out = out + x[:, self.cf*4:self.cf*4+4]

        if self.norm:
            out = data_inv_normalize(out, lb, ub)
            
        return out

class ResUnet(nn.Module):
    def __init__(self, args=None):
        super().__init__()
        self.args = args
        self.nframes = args['nframes']
        self.res = args['res']
        self.norm = args['norm'] if 'norm' in args else False
        nframes = self.args['nframes'] if 'nframes' in args else 1
        self.cf = 0
        nf = args['nf']
        in_nc = args['in_nc']
        out_nc = args['out_nc']

        self.conv_in = nn.Conv2d(in_nc*nframes, nf, kernel_size=3, stride=1, padding=1)

        self.conv1 = ResidualBlock(nf, nf, is_activate=False)
        self.pool1 = conv3x3(nf, nf*2)
        
        self.conv2 = ResidualBlock(nf*2, nf*2, is_activate=False)
        self.pool2 = conv3x3(nf*2, nf*4)
        
        self.conv3 = ResidualBlock(nf*4, nf*4, is_activate=False)
        self.pool3 = conv3x3(nf*4, nf*8)
        
        self.conv4 = ResidualBlock(nf*8, nf*8, is_activate=False)
        self.pool4 = conv3x3(nf*8, nf*16)
        
        self.conv5 = ResidualBlock(nf*16, nf*16, is_activate=False)
        
        self.upv6 = nn.ConvTranspose2d(nf*16, nf*8, 2, stride=2)
        self.conv6 = ResidualBlock(nf*16, nf*8, is_activate=False)
        
        self.upv7 = nn.ConvTranspose2d(nf*8, nf*4, 2, stride=2)
        self.conv7 = ResidualBlock(nf*8, nf*4, is_activate=False)
        
        self.upv8 = nn.ConvTranspose2d(nf*4, nf*2, 2, stride=2)
        self.conv8 = ResidualBlock(nf*4, nf*2, is_activate=False)
        
        self.upv9 = nn.ConvTranspose2d(nf*2, nf, 2, stride=2)
        self.conv9 = ResidualBlock(nf*2, nf, is_activate=False)
        
        self.conv10 = nn.Conv2d(nf, out_nc, kernel_size=1, stride=1)
        # self.relu = nn.ReLU(inplace=True)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x, noise_map=None):
        # shape= x.size()
        # x = x.view(-1,shape[-3],shape[-2],shape[-1])
        if self.norm:
            x, lb, ub = data_normalize(x)
        conv_in = self.lrelu(self.conv_in(x))
        
        conv1 = self.conv1(conv_in)
        pool1 = self.pool1(conv1)
        
        conv2 = self.conv2(pool1)
        pool2 = self.pool2(conv2)
        
        conv3 = self.conv3(pool2)
        pool3 = self.pool3(conv3)
        
        conv4 = self.conv4(pool3)
        pool4 = self.pool4(conv4)
        
        conv5 = self.conv5(pool4)
        
        up6 = self.upv6(conv5)
        up6 = torch.cat([up6, conv4], 1)
        conv6 = self.conv6(up6)
        
        up7 = self.upv7(conv6)
        up7 = torch.cat([up7, conv3], 1)
        conv7 = self.conv7(up7)
        
        up8 = self.upv8(conv7)
        up8 = torch.cat([up8, conv2], 1)
        conv8 = self.conv8(up8)
        
        up9 = self.upv9(conv8)
        up9 = torch.cat([up9, conv1], 1)
        conv9 = self.conv9(up9)
        
        out = self.conv10(conv9)
        if self.res:
            out = out + x[:, self.cf*4:self.cf*4+4]

        if self.norm:
            out = data_inv_normalize(out, lb, ub)

        return out
    
class ResUnet2(nn.Module):
    def __init__(self, args=None):
        super().__init__()
        self.args = args
        self.nframes = args['nframes']
        self.res = args['res']
        self.norm = args['norm'] if 'norm' in args else False
        nframes = self.args['nframes'] if 'nframes' in args else 1
        self.cf = 0
        nf = args['nf']
        in_nc = args['in_nc']
        out_nc = args['out_nc']

        self.conv_in = nn.Conv2d(in_nc*nframes, nf, kernel_size=3, stride=1, padding=1)

        self.conv1 = ResBlock(nf, nf, is_activate=False)
        self.pool1 = conv3x3(nf, nf*2)
        
        self.conv2 = ResBlock(nf*2, nf*2, is_activate=False)
        self.pool2 = conv3x3(nf*2, nf*4)
        
        self.conv3 = ResBlock(nf*4, nf*4, is_activate=False)
        self.pool3 = conv3x3(nf*4, nf*8)
        
        self.conv4 = ResBlock(nf*8, nf*8, is_activate=False)
        self.pool4 = conv3x3(nf*8, nf*16)
        
        self.conv5 = ResBlock(nf*16, nf*16, is_activate=False)
        
        self.upv6 = nn.ConvTranspose2d(nf*16, nf*8, 2, stride=2)
        self.conv6 = ResBlock(nf*16, nf*8, is_activate=False)
        
        self.upv7 = nn.ConvTranspose2d(nf*8, nf*4, 2, stride=2)
        self.conv7 = ResBlock(nf*8, nf*4, is_activate=False)
        
        self.upv8 = nn.ConvTranspose2d(nf*4, nf*2, 2, stride=2)
        self.conv8 = ResBlock(nf*4, nf*2, is_activate=False)
        
        self.upv9 = nn.ConvTranspose2d(nf*2, nf, 2, stride=2)
        self.conv9 = ResBlock(nf*2, nf, is_activate=False)
        
        self.conv10 = nn.Conv2d(nf, out_nc, kernel_size=1, stride=1)
        # self.relu = nn.ReLU(inplace=True)
        self.lrelu = nn.LeakyReLU(0.2)

    def forward(self, x, noise_map=None):
        # shape= x.size()
        # x = x.view(-1,shape[-3],shape[-2],shape[-1])
        if self.norm:
            x, lb, ub = data_normalize(x)
        conv_in = self.lrelu(self.conv_in(x))
        
        conv1 = self.conv1(conv_in)
        pool1 = self.pool1(conv1)
        
        conv2 = self.conv2(pool1)
        pool2 = self.pool2(conv2)
        
        conv3 = self.conv3(pool2)
        pool3 = self.pool3(conv3)
        
        conv4 = self.conv4(pool3)
        pool4 = self.pool4(conv4)
        
        conv5 = self.conv5(pool4)
        
        up6 = self.upv6(conv5)
        up6 = torch.cat([up6, conv4], 1)
        conv6 = self.conv6(up6)
        
        up7 = self.upv7(conv6)
        up7 = torch.cat([up7, conv3], 1)
        conv7 = self.conv7(up7)
        
        up8 = self.upv8(conv7)
        up8 = torch.cat([up8, conv2], 1)
        conv8 = self.conv8(up8)
        
        up9 = self.upv9(conv8)
        up9 = torch.cat([up9, conv1], 1)
        conv9 = self.conv9(up9)
        
        out = self.conv10(conv9)
        if self.res:
            out = out + x[:, self.cf*4:self.cf*4+4]

        if self.norm:
            out = data_inv_normalize(out, lb, ub)

        return out

class SNRnet(nn.Module):
    def __init__(self, args=None):
        super().__init__()
        self.args = args
        self.nframes = args['nframes']
        self.cf = 0
        self.res = args['res']
        self.norm = args['norm'] if 'norm' in args else False
        nframes = self.args['nframes'] if 'nframes' in args else 1
        nf = args['nf']
        in_nc = args['in_nc']
        out_nc = args['out_nc']

        self.conv_in = nn.Conv2d(in_nc*nframes, nf, kernel_size=3, stride=1, padding=1)

        self.conv1 = SNR_Block(nf, nf, is_activate=False)
        self.pool1 = conv3x3(nf, nf*2)
        
        self.conv2 = SNR_Block(nf*2, nf*2, is_activate=False)
        self.pool2 = conv3x3(nf*2, nf*4)
        
        self.conv3 = SNR_Block(nf*4, nf*4, is_activate=False)
        self.pool3 = conv3x3(nf*4, nf*8)
        
        self.conv4 = SNR_Block(nf*8, nf*8, is_activate=False)
        self.pool4 = conv3x3(nf*8, nf*16)
        
        self.conv5 = SNR_Block(nf*16, nf*16, is_activate=False)
        
        self.upv6 = nn.ConvTranspose2d(nf*16, nf*8, 2, stride=2)
        self.conv6 = SNR_Block(nf*16, nf*8, is_activate=False)
        
        self.upv7 = nn.ConvTranspose2d(nf*8, nf*4, 2, stride=2)
        self.conv7 = SNR_Block(nf*8, nf*4, is_activate=False)
        
        self.upv8 = nn.ConvTranspose2d(nf*4, nf*2, 2, stride=2)
        self.conv8 = SNR_Block(nf*4, nf*2, is_activate=False)
        
        self.upv9 = nn.ConvTranspose2d(nf*2, nf, 2, stride=2)
        self.conv9 = SNR_Block(nf*2, nf, is_activate=False)
        
        self.conv10 = nn.Conv2d(nf, out_nc, kernel_size=1, stride=1)
        self.lrelu = nn.LeakyReLU(inplace=True)

    def forward(self, x, t):
        # shape= x.size()
        # x = x.view(-1,shape[-3],shape[-2],shape[-1])
        if self.norm:
            x, lb, ub = data_normalize(x)
            t = t / (ub-lb)

        conv_in = self.lrelu(self.conv_in(x))
        
        conv1 = self.conv1(conv_in, t)
        pool1 = self.pool1(conv1)
        
        conv2 = self.conv2(pool1, t)
        pool2 = self.pool2(conv2)
        
        conv3 = self.conv3(pool2, t)
        pool3 = self.pool3(conv3)
        
        conv4 = self.conv4(pool3, t)
        pool4 = self.pool4(conv4)
        
        conv5 = self.conv5(pool4, t)
        
        up6 = self.upv6(conv5)
        up6 = torch.cat([up6, conv4], 1)
        conv6 = self.conv6(up6, t)
        
        up7 = self.upv7(conv6)
        up7 = torch.cat([up7, conv3], 1)
        conv7 = self.conv7(up7, t)
        
        up8 = self.upv8(conv7)
        up8 = torch.cat([up8, conv2], 1)
        conv8 = self.conv8(up8, t)
        
        up9 = self.upv9(conv8)
        up9 = torch.cat([up9, conv1], 1)
        conv9 = self.conv9(up9, t)
        
        out = self.conv10(conv9)
        if self.res:
            out = out + x[:, self.cf*4:self.cf*4+4]

        if self.norm:
            out = data_inv_normalize(out, lb, ub)

        return out

class GuidedResUnet(nn.Module):
    def __init__(self, args=None):
        super().__init__()
        self.args = args
        self.nframes = args['nframes']
        self.cf = 0
        self.res = args['res']
        self.norm = args['norm'] if 'norm' in args else False
        nframes = self.args['nframes'] if 'nframes' in args else 1
        nf = args['nf']
        in_nc = args['in_nc']
        out_nc = args['out_nc']

        self.conv_in = nn.Conv2d(in_nc*nframes, nf, kernel_size=3, stride=1, padding=1)

        self.conv1 = GuidedResidualBlock(nf, nf, is_activate=False)
        self.pool1 = conv3x3(nf, nf*2)
        
        self.conv2 = GuidedResidualBlock(nf*2, nf*2, is_activate=False)
        self.pool2 = conv3x3(nf*2, nf*4)
        
        self.conv3 = GuidedResidualBlock(nf*4, nf*4, is_activate=False)
        self.pool3 = conv3x3(nf*4, nf*8)
        
        self.conv4 = GuidedResidualBlock(nf*8, nf*8, is_activate=False)
        self.pool4 = conv3x3(nf*8, nf*16)
        
        self.conv5 = GuidedResidualBlock(nf*16, nf*16, is_activate=False)
        
        self.upv6 = nn.ConvTranspose2d(nf*16, nf*8, 2, stride=2)
        self.conv6 = GuidedResidualBlock(nf*16, nf*8, is_activate=False)
        
        self.upv7 = nn.ConvTranspose2d(nf*8, nf*4, 2, stride=2)
        self.conv7 = GuidedResidualBlock(nf*8, nf*4, is_activate=False)
        
        self.upv8 = nn.ConvTranspose2d(nf*4, nf*2, 2, stride=2)
        self.conv8 = GuidedResidualBlock(nf*4, nf*2, is_activate=False)
        
        self.upv9 = nn.ConvTranspose2d(nf*2, nf, 2, stride=2)
        self.conv9 = GuidedResidualBlock(nf*2, nf, is_activate=False)
        
        self.conv10 = nn.Conv2d(nf, out_nc, kernel_size=1, stride=1)
        self.lrelu = nn.LeakyReLU(inplace=True)

    def forward(self, x, t):
        # shape= x.size()
        # x = x.view(-1,shape[-3],shape[-2],shape[-1])
        if self.norm:
            x, lb, ub = data_normalize(x)
            t = t / (ub-lb)

        conv_in = self.lrelu(self.conv_in(x))
        
        conv1 = self.conv1(conv_in, t)
        pool1 = self.pool1(conv1)
        
        conv2 = self.conv2(pool1, t)
        pool2 = self.pool2(conv2)
        
        conv3 = self.conv3(pool2, t)
        pool3 = self.pool3(conv3)
        
        conv4 = self.conv4(pool3, t)
        pool4 = self.pool4(conv4)
        
        conv5 = self.conv5(pool4, t)
        
        up6 = self.upv6(conv5)
        up6 = torch.cat([up6, conv4], 1)
        conv6 = self.conv6(up6, t)
        
        up7 = self.upv7(conv6)
        up7 = torch.cat([up7, conv3], 1)
        conv7 = self.conv7(up7, t)
        
        up8 = self.upv8(conv7)
        up8 = torch.cat([up8, conv2], 1)
        conv8 = self.conv8(up8, t)
        
        up9 = self.upv9(conv8)
        up9 = torch.cat([up9, conv1], 1)
        conv9 = self.conv9(up9, t)
        
        out = self.conv10(conv9)
        if self.res:
            out = out + x[:, self.cf*4:self.cf*4+4]

        if self.norm:
            out = data_inv_normalize(out, lb, ub)

        return out

from .comp import DownConv, UpConv

class EstUnet(nn.Module):
    """ `UNet` class is based on https://arxiv.org/abs/1505.04597
    The U-Net is a convolutional encoder-decoder neural network.
    Contextual spatial information (from the decoding,
    expansive pathway) about an input tensor is merged with
    information representing the localization of details
    (from the encoding, compressive pathway).
    Modifications to the original paper:
    (1) padding is used in 3x3 convolutions to prevent loss
        of border pixels
    (2) merging outputs does not require cropping due to (1)
    (3) residual connections can be used by specifying
        UNet(merge_mode='add')
    (4) if non-parametric upsampling is used in the decoder
        pathway (specified by upmode='upsample'), then an
        additional 1x1 2d convolution occurs after upsampling
        to reduce channel dimensionality by a factor of 2.
        This channel halving happens with the convolution in
        the tranpose convolution (specified by upmode='transpose')
    """
    def default_args(self):
        self.args = {}
        self.args['out_nc'] = 4
        self.args['in_nc'] = 4
        self.args['depth'] = 3
        self.args['nf'] = 64
        self.args['nframes'] = 1
        self.args['res'] = False
        self.args['up_mode'] = 'transpose' # transpose, upsammple
        self.args['merge_mode'] = 'add' # add/concat
        self.args['use_type'] = 'std' # std/var
        self.args['pge'] = True

    def __init__(self, args):
        """
        Arguments:
            in_channels: int, number of channels in the input tensor.
                Default is 3 for RGB images.
            depth: int, number of MaxPools in the U-Net.
            start_filts: int, number of convolutional filters for the
                first conv.
            up_mode: string, type of upconvolution. Choices: 'transpose'
                for transpose convolution or 'upsample' for nearest neighbour
                upsampling.
        """
        super().__init__()
        self.default_args()
        if args is not None:
            for key in args:
                self.args[key] = args[key]
        
        if self.args['up_mode'] in ('transpose', 'upsample'):
            self.up_mode = self.args['up_mode']
        else:
            raise ValueError("\"{}\" is not a valid mode for "
                             "upsampling. Only \"transpose\" and "
                             "\"upsample\" are allowed.".format(self.args['up_mode']))

        if self.args['merge_mode'] in ('concat', 'add'):
            self.merge_mode = self.args['merge_mode']
        else:
            raise ValueError("\"{}\" is not a valid mode for"
                             "merging up and down paths. "
                             "Only \"concat\" and "
                             "\"add\" are allowed.".format(self.args['merge_mode']))

        # NOTE: up_mode 'upsample' is incompatible with merge_mode 'add'
        if self.up_mode == 'upsample' and self.merge_mode == 'add':
            raise ValueError("up_mode \"upsample\" is incompatible "
                             "with merge_mode \"add\" at the moment "
                             "because it doesn't make sense to use "
                             "nearest neighbour to reduce "
                             "depth channels (by half).")

        self.out_nc = self.args['out_nc']
        self.in_nc = self.args['in_nc'] * self.args['nframes']
        self.depth = self.args['depth']
        self.start_filts = self.args['nf']

        self.down_convs = []
        self.up_convs = []
        # it seems useless?
        self.noiseSTD = nn.Parameter(data=torch.log(torch.tensor(0.5)))

        # create the encoder pathway and add to a list
        for i in range(self.depth):
            ins = self.in_nc if i == 0 else outs
            outs = self.start_filts*(2**i)
            pooling = True if i < self.depth-1 else False

            down_conv = DownConv(ins, outs, pooling=pooling)
            self.down_convs.append(down_conv)

        # create the decoder pathway and add to a list
        # - careful! decoding only requires depth-1 blocks
        for i in range(self.depth-1):
            ins = outs
            outs = ins // 2
            up_conv = UpConv(ins, outs, up_mode=self.up_mode,
                merge_mode=self.merge_mode)
            self.up_convs.append(up_conv)

        self.conv_final = conv1x1(outs, self.out_nc)
        self.sigmoid=nn.Sigmoid().cuda()
        # add the list of modules to current module
        self.down_convs = nn.ModuleList(self.down_convs)
        self.up_convs = nn.ModuleList(self.up_convs)

        self.reset_params()

    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_normal(m.weight)
            nn.init.constant(m.bias, 0)

    def reset_params(self):
        for i, m in enumerate(self.modules()):
            self.weight_init(m)

    def forward(self, inp):
        encoder_outs = []
        x = inp
        # encoder pathway, save outputs for merging
        for i, module in enumerate(self.down_convs):
            x, before_pool = module(x)
            encoder_outs.append(before_pool)

        for i, module in enumerate(self.up_convs):
            before_pool = encoder_outs[-(i+2)]
            x = module(before_pool, x)

        before_x=self.conv_final(x)
        
        x = x+inp[:,:self.args['in_nc']] if self.args['res'] else x
        x = before_x if self.args['use_type']=='std' else before_x**2
        x = torch.mean(x, dim=(2,3)).squeeze() if self.args['pge'] else x

        return x