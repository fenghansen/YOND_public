from .modules import *

class DnCNN(nn.Module):
    def __init__(self, args=None):
        super().__init__()
        self.args = args
        self.res = args['res']
        self.raw2rgb = True if args['in_nc']==4 and args['out_nc']==3 else False
        nf = args['nf']
        in_nc = args['in_nc']
        out_nc = args['out_nc']
        depth = args['depth']
        use_bn = args['use_bn']

        layers = []

        layers.append(nn.Conv2d(in_channels=in_nc, out_channels=nf, kernel_size=3, padding=1, bias=True))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(depth-2):
            layers.append(nn.Conv2d(in_channels=nf, out_channels=nf, kernel_size=3, padding=1, bias=False))
            if use_bn:
                layers.append(nn.BatchNorm2d(nf, eps=0.0001, momentum=0.95))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(in_channels=nf, out_channels=out_nc, kernel_size=3, padding=1, bias=False))
        self.dncnn = nn.Sequential(*layers)

    def forward(self, x):
        out = self.dncnn(x)
        if self.raw2rgb:
            out = nn.functional.pixel_shuffle(out, 2)
        elif self.res: 
            out = x - out #out = out + x
        return out

def conv33(in_channels, out_channels, stride=1,
            padding=1, bias=True, groups=1):
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=3,
        stride=stride,
        padding=padding,
        bias=bias,
        groups=groups)

def upconv2x2(in_channels, out_channels, mode='transpose'):
    if mode == 'transpose':
        return nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=2,
            stride=2)
    else:
        # out_channels is always going to be the same
        # as in_channels
        return nn.Sequential(
            nn.Upsample(mode='bilinear', scale_factor=2),
            conv1x1(in_channels, out_channels))

class DownConv(nn.Module):
    """
    A helper Module that performs 2 convolutions and 1 MaxPool.
    A ReLU activation follows each convolution.
    """
    def __init__(self, in_channels, out_channels, pooling=True):
        super(DownConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.pooling = pooling

        self.conv1 = conv33(self.in_channels, self.out_channels)
        self.conv2 = conv33(self.out_channels, self.out_channels)

        if self.pooling:
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        before_pool = x
        if self.pooling:
            x = self.pool(x)
        return x, before_pool


class UpConv(nn.Module):
    """
    A helper Module that performs 2 convolutions and 1 UpConvolution.
    A ReLU activation follows each convolution.
    """
    def __init__(self, in_channels, out_channels,
                 merge_mode='concat', up_mode='transpose'):
        super(UpConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.merge_mode = merge_mode
        self.up_mode = up_mode

        self.upconv = upconv2x2(self.in_channels, self.out_channels,
            mode=self.up_mode)

        if self.merge_mode == 'concat':
            self.conv1 = conv33(
                2*self.out_channels, self.out_channels)
        else:
            # num of input channels to conv2 is same
            self.conv1 = conv33(self.out_channels, self.out_channels)
        self.conv2 = conv33(self.out_channels, self.out_channels)


    def forward(self, from_down, from_up):
        """ Forward pass
        Arguments:
            from_down: tensor from the encoder pathway
            from_up: upconv'd tensor from the decoder pathway
        """
        from_up = self.upconv(from_up)
        if self.merge_mode == 'concat':
            x = torch.cat((from_up, from_down), 1)
        else:
            x = from_up + from_down
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        return x

class est_UNet(nn.Module):
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
        super(est_UNet, self).__init__()

        num_classes = args['out_nc']
        in_channels = args['in_nc']
        depth = args['depth']
        start_filts = args['nf']
        up_mode='transpose'
        merge_mode='add'
        use_type='optimize_gat'

        self.use_type=use_type
        if up_mode in ('transpose', 'upsample'):
            self.up_mode = up_mode
        else:
            raise ValueError("\"{}\" is not a valid mode for "
                             "upsampling. Only \"transpose\" and "
                             "\"upsample\" are allowed.".format(up_mode))

        if merge_mode in ('concat', 'add'):
            self.merge_mode = merge_mode
        else:
            raise ValueError("\"{}\" is not a valid mode for"
                             "merging up and down paths. "
                             "Only \"concat\" and "
                             "\"add\" are allowed.".format(up_mode))

        # NOTE: up_mode 'upsample' is incompatible with merge_mode 'add'
        if self.up_mode == 'upsample' and self.merge_mode == 'add':
            raise ValueError("up_mode \"upsample\" is incompatible "
                             "with merge_mode \"add\" at the moment "
                             "because it doesn't make sense to use "
                             "nearest neighbour to reduce "
                             "depth channels (by half).")

        self.num_classes = num_classes
        self.in_channels = in_channels
        self.start_filts = start_filts
        self.depth = depth

        self.down_convs = []
        self.up_convs = []

        self.noiseSTD = nn.Parameter(data=torch.log(torch.tensor(0.5)))



        # create the encoder pathway and add to a list
        for i in range(depth):
            ins = self.in_channels if i == 0 else outs
            outs = self.start_filts*(2**i)
            pooling = True if i < depth-1 else False

            down_conv = DownConv(ins, outs, pooling=pooling)
            self.down_convs.append(down_conv)

        # create the decoder pathway and add to a list
        # - careful! decoding only requires depth-1 blocks
        for i in range(depth-1):
            ins = outs
            outs = ins // 2
            up_conv = UpConv(ins, outs, up_mode=up_mode,
                merge_mode=merge_mode)
            self.up_convs.append(up_conv)

        self.conv_final = conv1x1(outs, self.num_classes)
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

    def forward(self, x):
        encoder_outs = []

        # encoder pathway, save outputs for merging
        for i, module in enumerate(self.down_convs):
            x, before_pool = module(x)
            encoder_outs.append(before_pool)

        for i, module in enumerate(self.up_convs):
            before_pool = encoder_outs[-(i+2)]
            x = module(before_pool, x)

        before_x=self.conv_final(x)
        if self.use_type=='optimze_gat':
            x=before_x
        else:
            x = before_x**2

        return torch.mean(x, dim=(2,3)).squeeze()

class New1(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(New1, self).__init__()
       
        self.mask = torch.from_numpy(np.array([[1,1,1],[1,0,1],[1,1,1]], dtype=np.float32)).cuda()
        self.conv1 = nn.Conv2d(in_channels=in_ch, out_channels=out_ch, padding = 1, kernel_size = 3)

    def forward(self, x):
        self.conv1.weight.data =  self.conv1.weight * self.mask
        x = self.conv1(x)
        
        return x   
    
class New2(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(New2, self).__init__()
        
        self.mask = torch.from_numpy(np.array([[0,1,0,1,0],[1,0,0,0,1],[0,0,1,0,0],[1,0,0,0,1],[0,1,0,1,0]], dtype=np.float32)).cuda()
        self.conv1 = nn.Conv2d(in_channels=in_ch, out_channels=out_ch, padding = 2, kernel_size = 5)

    def forward(self, x):
        self.conv1.weight.data =  self.conv1.weight * self.mask
        x = self.conv1(x)

        return x
    
class New3(nn.Module):
    def __init__(self, in_ch, out_ch, dilated_value):
        super(New3, self).__init__()
        
        self.mask = torch.from_numpy(np.array([[1,0,1],[0,1,0],[1,0,1]], dtype=np.float32)).cuda()
        self.conv1 = nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size = 3, padding=dilated_value, dilation=dilated_value)

    def forward(self, x):
        self.conv1.weight.data =  self.conv1.weight * self.mask
        x = self.conv1(x)

        return x
    
class Residual_module(nn.Module):
    def __init__(self, in_ch, mul = 1):
        super(Residual_module, self).__init__()
        
        self.activation1 = nn.PReLU(in_ch*mul,0).cuda()
        self.activation2 = nn.PReLU(in_ch,0).cuda()
            
        self.conv1_1by1 = nn.Conv2d(in_channels=in_ch, out_channels=in_ch*mul, kernel_size = 1)
        self.conv2_1by1 = nn.Conv2d(in_channels=in_ch*mul, out_channels=in_ch, kernel_size = 1)

    def forward(self, input):

        output_residual = self.conv1_1by1(input)
        output_residual = self.activation1(output_residual)
        output_residual = self.conv2_1by1(output_residual)
        
        output = (input + output_residual) / 2.
        output = self.activation2(output)
        
        return output
    
class Gaussian(nn.Module):
    def forward(self,input):
        return torch.exp(-torch.mul(input,input))
    

class Receptive_attention(nn.Module):
    def __init__(self, in_ch, at_type = 'softmax'):
        super(Receptive_attention, self).__init__()
        
        self.activation1 = nn.ReLU().cuda()
        self.activation2 = nn.ReLU().cuda()
        self.activation3 = nn.PReLU(in_ch,0).cuda()
            
        self.conv1_1by1 = nn.Conv2d(in_channels=in_ch, out_channels=in_ch*4, kernel_size = 1)
        self.conv2_1by1 = nn.Conv2d(in_channels=in_ch*4, out_channels=in_ch*4, kernel_size = 1)
        self.conv3_1by1 = nn.Conv2d(in_channels=in_ch*4, out_channels=9, kernel_size = 1)
        self.at_type = at_type
        if at_type == 'softmax':
            self.softmax = nn.Softmax()
        else:
            self.gaussian = Gaussian()
            self.sigmoid = nn.Sigmoid()
            

    def forward(self, input, receptive):

        if self.at_type == 'softmax':
            output_residual = self.conv1_1by1(input)
            output_residual = self.activation1(output_residual)
            output_residual = self.conv2_1by1(output_residual)
            output_residual = self.activation2(output_residual)
            output_residual = self.conv3_1by1(output_residual)
            output_residual = F.adaptive_avg_pool2d(output_residual, (1, 1))
    #         output_residual = self.Gaussian(output_residual)
            output_residual = self.softmax(output_residual).permute((1,0,2,3)).unsqueeze(-1)
        else:
            
            output_residual = self.conv1_1by1(input)
            output_residual = self.activation1(output_residual)
            output_residual = self.conv2_1by1(output_residual)
            output_residual = self.activation2(output_residual)
            output_residual = self.conv3_1by1(output_residual)
            output_residual = F.adaptive_avg_pool2d(output_residual, (1, 1))
            output_residual = self.gaussian(output_residual)
            output_residual = self.sigmoid(output_residual).permute((1,0,2,3)).unsqueeze(-1)
        
        output = torch.sum(receptive * output_residual, dim = 0)
        output = self.activation3(output)
        
        return output
    
class New1_layer(nn.Module):
    def __init__(self, in_ch, out_ch, case = 'FBI_Net', mul = 1):
        super(New1_layer, self).__init__()
        self.case = case
        self.new1 = New1(in_ch,out_ch).cuda()
        if case == 'case1' or case == 'case2' or case == 'case7' or case == 'FBI_Net':
            self.residual_module = Residual_module(out_ch, mul)
            
        self.activation_new1 = nn.PReLU(in_ch,0).cuda()
        

    def forward(self, x):
        
        
        if self.case == 'case1' or self.case =='case2'  or self.case =='case7' or self.case == 'FBI_Net': # plain NN architecture wo residual module and residual connection
            
            output_new1 = self.new1(x)
            output_new1 = self.activation_new1(output_new1)
            output = self.residual_module(output_new1)

            return output, output_new1

        else: # final model
        
            output_new1 = self.new1(x)
            output = self.activation_new1(output_new1)

            return output, output_new1
   
class New2_layer(nn.Module):
    def __init__(self, in_ch, out_ch, case = 'FBI_Net', mul = 1):
        super(New2_layer, self).__init__()
        
        self.case = case
        
        self.new2 = New2(in_ch,out_ch).cuda()
        self.activation_new1 = nn.PReLU(in_ch,0).cuda()
        if case == 'case1' or case == 'case2' or case == 'case7' or case == 'FBI_Net':
            self.residual_module = Residual_module(out_ch, mul)
        if case == 'case1' or case == 'case3' or case == 'case6' or case == 'FBI_Net':
            self.activation_new2 = nn.PReLU(in_ch,0).cuda()
        

    def forward(self, x, output_new):
        
        if self.case == 'case1': #
            
            output_new2 = self.new2(output_new)
            output_new2 = self.activation_new1(output_new2)

            output = (output_new2 + x) / 2.
            output = self.activation_new2(output)
            output = self.residual_module(output)

            return output, output_new2
            

        elif self.case == 'case2' or self.case == 'case7': #
            
            output_new2 = self.new2(x)
            output_new2 = self.activation_new1(output_new2)

            output = output_new2
            output = self.residual_module(output)

            return output, output_new2
        
        elif self.case == 'case3' or self.case == 'case6': #
            
            output_new2 = self.new2(output_new)
            output_new2 = self.activation_new1(output_new2)

            output = (output_new2 + x) / 2.
            output = self.activation_new2(output)

            return output, output_new2

        elif self.case == 'case4': #
            
            output_new2 = self.new2(x)
            output_new2 = self.activation_new1(output_new2)

            output = output_new2
            
            return output, output_new2
        
        elif self.case == 'case5' : #
            
            output_new2 = self.new2(x)
            output_new2 = self.activation_new1(output_new2)

            output = output_new2
            
            return output, output_new2
        
        else:

            output_new2 = self.new2(output_new)
            output_new2 = self.activation_new1(output_new2)

            output = (output_new2 + x) / 2.
            output = self.activation_new2(output)
            output = self.residual_module(output)

            return output, output_new2
            
    
class New3_layer(nn.Module):
    def __init__(self, in_ch, out_ch, dilated_value=3, case = 'FBI_Net', mul = 1):
        super(New3_layer, self).__init__()
        
        self.case = case
        
        self.new3 = New3(in_ch,out_ch,dilated_value).cuda()
        self.activation_new1 = nn.PReLU(in_ch,0).cuda()
        if case == 'case1' or case == 'case2'  or case == 'case7' or case == 'FBI_Net':
            self.residual_module = Residual_module(out_ch, mul)
        if case == 'case1' or case == 'case3' or case == 'case6'or case == 'FBI_Net':
            self.activation_new2 = nn.PReLU(in_ch,0).cuda()
        

    def forward(self, x, output_new):
        
        if self.case == 'case1': #
            
            output_new3 = self.new3(output_new)
            output_new3 = self.activation_new1(output_new3)

            output = (output_new3 + x) / 2.
            output = self.activation_new2(output)
            output = self.residual_module(output)

            return output, output_new3
            

        elif self.case == 'case2' or self.case == 'case7': #
            
            output_new3 = self.new3(x)
            output_new3 = self.activation_new1(output_new3)

            output = output_new3
            output = self.residual_module(output)

            return output, output_new3
        
        elif self.case == 'case3' or self.case == 'case6': #
            
            output_new3 = self.new3(output_new)
            output_new3 = self.activation_new1(output_new3)

            output = (output_new3 + x) / 2.
            output = self.activation_new2(output)

            return output, output_new3

        elif self.case == 'case4': #
            
            output_new3 = self.new3(x)
            output_new3 = self.activation_new1(output_new3)

            output = output_new3
            
            return output, output_new3
        
        elif self.case == 'case5': #
            
            output_new3 = self.new3(x)
            output_new3 = self.activation_new1(output_new3)

            output = output_new3
            
            return output, output_new3
        
        else:

            output_new3 = self.new3(output_new)
            output_new3 = self.activation_new1(output_new3)

            output = (output_new3 + x) / 2.
            output = self.activation_new2(output)
            output = self.residual_module(output)

            return output, output_new3

class AttrProxy(object):
    """Translates index lookups into attribute lookups."""
    def __init__(self, module, prefix):
        self.module = module
        self.prefix = prefix

    def __getitem__(self, i):
        return getattr(self.module, self.prefix + str(i))

class FBI_Net(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        channel = args['channel']
        output_channel = args['output_channel']
        filters = args['nf']
        mul = args['mul']
        num_of_layers = args['num_of_layers']
        case = args['case']
        output_type = args['output_type']
        sigmoid_value = args['sigmoid_value']
        self.res = args['res']

        self.case = case

        self.new1 = New1_layer(channel, filters, mul = mul, case = case).cuda()
        self.new2 = New2_layer(filters, filters, mul = mul, case = case).cuda()
        
        self.num_layers = num_of_layers
        self.output_type = output_type
        self.sigmoid_value = sigmoid_value

        dilated_value = 3
        
        for layer in range (num_of_layers-2):
            self.add_module('new_' + str(layer), New3_layer(filters, filters, dilated_value, mul = mul, case = case).cuda())
            
        self.residual_module = Residual_module(filters, mul)
        self.activation = nn.PReLU(filters,0).cuda()
        self.output_layer = nn.Conv2d(in_channels=filters, out_channels=output_channel, kernel_size = 1).cuda()
        
        if self.output_type == 'sigmoid':
            self.sigmoid=nn.Sigmoid().cuda()
        
        self.new = AttrProxy(self, 'new_')

    def forward(self, x):
        
        if self.case == 'FBI_Net' or self.case == 'case2' or self.case == 'case3' or self.case == 'case4':

            output, output_new = self.new1(x)
            output_sum = output
            output, output_new = self.new2(output, output_new)
            output_sum = output + output_sum

            for i, (new_layer)  in enumerate(self.new):

                output, output_new  = new_layer(output, output_new)
                output_sum = output + output_sum

                if i == self.num_layers - 3:
                    break

            final_output = self.activation(output_sum/self.num_layers)
            final_output = self.residual_module(final_output)
            final_output = self.output_layer(final_output)
            
        else:

            output, output_new = self.new1(x)
            output, output_new = self.new2(output, output_new)

            for i, (new_layer)  in enumerate(self.new):

                output, output_new  = new_layer(output, output_new)

                if i == self.num_layers - 3:
                    break

            final_output = self.activation(output)
            final_output = self.residual_module(final_output)
            final_output = self.output_layer(final_output)
            
        if self.output_type=='sigmoid':
               final_output[:,0]=(torch.ones_like(final_output[:,0])*self.sigmoid_value)*self.sigmoid(final_output[:,0])

        if self.res:
            final_output = final_output[:,:1] * x + final_output[:,1:]

        return final_output

class SelfSupUNet(nn.Module):
    def __init__(self, args):
        """
        Args:
            in_channels (int): number of input channels, Default 4
            depth (int): depth of the network, Default 5
            nf (int): number of filters in the first layer, Default 48
        """
        super().__init__()
        in_channels = args['in_nc']
        out_channels = args['out_nc']
        depth = args['depth'] if 'depth' in args else 5
        nf = args['nf'] if 'nf' in args else 48
        slope = args['slope'] if 'slope' in args else 0.1
        self.norm = args['norm'] if 'norm' in args else False
        self.res = args['res'] if 'res' in args else False

        self.depth = depth
        self.head = nn.Sequential(
            LR(in_channels, nf, 3, slope), LR(nf, nf, 3, slope))
        self.down_path = nn.ModuleList()
        for i in range(depth):
            self.down_path.append(LR(nf, nf, 3, slope))

        self.up_path = nn.ModuleList()
        for i in range(depth):
            if i != depth-1:
                self.up_path.append(UP(nf*2 if i==0 else nf*3, nf*2, slope))
            else:
                self.up_path.append(UP(nf*2+in_channels, nf*2, slope))

        self.last = nn.Sequential(LR(2*nf, 2*nf, 1, slope), 
                    LR(2*nf, 2*nf, 1, slope), conv1x1(2*nf, out_channels, bias=True))

    def forward(self, x):
        if self.norm:
            x, lb, ub = data_normalize(x)
        blocks = []
        blocks.append(x)
        x = self.head(x)
        for i, down in enumerate(self.down_path):
            x = F.max_pool2d(x, 2)
            if i != len(self.down_path) - 1:
                blocks.append(x)
            x = down(x)

        for i, up in enumerate(self.up_path):
            x = up(x, blocks[-i-1])

        out = self.last(x)
        if self.res:
            out = out + x

        if self.norm:
            out = data_inv_normalize(out, lb, ub)

        return out


class LR(nn.Module):
    def __init__(self, in_size, out_size, ksize=3, slope=0.1):
        super(LR, self).__init__()
        block = []
        block.append(nn.Conv2d(in_size, out_size,
                     kernel_size=ksize, padding=ksize//2, bias=True))
        block.append(nn.LeakyReLU(slope, inplace=False))

        self.block = nn.Sequential(*block)

    def forward(self, x):
        out = self.block(x)
        return out


class UP(nn.Module):
    def __init__(self, in_size, out_size, slope=0.1):
        super(UP, self).__init__()
        self.conv_1 = LR(in_size, out_size)
        self.conv_2 = LR(out_size, out_size)

    def up(self, x):
        s = x.shape
        x = x.reshape(s[0], s[1], s[2], 1, s[3], 1)
        x = x.repeat(1, 1, 1, 2, 1, 2)
        x = x.reshape(s[0], s[1], s[2]*2, s[3]*2)
        return x

    def forward(self, x, pool):
        x = self.up(x)
        x = torch.cat([x, pool], 1)
        x = self.conv_1(x)
        x = self.conv_2(x)

        return x

class SelfResUNet(nn.Module):
    def __init__(self, args):
        """
        Args:
            in_channels (int): number of input channels, Default 4
            depth (int): depth of the network, Default 5
            nf (int): number of filters in the first layer, Default 32
        """
        super().__init__()
        in_channels = args['in_nc']
        out_channels = args['out_nc']
        depth = args['depth'] if 'depth' in args else 5
        nf = args['nf'] if 'nf' in args else 32
        slope = args['slope'] if 'slope' in args else 0.1
        self.norm = args['norm'] if 'norm' in args else False
        self.res = args['res'] if 'res' in args else False

        self.depth = depth
        self.head = Res(in_channels, nf, slope)
        self.down_path = nn.ModuleList()
        for i in range(depth):
            self.down_path.append(Res(nf, nf, slope, ksize=3))

        self.up_path = nn.ModuleList()
        for i in range(depth):
            if i != depth-1:
                self.up_path.append(RUP(nf*2 if i==0 else nf*3, nf*2, slope))
            else:
                self.up_path.append(RUP(nf*2+in_channels, nf*2, slope))

        self.last = Res(2*nf, 2*nf, slope, ksize=1)
        self.out = conv1x1(2*nf, out_channels, bias=True)

    def forward(self, x):
        if self.norm:
            x, lb, ub = data_normalize(x)
        inp = x
        blocks = []
        blocks.append(x)
        x = self.head(x)
        for i, down in enumerate(self.down_path):
            x = F.max_pool2d(x, 2)
            if i != len(self.down_path) - 1:
                blocks.append(x)
            x = down(x)

        for i, up in enumerate(self.up_path):
            x = up(x, blocks[-i-1])

        out = self.last(x)
        out = self.out(out)
        if self.res:
            out = out + inp

        if self.norm:
            out = data_inv_normalize(out, lb, ub)

        return out

class RUP(nn.Module):
    def __init__(self, in_size, out_size, slope=0.1, ksize=3):
        super(RUP, self).__init__()
        self.conv_1 = LR(out_size, out_size, ksize=ksize, slope=slope)
        self.conv_2 = LR(out_size, out_size, ksize=ksize, slope=slope)
        if in_size != out_size:
            self.short_cut = nn.Sequential(conv1x1(in_size, out_size))
        else:
            self.short_cut = nn.Sequential(OrderedDict([]))

    def up(self, x):
        s = x.shape
        x = x.reshape(s[0], s[1], s[2], 1, s[3], 1)
        x = x.repeat(1, 1, 1, 2, 1, 2)
        x = x.reshape(s[0], s[1], s[2]*2, s[3]*2)
        return x

    def forward(self, x, pool):
        x = self.up(x)
        x = torch.cat([x, pool], 1)
        x = self.short_cut(x)
        z = self.conv_1(x)
        z = self.conv_2(z)
        z += x
        return z

class Res(nn.Module):
    def __init__(self, in_size, out_size, slope=0.1, ksize=3):
        super().__init__()
        self.conv_1 = LR(out_size, out_size, ksize=ksize, slope=slope)
        self.conv_2 = LR(out_size, out_size, ksize=ksize, slope=slope)
        if in_size != out_size:
            self.short_cut = nn.Sequential(conv1x1(in_size, out_size))
        else:
            self.short_cut = nn.Sequential(OrderedDict([]))

    def forward(self, x):
        x = self.short_cut(x)
        z = self.conv_1(x)
        z = self.conv_2(z)
        z += x
        return z

def conv1x1(in_chn, out_chn, bias=True):
    layer = nn.Conv2d(in_chn, out_chn, kernel_size=1,
                      stride=1, padding=0, bias=bias)
    return layer

class GuidedSelfUnet(nn.Module):
    def __init__(self, args):
        """
        Args:
            in_channels (int): number of input channels, Default 4
            depth (int): depth of the network, Default 5
            nf (int): number of filters in the first layer, Default 32
        """
        super().__init__()
        in_channels = args['in_nc']
        out_channels = args['out_nc']
        depth = args['depth'] if 'depth' in args else 5
        nf = args['nf'] if 'nf' in args else 32
        slope = args['slope'] if 'slope' in args else 0.1
        self.norm = args['norm'] if 'norm' in args else False
        self.res = args['res'] if 'res' in args else False

        self.depth = depth
        self.head = GRes(in_channels, nf, slope)
        self.down_path = nn.ModuleList()
        for i in range(depth):
            self.down_path.append(GLR(nf, nf, 3, slope))

        self.up_path = nn.ModuleList()
        for i in range(depth):
            if i != depth-1:
                self.up_path.append(GUP(nf*2 if i==0 else nf*3, nf*2, slope))
            else:
                self.up_path.append(GUP(nf*2+in_channels, nf*2, slope))

        self.last = GRes(2*nf, 2*nf, slope, ksize=1)
        self.out = conv1x1(2*nf, out_channels, bias=True)

    def forward(self, x, t):
        if self.norm:
            x, lb, ub = data_normalize(x)
            t = t / (ub-lb)
        blocks = []
        blocks.append(x)
        x = self.head(x, t)
        for i, down in enumerate(self.down_path):
            x = F.max_pool2d(x, 2)
            if i != len(self.down_path) - 1:
                blocks.append(x)
            x = down(x, t)

        for i, up in enumerate(self.up_path):
            x = up(x, blocks[-i-1], t)

        out = self.last(x, t)
        out = self.out(out)

        if self.res:
            out = out + x

        if self.norm:
            out = data_inv_normalize(out, lb, ub)

        return out
    
class GLR(nn.Module):
    def __init__(self, in_size, out_size, ksize=3, slope=0.1):
        super(GLR, self).__init__()
        self.block = nn.Conv2d(in_size, out_size,
                     kernel_size=ksize, padding=ksize//2, bias=True)
        self.act = nn.LeakyReLU(slope, inplace=False)
        self.gamma = nn.Sequential(
            conv1x1(1, out_size),
            nn.SiLU(),
            conv1x1(out_size, out_size),
        )
        self.beta = nn.Sequential(
            nn.SiLU(),
            conv1x1(out_size, out_size),
        )

    def forward(self, x, t):
        z = self.block(x)
        tk = self.gamma(t)
        tb = self.beta(tk)
        z = z * tk + tb
        out = self.act(z)
        return out

class GRes(nn.Module):
    def __init__(self, in_size, out_size, slope=0.1, ksize=3):
        super(GRes, self).__init__()
        self.conv_1 = LR(out_size, out_size, ksize=ksize)
        self.conv_2 = GLR(out_size, out_size, ksize=ksize)
        if in_size != out_size:
            self.short_cut = nn.Sequential(
                conv1x1(in_size, out_size)
            )
        else:
            self.short_cut = nn.Sequential(OrderedDict([]))

    def forward(self, x, t):
        x = self.short_cut(x)
        z = self.conv_1(x)
        z = self.conv_2(z, t)
        z += x

        return z

class GUP(nn.Module):
    def __init__(self, in_size, out_size, slope=0.1):
        super(GUP, self).__init__()
        self.conv_1 = LR(out_size, out_size)
        self.conv_2 = GLR(out_size, out_size)
        if in_size != out_size:
            self.short_cut = nn.Sequential(
                conv1x1(in_size, out_size)
            )
        else:
            self.short_cut = nn.Sequential(OrderedDict([]))

    def up(self, x):
        s = x.shape
        x = x.reshape(s[0], s[1], s[2], 1, s[3], 1)
        x = x.repeat(1, 1, 1, 2, 1, 2)
        x = x.reshape(s[0], s[1], s[2]*2, s[3]*2)
        return x

    def forward(self, x, pool, t):
        x = self.up(x)
        x = torch.cat([x, pool], 1)
        x = self.short_cut(x)
        z = self.conv_1(x)
        z = self.conv_2(z, t)
        z += x

        return z
    

class N2NF_Unet(nn.Module):
    def __init__(self, args=None):
        super().__init__()
        self.args = args
        in_nc = args['in_nc']
        out_nc = args['out_nc']
        self.norm = args['norm'] if 'norm' in args else False

        # Layers: enc_conv0, enc_conv1, pool1
        self._block1 = nn.Sequential(
            nn.Conv2d(in_nc, 48, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(48, 48, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2))

        # Layers: enc_conv(i), pool(i); i=2..5
        self._block2 = nn.Sequential(
            nn.Conv2d(48, 48, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2))

        # Layers: enc_conv6, upsample5
        self._block3 = nn.Sequential(
            nn.Conv2d(48, 48, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(48, 48, 3, stride=2, padding=1, output_padding=1))
            #nn.Upsample(scale_factor=2, mode='nearest'))

        # Layers: dec_conv5a, dec_conv5b, upsample4
        self._block4 = nn.Sequential(
            nn.Conv2d(96, 96, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(96, 96, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(96, 96, 3, stride=2, padding=1, output_padding=1))
            #nn.Upsample(scale_factor=2, mode='nearest'))

        # Layers: dec_deconv(i)a, dec_deconv(i)b, upsample(i-1); i=4..2
        self._block5 = nn.Sequential(
            nn.Conv2d(144, 96, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(96, 96, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(96, 96, 3, stride=2, padding=1, output_padding=1))
            #nn.Upsample(scale_factor=2, mode='nearest'))

        # Layers: dec_conv1a, dec_conv1b, dec_conv1c,
        self._block6 = nn.Sequential(
            nn.Conv2d(96 + in_nc, 64, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, out_nc, 3, stride=1, padding=1),
            nn.LeakyReLU(0.1))

        # Initialize weights
        self._init_weights()


    def _init_weights(self):
        """Initializes weights using He et al. (2015)."""

        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data)
                m.bias.data.zero_()


    def forward(self, x):
        if self.norm:
            x, lb, ub = data_normalize(x)

        # Encoder
        pool1 = self._block1(x)
        pool2 = self._block2(pool1)
        pool3 = self._block2(pool2)
        pool4 = self._block2(pool3)
        pool5 = self._block2(pool4)

        # Decoder
        upsample5 = self._block3(pool5)
        concat5 = torch.cat((upsample5, pool4), dim=1)
        upsample4 = self._block4(concat5)
        concat4 = torch.cat((upsample4, pool3), dim=1)
        upsample3 = self._block5(concat4)
        concat3 = torch.cat((upsample3, pool2), dim=1)
        upsample2 = self._block5(concat3)
        concat2 = torch.cat((upsample2, pool1), dim=1)
        upsample1 = self._block5(concat2)
        concat1 = torch.cat((upsample1, x), dim=1)

        # Final activation
        out = self._block6(concat1)
        if self.norm:
            out = data_inv_normalize(out, lb, ub)
        return out
