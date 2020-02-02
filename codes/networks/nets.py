import torch
from torch import nn
from torch.nn import init
from networks.blocks import ResBlock, DisResDownBlocks, ResAdaINBlock, Upsample

##############################################################################
# Encoder 
##############################################################################
def define_E(input_nc, output_nc, nef, n_downsample, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Create a encoder

    Parameters:
        input_nc (int) -- the number of channels in input feature
        output_nc (int) -- the number of channels in output feature
        nef (int) -- number of filters in the first conv layer
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Returns a encoder
    The encoder has been initialized by <init_net>.
    """

    net = Encoder(input_nc, output_nc, nef, n_downsample)
    return init_net(net, init_type, init_gain, gpu_ids)

class Encoder(nn.Module):
    """ Defines an encoder with architecture:
    Conv(5,1,2)ReLU -> 5*Conv(4,2,1)ReLU -> AdaptiveAvgPool(1) -> Conv(1,1,0)
    """

    def __init__(self, input_nc, output_nc, nef, n_downsample):
        """ Construct an encoder

        Parameters:
            input_nc (int) -- the number of channels in input image
            output_nc (int) -- the number of channels in output latent code
        """

        super(Encoder, self).__init__()
        self.output_nc = output_nc

        layers = []
        layers += [    # B*3*256*256 ->B*64*256*256
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels=input_nc, out_channels=nef, kernel_size=7, stride=1),
            nn.ReLU(inplace=True)
        ]

        for n_down in range(n_downsample-2):
            layers += [    # B*64*256*256 -> B*128*128*128 -> B*256*64*64 -> B*512*32*32
                nn.ReflectionPad2d(1),
                nn.Conv2d(in_channels=nef, out_channels=nef*2, kernel_size=4, stride=2),
                nn.ReLU(inplace=True)
            ]
            nef *= 2

        for n_down in range(2):
            layers += [    # B*512*32*32 -> B*512*16*16 -> B*512*8*8
                nn.ReflectionPad2d(1),
                nn.Conv2d(in_channels=nef, out_channels=nef, kernel_size=4, stride=2),
                nn.ReLU(inplace=True)
            ]

        layers += [
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=nef, out_channels=output_nc, kernel_size=1, stride=1)
        ]

        self.model = nn.Sequential(*layers)
        self.layers = layers

    def forward(self, x):
        out = self.model(x)
        out = out.view(-1, self.output_nc)
        return out

##############################################################################
# Generator 
##############################################################################
def define_G(input_nc, latent_nc, ngf, ng_downsample, ng_upsample, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Create a generator

    Parameters:
        input_nc (int) -- the number of channels in input image
        output_nc (int) -- the number of channels in output image
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Returns a generator
    The generator has been initialized by <init_net>.
    """
    net = Generator(input_nc, latent_nc, ngf, ng_downsample, ng_upsample)
    return init_net(net, init_type, init_gain, gpu_ids)

class Generator(nn.Module):
    """ Architecture:
    AdaIN:
    Linear(512,2048)ReLU -> Linear(2048,4096)ReLU

    G:
                                                adain
                                                --||
    [conv(7,1,3)ReLU] -> 3*[conv(4,2,1)ReLU] -> 4*[Res] -> 3*UpConv(5,1,2)ReLU -> conv(7,1,3)ReLU
    
    ----------------
    1. ReLU
    2. IN(+2 AdaIN)
    3. Symmetry architecture
    """
    def __init__(self, input_nc, latent_nc, ngf, ng_downsample, ng_upsample):
        """Construct a Resnet-based generator
        
        Parameters:
            input_nc (int)      -- the number of channels in input image
            output_nc (int)     -- the number of channels in output image
        """
        super(Generator, self).__init__()

        layers_beforeAdaIN = []

        layers_beforeAdaIN += [# B*3*256*256 ->B*64*256*256
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels=input_nc, out_channels=ngf, kernel_size=7, stride=1),
            nn.InstanceNorm2d(ngf),
            nn.ReLU(inplace=True)            
        ]

        for n_down in range(ng_downsample-1):
            layers_beforeAdaIN += [    # B*64*256*256 -> B*128*128*128 -> B*256*64*64 -> B*512*32*32
                nn.ReflectionPad2d(1),
                nn.Conv2d(in_channels=ngf, out_channels=ngf*2, kernel_size=4, stride=2),
                nn.InstanceNorm2d(ngf*2),
                nn.ReLU(inplace=True)  
            ]
            ngf *= 2

        layers_beforeAdaIN += [   
            # 2*ResBlock
            ResBlock(ngf, kernel_size=3, stride=1, padding=1),
            ResBlock(ngf, kernel_size=3, stride=1, padding=1)
        ]

        self.model_beforeAdaIN = nn.Sequential(*layers_beforeAdaIN)

        self.adain_dim = ngf
        layers_MLP = [ # B*512 -> B*2048 -> B*4096
            nn.Linear(latent_nc, ngf*4),
            nn.ReLU(inplace=True),
            nn.Linear(ngf*4, ngf*8),
            nn.ReLU(inplace=True)
        ]

        self.model_MLP = nn.Sequential(*layers_MLP)

        self.ResAdaINBlock = ResAdaINBlock(ngf, kernel_size=3, stride=1, padding=1)

        layers_afterAdaIN = []

        for n_up in range(ng_upsample): # B*512*32*32 -> B*256*64*64 -> B*128*128*128 -> B*64*256*256
            layers_afterAdaIN += [
                Upsample(scale_factor=2),
                nn.ReflectionPad2d(2),
                nn.Conv2d(in_channels=ngf, out_channels=ngf//2, kernel_size=5, stride=1),
                nn.InstanceNorm2d(ngf//2),
                nn.ReLU(inplace=True) 
            ]
            ngf //= 2
        

        layers_afterAdaIN += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels=ngf, out_channels=input_nc, kernel_size=7, stride=1),
            nn.Tanh()
        ]

        self.model_afterAdaIN = nn.Sequential(*layers_afterAdaIN)
        

    def forward(self, x, code):
        out_beforeAdaIN = self.model_beforeAdaIN(x)
        out_MLP = self.model_MLP(code)
        out_ResAdaINBlock1 = self.ResAdaINBlock(out_beforeAdaIN, out_MLP[:, 0: self.adain_dim*4])
        out_ResAdaINBlock2 = self.ResAdaINBlock(out_ResAdaINBlock1, out_MLP[:, self.adain_dim*4: self.adain_dim*8]) # Size([8, 512, 16, 16])
        out = self.model_afterAdaIN(out_ResAdaINBlock2)
        return out

##############################################################################
#  Discriminator
##############################################################################
def define_D(input_nc, ndf, num_cls, n_downsample, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Create a discriminator

    Parameters:
        input_nc (int) -- the number of channels in input images
        ndf (int)          -- the number of filters in the first conv layer
        init_type (str)    -- the name of the initialization method.
        init_gain (float)  -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Returns a discriminator
    The discriminator has been initialized by <init_net>. It uses Leakly ReLU for non-linearity.
    """
    net = MultiTaskDiscriminator(input_nc, ndf, num_cls, n_downsample)
    return init_net(net, init_type, init_gain, gpu_ids)

class MultiTaskDiscriminator(nn.Module):
    """Defines a multi-task discriminator.

        1. spectral norm
        2. LeakyRuLU
        3. Activation unit first in ResBlock
    """

    def __init__(self, input_nc, ndf, num_cls, n_downsample):
        """Construct a multi-task discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            num_cls (int)   -- the number of classes of the source data
            n_res (int)  -- the number of n_res layers in the discriminator
        """
        super(MultiTaskDiscriminator, self).__init__()
        self.num_cls = num_cls
        layers_f = [  # B*3*256*256 -> B*64*256*256
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels=input_nc, out_channels=ndf, kernel_size=7, stride=1),
            nn.LeakyReLU(0.2, inplace=True)
        ]        

        for _ in range(n_downsample):  # B*64*256*256 -> B*128*128*128 -> B*256*64*64 -> B*512*32*32
            layers_f += [
                # DisResDownBlocks(ndf, ndf, kernel_size=3, stride=1, padding=1),
                DisResDownBlocks(ndf, ndf*2, kernel_size=3, stride=1, padding=1),
                nn.AvgPool2d(kernel_size=2, stride=2)
            ]
            ndf *= 2

        self.model_f = nn.Sequential(*layers_f)
        
        layers_c = [  # B*512*32*32 -> B*C*32*32
            nn.Conv2d(ndf, num_cls, kernel_size=1, stride=1),
            nn.LeakyReLU(0.2, inplace=True)
        ]
        self.model_c = nn.Sequential(*layers_c)

        layers_finetune = [ # B*512*32*32 -> B*1*32*32
            nn.Conv2d(ndf, 1, kernel_size=1, stride=1),
            nn.LeakyReLU(0.2, inplace=True)
        ]
        self.model_finetune = nn.Sequential(*layers_finetune)

    def feature_matching(self, x):
        return self.model_f(x)
        
    def forward(self, x, y):
        out = self.model_f(x)
        out = self.model_c(out) # B*C*16*16
        if y[0] != -1:   # With label, return the C channel prediction  
            output = torch.empty(out[:,1,:,:].size(),device=out.device).unsqueeze_(1)
            for i in range(x.size()[0]):
                output[i] = out[i, y[i],:,:]
            return output
            # return out[:, y, :, :] 
        else:   # No label, return the all prediction (only used in finetune)
            return out


###############################################################################
# Helper Functions
###############################################################################
def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Initialize a network: 
        1. register CPU/GPU device (with multi-GPU support); 
        2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Return an initialized network.
    """
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
    init_weights(net, init_type, init_gain=init_gain)
    return net

def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>
