import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import lr_scheduler

class ResBlock(nn.Module):
    """ Regular ResBlock
        #output == #input
    """
    def __init__(self, dim, kernel_size=3, stride=1, padding=1):
        super(ResBlock, self).__init__()

        layers = []
        layers += [
        	nn.ReflectionPad2d(padding),
			nn.Conv2d(dim, dim, kernel_size, stride),
			nn.InstanceNorm2d(dim),
			nn.ReLU(inplace=True) 
    	]
        layers += [
        	nn.ReflectionPad2d(padding),
			nn.Conv2d(dim, dim, kernel_size, 1),
			nn.InstanceNorm2d(dim)
    	]

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        residual = x
        
        logit = self.model(x)
        logit += residual
        return logit

class DisResDownBlocks(nn.Module):
    """ ResBlock for discriminator.
        
        1.activate unit before conv.
        2. #output != #input

    """
    def __init__(self, input_nc, output_nc, kernel_size=3, stride=1, padding=1):
        super(DisResDownBlocks, self).__init__()

        layers_residual = [
            nn.ReflectionPad2d(padding),
            nn.Conv2d(input_nc, output_nc, kernel_size, stride)
        ]
        self.model_residual = nn.Sequential(*layers_residual)

        layers = [
            nn.ReflectionPad2d(padding),
            nn.Conv2d(input_nc, output_nc, kernel_size, stride),
            nn.LeakyReLU(0.2, inplace=False)
            
        ]
        layers += [
            nn.ReflectionPad2d(padding),
            nn.Conv2d(output_nc, output_nc, kernel_size, stride),
            nn.LeakyReLU(0.2, inplace=False)
        ]

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        residual = self.model_residual(x)
        
        logit = self.model(x)
        logit += residual
        return logit

class ResAdaINBlock(nn.Module):
    def __init__(self, dim, kernel_size=3, stride=1, padding=1):
        super(ResAdaINBlock, self).__init__()

        self.padding = nn.ReflectionPad2d(padding)
        self.conv = nn.Conv2d(dim, dim, kernel_size, stride)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x, mean_std):
        residual = x
        dim = x.size(1)

        out = self.padding(x)
        out = self.conv(out)
        out = AdaIN(out, mean_std[:, 0:dim], mean_std[:, dim:dim*2])
        out = self.relu(out)

        out = self.padding(out)
        out = self.conv(out)
        out = AdaIN(out, mean_std[:, dim*2:dim*3], mean_std[:, dim*3:dim*4])

        out += residual
        return out


##################################################################################
# Utils
##################################################################################

class Upsample(nn.Module):
    """ nn.Upsample is deprecated """

    def __init__(self, scale_factor, mode="nearest"):
        super(Upsample, self).__init__()
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)
        return x

def AdaIN(feature, mean, std, eps=1e-5):
    size = feature.size()
    assert (len(size) == 4)
    N, C = size[:2]

    mean = mean.view(N,C,1,1)
    std = std.view(N,C,1,1)

    feature_var = feature.view(N, C, -1).var(dim=2) + eps
    feature_std = feature_var.sqrt().view(N, C, 1, 1)
    feature_mean = feature.view(N, C, -1).mean(dim=2).view(N, C, 1, 1) 

    normalized_feat = (feature - feature_mean.expand(size)) / feature_std.expand(size)
    adain = normalized_feat * std.expand(size) + mean.expand(size)

    return adain

def get_scheduler(optimizer, opt):
    """Return a learning rate scheduler

    Parameters:
        optimizer          -- the optimizer of the network
        opt (option dicts) -- stores all the experiment flags;
                              opt['lr_policy'] is the name of learning rate policy: linear | step | plateau | cosine

    For 'linear', we keep the same learning rate for the first <opt['nepoch']> epochs
    and linearly decay the rate to zero over the next <opt['nepoch_decay']> epochs.
    For other schedulers (step, plateau, and cosine), we use the default PyTorch schedulers.
    See https://pytorch.org/docs/stable/optim.html for more details.
    """
    if opt['lr_policy'] == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt['epoch_count'] - opt['nepoch']) / float(opt['nepoch_decay'] + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt['lr_policy'] == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt['lr_decay_iters'], gamma=0.1)
    elif opt['lr_policy'] == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt['lr_policy'] == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt['nepoch'], eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt['lr_policy'])
    return scheduler