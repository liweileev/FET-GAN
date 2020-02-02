import torch
import torch.nn as nn
from .base_model import BaseModel
from networks import nets, loss

class FET_model(BaseModel):
    """
    This class implements the FET model, for learning cross-domain font effect transfer.
    """
    
    def __init__(self, opt):
        """Initialize the FET-model class.

        Parameters:
            opt (Option dicts)-- stores all the experiment flags;
        """
        
        BaseModel.__init__(self, opt)

        self.opt = opt
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['GE', 'D', 'G_GAN', 'D_GAN', 'recons_L1', 'transfer_L1', 'code_L1', 'GP']
        
        if self.opt['isTrain']:
            # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
            self.visual_names = ['source', 'refs', 'target', 'reconstruction', 'transfered']
            # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>.
            self.model_names = ['G', 'E', 'D']            
        else:
            self.visual_names = ['source', 'refs', 'transfered']
            self.model_names = ['G', 'E']
        

        # define networks (Generator, Encoder and discriminator)        
        self.netG = nets.define_G(opt['input_nc'], opt['latent_nc'], opt['ngf'], opt['ng_downsample'], opt['ng_upsample'], opt['init_type'], opt['init_gain'], self.gpu_ids)    
        # define encoder
        self.netE = nets.define_E(opt['input_nc'], opt['latent_nc'], opt['nef'], opt['ne_downsample'], opt['init_type'], opt['init_gain'], self.gpu_ids)   
        if self.isTrain:  
            # define discriminator
            self.netD = nets.define_D(opt['input_nc'], opt['ndf'], opt['num_cls'], opt['nd_downsample'], opt['init_type'], opt['init_gain'], self.gpu_ids)
            
            # define loss functions
            if self.opt['GAN_type'] == 'vanilla':
                self.criterionGAN = loss.GANLoss().to(self.device)
            elif self.opt['GAN_type'] == 'hinge':
                self.criterionGAN = loss.GANLoss_hinge().to(self.device)
            else:
                print("Invalid GAN loss type.")

            if self.opt['finetune']:
                self.criterionGAN = loss.GANLoss_hinge_finetune().to(self.device)

            self.criterionRecons = nn.L1Loss()
            self.criterionTransfer = nn.L1Loss()
            self.criterionCode = nn.L1Loss()
            self.criterionGP = loss.GPLoss().to(self.device)
            
            # initialize optimizers;
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt['lr'], betas=(opt['beta1'], 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt['lr'], betas=(opt['beta1'], 0.999))
            self.optimizer_E = torch.optim.Adam(self.netE.parameters(), lr=opt['lr'], betas=(opt['beta1'], 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)
            self.optimizers.append(self.optimizer_E)
            

    def set_input(self, input):
        """Unpack input data from the dataloader

        Parameters:
            input (dict): include the datasour
        """        
        if self.opt['isTrain']:
            self.target = input['target'].to(self.device)   # B*3*256*256
            if self.opt['finetune'] and hasattr(self, 'source_label'): 
                self.source_label = self.source_label
                self.refs_label = self.refs_label
            else:
                self.source_label = input['source_label'].type(torch.long).to(self.device)   # B
                self.refs_label = input['refs_label'].type(torch.long).to(self.device)   # B

        self.source = input['source'].to(self.device)   # B*3*256*256
        self.refs = input['refs'].to(self.device)   # B*K*3*256*256
        self.refs_flatten = self.refs.view(-1, self.opt['input_nc'], self.opt['crop_size'], self.opt['crop_size'])
        
    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>.""" 
        refs_code_flatten = self.netE(self.refs_flatten)
        self.refs_code = refs_code_flatten.view(-1, self.opt['K'], self.opt['latent_nc'])
        mean_code = torch.mean(self.refs_code, dim=1)

        self.transfered = self.netG(self.source, mean_code)

        source_code = self.netE(self.source)
        self.reconstruction = self.netG(self.source, source_code)

    def backward_GE(self):
        """Calculate reconstruction loss and code loss for the generator and encoder, GAN loss for the generator"""

        # L1 reconsitution loss
        self.loss_recons_L1 = self.criterionRecons(self.reconstruction, self.source) * self.opt['lambda_recons']

        # L1 transfer loss
        self.loss_transfer_L1 = self.criterionTransfer(self.transfered, self.target) * self.opt['lambda_transfer']

        # L1 code loss
        self.loss_code_L1 = 0
        for i in range(self.opt['K']):
            for j in range(i, self.opt['K']):
                self.loss_code_L1 += self.criterionCode(self.refs_code[:,i,:], self.refs_code[:,j,:])
        self.loss_code_L1 = self.loss_code_L1 * self.opt['lambda_code'] 

        # fool the discriminator
        pred_fake_refs = self.netD(self.transfered.detach(), self.refs_label)
        pred_fake_source = self.netD(self.reconstruction.detach(), self.source_label)
        self.loss_G_GAN = (self.criterionGAN(pred_fake_refs, True) + self.criterionGAN(pred_fake_source, True)) * self.opt['lambda_GAN']

        # combine loss and calculate gradients
        self.loss_GE = self.loss_recons_L1 + self.loss_transfer_L1 + self.loss_code_L1 + self.loss_G_GAN
        self.loss_GE.backward()
        
    def backward_D(self):
        """Calculate GAN loss for the discriminator"""

        # real to discriminator        
        self.refs_flatten.requires_grad_()
        self.source.requires_grad_()
        if self.opt['finetune'] and self.source_label[0] == -1:
            # find the nearest class
            pred_real_source = self.netD(self.source, self.source_label)
            nearest_label, _ = self.criterionGAN(pred_real_source, True)
            print('nearest_label is :'+str(nearest_label.cpu()))
            self.source_label = nearest_label * torch.ones_like(self.source_label)
            self.refs_label = nearest_label * torch.ones_like(self.refs_label)
            refs_label_flatten = torch.transpose(self.refs_label.repeat(self.opt['K']).reshape(self.opt['K'], self.refs_label.size()[0]), 0,1).contiguous().view(-1)
            # calculate the GAN Loss with nearest_label
            pred_real_refs = self.netD(self.refs_flatten, refs_label_flatten)
            pred_real_source = self.netD(self.source, self.source_label)
            self.loss_D_real = self.criterionGAN(pred_real_refs, True) + self.criterionGAN(pred_real_source, True)
        else:
            refs_label_flatten = torch.transpose(self.refs_label.repeat(self.opt['K']).reshape(self.opt['K'], self.refs_label.size()[0]), 0,1).contiguous().view(-1)
            pred_real_refs = self.netD(self.refs_flatten, refs_label_flatten)
            pred_real_source = self.netD(self.source, self.source_label)
            self.loss_D_real = self.criterionGAN(pred_real_refs, True) + self.criterionGAN(pred_real_source, True)

        # gradient penalty to the real
        self.loss_GP = (self.criterionGP(pred_real_refs, self.refs_flatten) + self.criterionGP(pred_real_source, self.source)) * self.opt['lambda_gp']

        # fake to discriminnator
        pred_fake_refs = self.netD(self.transfered.detach(), self.refs_label)
        pred_fake_source = self.netD(self.reconstruction.detach(), self.source_label)
        self.loss_D_fake = self.criterionGAN(pred_fake_refs, False) + self.criterionGAN(pred_fake_source, False)

        self.loss_D_GAN = (self.loss_D_real + self.loss_D_fake) * 0.5

        self.loss_D = (self.loss_D_GAN + self.loss_GP) * self.opt['lambda_GAN']
        self.loss_D.backward()
     
    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # compute fake images, reconstruction code and reconstruction images.
        self.forward()    
        # update D
        self.set_requires_grad(self.netD, True)  # enable backprop for D
        self.optimizer_D.zero_grad()     # set D's gradients to zero
        self.backward_D()                # calculate gradients for D
        self.optimizer_D.step()          # update D's weights

        # update G and E
        self.set_requires_grad(self.netD, False)  # D requires no gradients when optimizing G and E
        self.optimizer_G.zero_grad()        # set G's gradients to zero
        self.optimizer_E.zero_grad()        # set E's gradients to zero
        self.backward_GE()                   # calculate graidents for G and E
        self.optimizer_G.step()             # udpate G's weights        
        self.optimizer_E.step()             # udpate E's weights        