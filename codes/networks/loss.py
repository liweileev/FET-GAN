import torch
import torch.nn as nn

##############################################################################
# Loss
##############################################################################
class GANLoss(nn.Module):
    """Define GAN objectives.
    The GANLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.
    """
    def __init__(self, target_real_label=1.0, target_fake_label=0.0):
        """ Initialize the GANLoss class.

        Parameters:
            target_real_label (bool) - - label for a real image
            target_fake_label (bool) - - label of a fake image

        Note: Do not use sigmoid as the last layer of Discriminator. LSGAN needs no sigmoid.
        """
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.loss = nn.MSELoss()

    def get_target_tensor(self, prediction, target_is_real):
        """Create label tensors with the same size as the input.

        Parameters:
            prediction (tensor) - - tpyically the prediction from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            A label tensor filled with ground truth label, and with the size of the input
        """
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real):
        """Calculate loss given Discriminator's output and grount truth labels.

        Parameters:
            prediction (tensor) - - tpyically the prediction output from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            the calculated loss.
        """
        target_tensor = self.get_target_tensor(prediction, target_is_real)
        loss = self.loss(prediction, target_tensor)
        
        return loss

class GANLoss_hinge(nn.Module):
    """Define GAN objectives of hinge version.
    """
    def __init__(self):
        super(GANLoss_hinge, self).__init__()

    def __call__(self, prediction, target_is_real):
        """Calculate loss given Discriminator's output and grount truth labels.

        Parameters:
            prediction (tensor) - - tpyically the prediction output from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            the calculated loss.
        """
        if target_is_real: # real images
            loss = nn.ReLU()(1.0 - prediction).mean()
        else:   # fake images
            loss = nn.ReLU()(1.0 + prediction).mean()

        return loss

class GANLoss_hinge_finetune(nn.Module):
    """Define GAN objectives of hinge version in finetune.
    """
    def __init__(self):
        super(GANLoss_hinge_finetune, self).__init__()

    def __call__(self, prediction, target_is_real):
        """Calculate loss given Discriminator's output and grount truth labels.

        Parameters:
            prediction (tensor) - - tpyically the prediction output from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            output the nearestlabel or the calculated loss.
        """

        if prediction.size()[1] != 1: # B*num_cls*16*16, return the nearest label
            temploss_list = torch.zeros(prediction.size()[1])
            for i in range(prediction.size()[1]):
                if target_is_real: # real images
                    temploss = nn.ReLU()(1.0 - prediction[:,i,:,:]).mean()
                else:   # fake images
                    temploss = nn.ReLU()(1.0 + prediction[:,i,:,:]).mean()
                temploss_list[i] = temploss
            nearest_label = torch.argmax(temploss_list)
            return nearest_label, None
        else: # B*1*16*16, return the loss
            if target_is_real: # real images
                loss = nn.ReLU()(1.0 - prediction).mean()
            else:   # fake images
                loss = nn.ReLU()(1.0 + prediction).mean()
            return loss

class GPLoss(nn.Module):
    """Define gradient panalty loss
    """
    def __init__(self):
        super(GPLoss, self).__init__()

    def __call__(self, d_out, x_in):
        """Calculate gradient panalty loss given Discriminator's output and input.

        Parameters:
            d_out (tensor) - - tpyically the prediction output from a discriminator
            x_in (tensor) - - the inputs corresponding to the d_out

        Returns:
            the calculated loss.
        """
        batch_size = x_in.size(0)
        grad_dout = torch.autograd.grad(outputs=d_out.mean(),
                                  inputs=x_in,
                                  create_graph=True,
                                  retain_graph=True,
                                  only_inputs=True)[0]
        grad_dout2 = grad_dout.pow(2)
        assert (grad_dout2.size() == x_in.size())
        loss = grad_dout2.sum()/batch_size
        return loss