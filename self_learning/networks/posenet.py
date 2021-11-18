import torch
import torch.nn as nn
import torch.nn.functional as F
from networks.base.basenet import BaseNet
from networks.base.googlenet import GoogLeNet

from self_learning.transformation import runSimulate, simImage, runSimulatefromPred, simImageGray
from PIL import Image

import kornia
import numpy as np

class Regression(nn.Module):
    """Pose regression module.
    Args:
        regid: id to map the length of the last dimension of the input
               feature maps.
        with_embedding: if set True, output activations before pose regression
                        together with regressed poses, otherwise only poses.
    Return:
        xyz: global camera position.
        wpqr: global camera orientation in quaternion.
    """
    def __init__(self, regid, with_embedding=False):
        super(Regression, self).__init__()
        conv_in = {"regress1": 512, "regress2": 528}
        self.with_embedding = with_embedding
        if regid != "regress3":
            self.projection = nn.Sequential(nn.AvgPool2d(kernel_size=5, stride=3),
                                            nn.Conv2d(conv_in[regid], 128, kernel_size=1),
                                            nn.ReLU())
            self.regress_fc_pose = nn.Sequential(nn.Linear(2048, 1024),
                                                 nn.ReLU(),
                                                 nn.Dropout(0.7))
            self.regress_fc_xyz = nn.Linear(1024, 3)
            self.regress_fc_wpqr = nn.Linear(1024, 4)
        else:
            self.projection = nn.AvgPool2d(kernel_size=7, stride=1)
            self.regress_fc_pose = nn.Sequential(nn.Linear(1024, 2048),
                                                 nn.ReLU(),
                                                 nn.Dropout(0.5))
            self.regress_fc_xyz = nn.Linear(2048, 3)
            self.regress_fc_wpqr = nn.Linear(2048, 4)

    def forward(self, x):
        x = self.projection(x)
        x = self.regress_fc_pose(x.view(x.size(0), -1))
        xyz = self.regress_fc_xyz(x)
        wpqr = self.regress_fc_wpqr(x)
        wpqr = F.normalize(wpqr, p=2, dim=1)
        if self.with_embedding:
            return (xyz, wpqr, x)
        return (xyz, wpqr)

class PoseNet(BaseNet):
    '''PoseNet model in [Kendall2015ICCV] Posenet: A convolutional network for real-time 6-dof camera relocalization.'''

    def __init__(self, config, with_embedding=False):
        super(PoseNet, self).__init__(config)
        self.extract = GoogLeNet(with_aux=True)
        self.regress1 = Regression('regress1')
        self.regress2 = Regression('regress2')
        self.regress3 = Regression('regress3', with_embedding=with_embedding)

        # Loss params
        self.learn_weighting = config.learn_weighting
        if self.learn_weighting:
            # Learned loss weighting during training
            sx, sq = config.homo_init
            # Variances variables to learn
            self.sx = nn.Parameter(torch.tensor(sx))
            self.sq = nn.Parameter(torch.tensor(sq))
        else:
            # Fixed loss weighting with beta
            self.beta = config.beta
        """        self.optimizer = optim.AdamW(self.parameters(), lr=0.001)
        self.exp_lr_scheduler = optim.lr_scheduler.CyclicLR(self.optimizer,base_lr=0.001,cycle_momentum=False,max_lr=0.0013,step_size_up=2000)"""

        self.to(self.device)
        self.init_weights_(config.weights_dict)
        self.set_optimizer_(config)

        #top down view of the soccer field
        self.ground_img_path = 'C:\\Users\\mrlix\\Desktop\\ethsecondyear\\project\\Semester-Project-main\\generate synthetic images\\robocup_thicker.jpeg'
        #specify if the self learning loss use gradient
        self.not_use_gradient = False

        self.image = Image.open(self.ground_img_path)
        self.image = self.image.convert('RGB')
        self.final_sl_weight = config.slw
        self.start_sl_weight = config.slws
        self.sl_weight = self.start_sl_weight
        self.sl_epochs = config.sle
        self.sl_step   =  (self.final_sl_weight- self.start_sl_weight)/self.sl_epochs


    def forward(self, x):
        if self.training:
            feat4a, feat4d, feat5b = self.extract(x)
            pose = [self.regress1(feat4a), self.regress2(feat4d), self.regress3(feat5b)]
        else:
            feat5b = self.extract(x)
            pose = self.regress3(feat5b)
        return pose

    def get_inputs_(self, batch, with_label=True,pretrain=False):

        im = batch['im']
        im = im.to(self.device)
        if with_label:
            xyz = batch['xyz'].to(self.device)
            wpqr = batch['wpqr'].to(self.device)
            return im, xyz, wpqr
        else:
            return im


    def predict_(self, batch):
        pose = self.forward(self.get_inputs_(batch, with_label=False))
        xyz, wpqr = pose[0], pose[1]
        return xyz.data.cpu().numpy(), wpqr.data.cpu().numpy()

    def init_weights_(self, weights_dict):
        '''Define how to initialize the model'''

        if weights_dict is None:
            print('Initialize all weigths')
            self.apply(self.xavier_init_func_)
        elif len(weights_dict.items()) == len(self.state_dict()):
            print('Load all weigths')
            self.load_state_dict(weights_dict)
        else:
            print('Init only part of weights')
            self.apply(self.normal_init_func_)
            self.load_state_dict(weights_dict, strict=False)
    def train_epoch(self, dataloader, epoch,pretrain = False ):
        if self.lr_scheduler:
            self.lr_scheduler.step()
        if not pretrain:
            self.sl_weight += self.sl_step
        for i, batch in enumerate(dataloader):
            loss, losses = self.optim_step_(batch,pretrain)
        return loss, losses
    def optim_step_(self, batch,pretrain):
        self.optimizer.zero_grad()
        loss, losses = self.loss_(batch,pretrain)
        loss.backward()
        self.optimizer.step()
        return loss, losses
    def loss_(self, batch,pretrain):
        if pretrain:
            im, xyz_, wpqr_ = self.get_inputs_(batch, with_label=True,pretrain=pretrain)

        else:
            im, xyz_, wpqr_, = self.get_inputs_(batch[0], with_label=True, pretrain=pretrain)
            im_real          = self.get_inputs_(batch[1],with_label=False)
            pred_real = self.forward(im_real)
        criterion = nn.MSELoss()
        pred = self.forward(im)
        loss = 0
        losses = []
        loss_sl = 0
        loss_weighting = [0.3, 0.3, 1.0]
        if self.learn_weighting:
            loss_func = lambda loss_xyz, loss_wpqr: self.learned_weighting_loss(loss_xyz, loss_wpqr, self.sx, self.sq)
        else:
            loss_func = lambda loss_xyz, loss_wpqr: self.fixed_weighting_loss(loss_xyz, loss_wpqr, beta=self.beta)
        for l, w in enumerate(loss_weighting):
            xyz, wpqr = pred[l]

            loss_xyz = criterion(xyz, xyz_)
            loss_wpqr = criterion(wpqr, wpqr_)
            losses.append((loss_xyz, loss_wpqr))  # Remove if not necessary
            loss += w * loss_func(loss_xyz, loss_wpqr)
            if not pretrain:
                #self learning loss: transform the top down view using prediction, compare the transformed top down view of the field
                #with the original image, match white line using grayscale image
                #todo: transforming done, grayscaling done, thresholding mask done, loss done
                xyz_real, wpqr_real = pred_real[l]
                if self.not_use_gradient == False:
                    #loss_sl = calculate_sl_loss(im,xyz,wpqr)
                    x = -np.pi / 2 - np.pi / 6
                    _, ch, row, col = self.img.shape
                    zoom = 1900  # the focal length?
                    K = torch.tensor([[zoom, 0, col / 2], [0, zoom, row / 2.5], [0, 0, 1]]).type(torch.FloatTensor) # todo: what are these design choices
                    #todo: make this function differentiable done
                    H, label = runSimulatefromPred(K, xyz_real, wpqr_real) #xyz.shape batch_size * 3,wpr.shape batch_size*4, H.shape
                    # transfrom ground view using prediction
                    trans_ground = simImageGray(self.image, H,batch_size=xyz_real.shape[0])
                    # gray scale im batch,
                    grayscale = kornia.color.gray.RgbToGrayscale()
                    gray_im = grayscale(im_real)
                    gray_scale_threshold = 0.8
                    mask = gray_im>gray_scale_threshold
                    # masking all parts in the transformed ground view thats not the white line


                    gray_im.register_hook(lambda grad: grad * mask.float())
                    # calculate loss as mean entropy? or maybe
                    loss_mse_f = torch.nn.MSELoss()# todo: note if use bce need first rescale to 0,1 first
                    loss_sl =loss_mse_f( trans_ground, gray_im)
                else:
                    raise(NotImplementedError)
                loss+= self.sl_weight*w* loss_sl # todo: add linear schedule here done
        return loss, losses