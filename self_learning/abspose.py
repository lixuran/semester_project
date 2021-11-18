import argparse
import os
import time

import kornia
import numpy as np
import torch
from torchvision import transforms
import torch.utils.data as data
import pandas as pd

from self_learning.utils.datasets.robocup import RobocupDataset
from utils.common.config_parser import AbsPoseConfig
from utils.common.setup import *
from utils.datasets.preprocess import *
from utils.datasets.abspose import AbsPoseDataset
from utils.common.visdom_templates import PoseNetVisTmp, OptimSearchVisTmp
import networks

from self_learning.transformation import runSimulatefromPred, simImageGray

def setup_config(config):
    print('Setup configurations...')
    # Seeding
    make_deterministic(config.seed)

    # Setup logging dir
    if not os.path.exists(config.odir):
        os.makedirs(config.odir)
    config.log = os.path.join(config.odir, 'log.txt') if config.training else os.path.join(config.odir, 'test_results.txt')
    config.ckpt_dir = os.path.join(config.odir, 'ckpt')
    if not os.path.exists(config.ckpt_dir):
        os.makedirs(config.ckpt_dir)

    # Setup running devices
    if torch.cuda.is_available():
        print('Use GPU device:{}.'.format(config.gpu))
        config.device = torch.device('cuda:{}'.format(config.gpu))
    else:
        print('No GPU available, use CPU device.')
        config.device = torch.device("cpu")
    delattr(config, 'gpu')

    if config.validate:
        config.validate = config.validate[0]

    # Setup datasets
    config.data_class = AbsPoseDataset

    # Define image preprocessing
    im_mean = os.path.join(config.data_root, config.dataset, config.image_mean) if config.image_mean else None
    if config.crop:
        crop = 'random' if config.training else 'center'
    else:
        crop = None
    #todo: might want to create 2 different ops for each dataset
    config.ops = get_transform_ops(config.rescale, im_mean, crop, crop_size=config.crop, normalize=config.normalize)
    config.val_ops = get_transform_ops(config.rescale, im_mean, 'center', crop_size=config.crop, normalize=config.normalize)
    delattr(config, 'crop')
    delattr(config, 'rescale')
    delattr(config, 'normalize')

    # Model initialization
    config.start_epoch = 0
    config.start_pre_epoc = 0
    config.weights_dir = None
    config.weights_dict = None
    config.optimizer_dict = None
    if config.pretrained:
        config.weights_dir = config.pretrained[0]
        config.weights_dict = torch.load(config.weights_dir)
    if config.resume:
        config.weights_dir = config.resume[0]
        checkpoint = torch.load(config.weights_dir)
        assert config.network == checkpoint['network']
        if (checkpoint['pretrain_done']):
            config.start_pre_epoch = config.pre_epoch # skip pretrain

            config.start_epoch = checkpoint['last_epoch'] + 1
        else:
            config.start_pre_epoch = checkpoint['last_epoch'] + 1
        config.weights_dict = checkpoint['state_dict']
        config.optimizer_dict = checkpoint['optimizer']
    delattr(config, 'resume')
    delattr(config, 'pretrained')

    # Setup optimizer
    optim = config.optim
    optim_tag = ''
    if config.optim == 'Adam':
        optim_tag = 'Adam_eps{}'.format(config.epsilon)
        delattr(config, 'momentum')
    elif config.optim == 'SGD':
        optim_tag = 'SGD_mom{}'.format(config.momentum)
        delattr(config, 'epsilon')
    optim_tag = '{}_{}'.format(optim_tag, config.lr_init)
    if config.lr_decay:
        config.lr_decay_step = int(config.lr_decay[1])
        config.lr_decay_factor = float(config.lr_decay[0])
        config.lr_decay = True
        optim_tag = '{}_lrd{}-{}'.format(optim_tag, config.lr_decay_step, config.lr_decay_factor)
    optim_tag = '{}_wd{}'.format(optim_tag, config.weight_decay)
    config.optim_tag = optim_tag

def train(net, config, log, simulated_train_loader,real_train_loader, simulated_val_loader=None,real_val_loader=None):

    optim_search = True
    # Setup visualizer
    if not optim_search:
        visman, tloss_meter, pos_acc_meter, rot_acc_meter, losses_meters, homo_meters = PoseNetVisTmp.get_meters(config, with_losses=False, with_homos=True)
    else:
        visman, tloss_meter, pos_acc_meter, rot_acc_meter = OptimSearchVisTmp.get_meters(config)
        homo_meters = None
    start_time = time.time()
    print('Start pre training on simulated data from {config.start_pre_epoch} to {config.pre_epochs}.'.format(config=config))
    for epoch in range(config.start_pre_epoch, config.pre_epochs): #todo: fix the restart value here by adding it to ckpt done
        net.train()  # Switch to training mode

        loss, losses = net.train_epoch(simulated_train_loader, epoch,pretrain= True)
        lprint('Epoch {}, loss:{}'.format(epoch + 1, loss), log)
        # Update homo variable meters
        if config.learn_weighting and homo_meters is not None:
            homo_meters[0].update(X=epoch + 1, Y=net.sx)
            homo_meters[1].update(X=epoch + 1, Y=net.sq)

        # Update loss meters
        """
        for i, val in enumerate(losses):
           losses_meters[i][0].update(X=epoch+1, Y=losses[i][0]) # pos_loss
           losses_meters[i][1].update(X=epoch+1, Y=losses[i][1]) # rot_loss
        """
        tloss_meter.update(X=epoch + 1, Y=loss)
        if config.validate and (epoch + 1) % config.validate == 0 and epoch > 0:
            # Evaluate on validation set
            abs_err = test(net, config, log, simulated_val_loader, real_val_loader)  # todo
            ckpt = {'last_epoch': epoch,
                    'network': config.network,
                    'state_dict': net.state_dict(),
                    'optimizer': net.optimizer.state_dict(),
                    'abs_err': abs_err,
                    'pretrain_done': False
                    }
            ckpt_name = 'checkpoint_{epoch}_{abs_err[0]:.2f}m_{abs_err[1]:.2f}deg.pth'.format(epoch=(epoch + 1),
                                                                                              abs_err=abs_err)
            torch.save(ckpt, os.path.join(config.ckpt_dir, ckpt_name))
            lprint('Save checkpoint: {}'.format(ckpt_name), log)

            # Update validation acc
            pos_acc_meter.update(X=epoch + 1, Y=abs_err[0])
            rot_acc_meter.update(X=epoch + 1, Y=abs_err[1])
        visman.save_state()
    print('Start training from {config.start_epoch} to {config.epochs}.'.format(config=config))

    for epoch in range(config.start_epoch, config.epochs):
        net.train() # Switch to training mode

        loss, losses = net.train_epoch(real_train_loader, epoch)#todo
        lprint('Epoch {}, loss:{}'.format(epoch+1, loss), log)
        # Update homo variable meters
        if config.learn_weighting and homo_meters is not None:
            homo_meters[0].update(X=epoch+1, Y=net.sx)
            homo_meters[1].update(X=epoch+1, Y=net.sq)

        # Update loss meters
        """
        for i, val in enumerate(losses):
           losses_meters[i][0].update(X=epoch+1, Y=losses[i][0]) # pos_loss
           losses_meters[i][1].update(X=epoch+1, Y=losses[i][1]) # rot_loss
        """
        tloss_meter.update(X=epoch+1, Y=loss)
        if config.validate and (epoch+1) % config.validate == 0 and epoch > 0 :
            # Evaluate on validation set
            abs_err = test(net, config, log, simulated_val_loader,real_val_loader)
            # todo change ckpt to indicate whether in pretraining
            ckpt ={'last_epoch': epoch,
                   'network': config.network,
                   'state_dict': net.state_dict(),
                   'optimizer' : net.optimizer.state_dict(),
                   'abs_err' : abs_err,
                   'pretrain_done': True
                   }
            ckpt_name = 'checkpoint_{epoch}_{abs_err[0]:.2f}m_{abs_err[1]:.2f}deg.pth'.format(epoch=(epoch+1), abs_err=abs_err)
            torch.save(ckpt, os.path.join(config.ckpt_dir, ckpt_name))
            lprint('Save checkpoint: {}'.format(ckpt_name), log)

            # Update validation acc
            pos_acc_meter.update(X=epoch+1, Y=abs_err[0])
            rot_acc_meter.update(X=epoch+1, Y=abs_err[1])
        visman.save_state()
    lprint('Total training time {0:.4f}s'.format((time.time() - start_time)), log)
#todo: change this to add loss on real data done
def test(net, config, log, simulated_data_loader,real_data_loader, err_thres=(2, 5)):
    realtxt=open("/home/xurali/Semester-Project/visloc-apr/real.txt", "w")
    predictedtxt=open("/home/xurali/Semester-Project/visloc-apr/predicted.txt", "w")

    ground_img_path = 'C:\\Users\\mrlix\\Desktop\\ethsecondyear\\project\\Semester-Project-main\\generate synthetic images\\robocup_thicker.jpeg'
    # specify if the self learning loss use gradient
    image = Image.open(ground_img_path)
    image = image.convert('RGB')

    print('Evaluate on dataset:{}'.format(real_data_loader.dataset.dataset))
    net.eval()
    pos_err = []
    ori_err = []
    sl_err  = []
    with torch.no_grad():
        for i,(batch,batch2) in enumerate(real_data_loader):
            xyz, wpqr = net.predict_(batch) #todo: fix this
            xyz_ = batch['xyz'].data.numpy()
            wpqr_ = batch['wpqr'].data.numpy()

            xyz_real, wpqr_real = net.predict_(batch2)

            x = -np.pi / 2 - np.pi / 6
            _, ch, row, col = image.shape
            zoom = 1900  # the focal length?
            K = torch.tensor([[zoom, 0, col / 2], [0, zoom, row / 2.5], [0, 0, 1]]).type(
                torch.FloatTensor)  # todo: what are these design choices
            # todo:  this function still not  differentiable, make this a batched version
            H, label = runSimulatefromPred(K, xyz_real,
                                           wpqr_real)  # xyz.shape batch_size * 3,wpr.shape batch_size*4, H.shape
            # transfrom ground view using prediction
            trans_ground = simImageGray(image, H)
            # gray scale im batch,
            grayscale = kornia.color.gray.RgbToGrayscale()
            gray_im = grayscale(batch2["im"])
            gray_scale_threshold = 0.8
            mask = gray_im > gray_scale_threshold
            # masking all parts in the transformed ground view thats not the white line

            gray_im.register_hook(lambda grad: grad * mask.float())
            # calculate loss as mean entropy? or maybe
            loss_mse_f = torch.nn.MSELoss()  # todo: if use bce need first rescale to 0,1 first
            loss_sl = loss_mse_f(trans_ground, gray_im)
            #save predicted
            df_predicted=pd.DataFrame(np.concatenate((xyz.transpose(),wpqr.transpose())).transpose())
            predictedtxt.write(df_predicted.to_string(header=False, index=False))

            #save real
            df_real=pd.DataFrame(np.concatenate((xyz_.transpose(),wpqr_.transpose())).transpose())
            realtxt.write(df_real.to_string(header=False, index=False))
            t_err = np.linalg.norm(xyz - xyz_, axis=1)
            q_err = cal_quat_angle_error(wpqr, wpqr_)
            pos_err += list(t_err)
            ori_err += list(q_err)
            sl_err += list(loss_sl)
    err = (np.median(pos_err), np.median(ori_err),np.median(loss_sl))
    passed = 0
    for i, perr in enumerate(pos_err):
        if perr < err_thres[0] and ori_err[i] < err_thres[1]:
            passed += 1
    lprint('Accuracy on simulation: ({err[0]:.2f}m, {err[1]:.2f}deg,{err[2]:.2f}) Pass({err_thres[0]}m, {err_thres[1]}deg): {rate:.2f}% '.format(err=err, err_thres=err_thres, rate=100.0 * passed / i), log)

    realtxt.close()
    predictedtxt.close()
    return err
class ConcatDataset(torch.utils.data.Dataset):
    def __init__(self, *datasets):
        self.datasets = datasets

    def __getitem__(self, i):
        return tuple(d[i] for d in self.datasets)

    def __len__(self):
        return min(len(d) for d in self.datasets)


def main():
    #todo: alter the train to have a pretrain state and a training state    done
    #todo: adjust the pipeline to add a self learning loss                  done
    #todo: write new loss using geometric computations                      done
    #todo: rewrite the loss to allow gradient update                        done
    #todo: change the self learning process                                 done
    #todo: testing performance                                              doing
    #11/14
    #--------------------------
    # todo: train original posenet, done

    #todo: update config file done
    #todo: change train and test func() done
    #todo: make sure the restart loading works done
    #todo: add linear schedule for loss done
    #todo: batchify           done
    #todo: some variable should not reuire grad done


    #todo: upload to github, sync with server done
    #todo: compile test                       doing
    #todo: submit for training                doing

    # --------------------------
    # 11/15

    # todo: modify loss with guassian kernel  to do
    # todo: compile test and submit for training  to do
    # todo: read the paper
    # todo: report progress
    # todo: think about what to do next, problems, and new approaches



    # 11/16
    # todo: visualize the transformed image
    # todo: limit degree of freedom for prediction
    # todo: work on something new


    #os.environ["CUDA_VISIBLE_DEVICES"]=""
    # Setup
    config = AbsPoseConfig().parse()
    setup_config(config)
    log = open(config.log, 'a')
    lprint(config_to_string(config), log)

    # Datasets configuration
    simulated_data_src = AbsPoseDataset(config.simulated_dataset, config.data_root, config.pose_txt, config.ops)
    simulated_data_loader = data.DataLoader(simulated_data_src, batch_size=config.batch_size, shuffle=config.training, num_workers=config.num_workers)
    lprint('simulated Dataset total samples: {}'.format(len(simulated_data_src)))

    real_data_src = RobocupDataset(config.real_dataset, config.data_root, config.ops)
    real_len = len(real_data_src)
    real_train_set, real_val_set = torch.utils.data.random_split(real_data_src, [real_len*8//10, real_len - real_len*8//10])
    cat_src = ConcatDataset(simulated_data_src,real_train_set)
    real_data_loader = data.DataLoader(cat_src, batch_size=config.batch_size, shuffle=config.training,
                                            num_workers=config.num_workers)
    lprint('real Dataset total samples: {}'.format(len(real_data_src)))

    if config.validate:
        val_simulated_data_src = AbsPoseDataset(config.simulated_dataset, config.data_root, config.val_pose_txt, config.val_ops)
        val_simulated_loader = data.DataLoader(val_simulated_data_src, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)

        cat_valid_src = ConcatDataset(val_simulated_data_src, real_val_set)

        val_real_loader = data.DataLoader(cat_valid_src, batch_size=config.batch_size, shuffle=False,
                                     num_workers=config.num_workers)

    else:
        val_simulated_loader = None
        val_real_loader = None

    if config.weights_dir:
        lprint('Load weights dict {}'.format(config.weights_dir))
    net = networks.__dict__[config.network](config)
    lprint('Model params: {} Optimizer params: {}'.format(len(net.state_dict()), len(net.optimizer.param_groups[0]['params'])))

    if config.training:
        train(net, config, log, simulated_data_loader,real_data_loader, val_simulated_loader,val_real_loader)
    else:
        test(net, config, log, simulated_data_loader,real_data_loader)
    log.close()

if __name__ == '__main__':
    main()