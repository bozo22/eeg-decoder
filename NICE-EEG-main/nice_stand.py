"""
Object recognition Things-EEG2 dataset

use 250 Hz data
"""

import os
import argparse
import itertools
import time
import numpy as np
import pandas as pd
from pprint import pprint
import logging as l
import wandb

import torch
import torch.nn as nn

from torch.utils.data import Subset

from models.modules import weights_init_normal, Enc_eeg, Proj_eeg, Proj_img, CrossAttention
from utils.utils import load_model, save_model, seed_experiments, wandb_login

# gpus = [0]
# os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
# os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, gpus))
NICE_path = os.path.dirname(os.path.abspath(__file__))
result_path = os.path.join(NICE_path, 'results')
model_checkpoint_path = os.path.join(result_path, 'checkpoints')
model_idx = 'test0'


parser = argparse.ArgumentParser(description='Experiment Stimuli Recognition test with CLIP encoder')
# Architectures
parser.add_argument('--dnn', default='clip', type=str)
# Training parameters
parser.add_argument('--epoch', default='200', type=int)
parser.add_argument('--num_sub', default=10, type=int,
                    help='number of subjects used in the experiments. ')
parser.add_argument('-batch_size', '--batch-size', default=1000, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--dataset_path', default='Things-EEG2/Preprocessed_data_250Hz/', type=str, help='Path to the dataset. ')
# Auxiliary parameters
parser.add_argument('--seed', default=2023, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--device', default='gpu', type=str, choices=['gpu', 'cpu'], help='Device to use for training.')
parser.add_argument('--debug', action='store_true', help='If True, will run in debug mode with only a fraction of the dataset.')
# WandB parameters
parser.add_argument('--disable_wandb', action='store_true', help='If True, will not use wandb.')
parser.add_argument('--run_group', default=None, type=str, help='Group name for the WandB run.')

# Attention experiment parameters
parser.add_argument('--use_attn', action='store_true', help='If True, will use attention.')
parser.add_argument('--att_heads', default=4, type=int, help='Number of attention heads.')
parser.add_argument('--att_blocks', default=2, type=int, help='Number of attention blocks.')
parser.add_argument('--att_dropout', default=0.3, type=float, help='Dropout rate for the attention.')

args = parser.parse_args()
pprint(args)

# Set device
device = torch.device('cuda' if torch.cuda.is_available() and args.device == 'gpu' else 'cpu')
print(f'Using device: {device}')

if args.debug:
    l.basicConfig(level=l.DEBUG, format='%(levelname)s: %(message)s')
    l.debug(">>> Running in DEBUG mode!")

# Seed experiments
seed_experiments(args.seed)

run_name = f"{args.dnn}"
if args.use_attn:
    run_name += f"attn(H-{args.att_heads}, B-{args.att_blocks}, DO-{args.att_dropout})"
if args.debug:
    run_name = "[DEBUG]" + run_name

wandb_login(args.disable_wandb)
wandb.init(
    entity="EEG_decoder",
    project="EEG-Decoder",
    name=run_name,
    config=vars(args),
    mode="disabled" if args.disable_wandb else "online",
    group=args.run_group
)
wandb.define_metric("epoch")
wandb.define_metric("train/*", step_metric="epoch")
wandb.define_metric("val/*", step_metric="epoch")
# sweep_configuration = read_sweep_config("NICE-EEG-main/sweep_config.yaml")

# Image2EEG
class IE():
    def __init__(self, args, nsub):
        super(IE, self).__init__()
        self.args = args
        self.num_class = 200
        self.batch_size = args.batch_size
        self.batch_size_test = 400
        self.batch_size_img = 500 
        self.n_epochs = args.epoch

        # Dim of projection layers for both Img and EEG + Dim of attention
        self.proj_dim = 768
        self.eeg_proj_do = 0.5
        self.img_proj_do = 0.3

        self.lr = 0.0002
        self.b1 = 0.5
        self.b2 = 0.999
        self.nSub = nsub

        self.start_epoch = 0
        self.eeg_data_path = args.dataset_path
        self.img_data_path = os.path.join(NICE_path, 'dnn_feature/')
        self.test_center_path = os.path.join(NICE_path, 'dnn_feature/')
        self.pretrain = False

        os.makedirs(result_path, exist_ok=True)

        self.Tensor = torch.FloatTensor
        self.LongTensor = torch.LongTensor

        self.Cross_att = CrossAttention(emb_dim = self.proj_dim, num_heads = args.att_heads,
                                        dropout_p = args.att_dropout, n_blocks=args.att_blocks,
                                        use_attention=args.use_attn).to(device)

        self.criterion_l1 = torch.nn.L1Loss().cuda()
        self.criterion_l2 = torch.nn.MSELoss().cuda()
        self.criterion_cls = torch.nn.CrossEntropyLoss().to(device)
        self.Enc_eeg = Enc_eeg().to(device)
        self.Proj_eeg = Proj_eeg(proj_dim = self.proj_dim, drop_proj = self.eeg_proj_do).to(device)
        self.Proj_img = Proj_img(proj_dim = self.proj_dim, drop_proj = self.img_proj_do).to(device)
        # self.Enc_eeg = nn.DataParallel(self.Enc_eeg, device_ids=[i for i in range(len(gpus))])
        # self.Proj_eeg = nn.DataParallel(self.Proj_eeg, device_ids=[i for i in range(len(gpus))])
        # self.Proj_img = nn.DataParallel(self.Proj_img, device_ids=[i for i in range(len(gpus))])

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        print('initial define done.')


    def get_eeg_data(self):
        train_data = []
        train_label = []
        test_data = []
        test_label = np.arange(200)

        train_data = np.load(os.path.join(self.eeg_data_path, 'sub-' + format(self.nSub, '02'), 'preprocessed_eeg_training.npy'), allow_pickle=True)
        train_data = train_data['preprocessed_eeg_data']
        # Average across repetitions
        train_data = np.mean(train_data, axis=1) # Shape: (total_nr_train_imgs x 1 x channels x 250)
        train_data = np.expand_dims(train_data, axis=1)

        test_data = np.load(self.eeg_data_path + '/sub-' + format(self.nSub, '02') + '/preprocessed_eeg_test.npy', allow_pickle=True)
        test_data = test_data['preprocessed_eeg_data']
        # Average across repetitions
        test_data = np.mean(test_data, axis=1) # Shape: (total_nr_test_imgs x 1 x channels x 250)
        test_data = np.expand_dims(test_data, axis=1)

        return train_data, train_label, test_data, test_label

    def get_image_data(self):
        train_img_feature = np.load(self.img_data_path + self.args.dnn + '_feature_maps_training.npy', allow_pickle=True)
        test_img_feature = np.load(self.img_data_path + self.args.dnn + '_feature_maps_test.npy', allow_pickle=True)

        train_img_feature = np.squeeze(train_img_feature)
        test_img_feature = np.squeeze(test_img_feature)

        return train_img_feature, test_img_feature
        
    def update_lr(self, optimizer, lr):
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


    def train(self):
        
        self.Enc_eeg.apply(weights_init_normal)
        self.Proj_eeg.apply(weights_init_normal)
        self.Proj_img.apply(weights_init_normal)
        self.Cross_att.init_weights()

        train_eeg, _, test_eeg, test_label = self.get_eeg_data()
        train_img_feature, test_img_feature = self.get_image_data() 
        test_center = np.load(self.test_center_path + 'center_' + self.args.dnn + '.npy', allow_pickle=True)

        # shuffle the training data
        train_shuffle = np.random.permutation(len(train_eeg))
        train_eeg = train_eeg[train_shuffle]
        train_img_feature = train_img_feature[train_shuffle]

        val_eeg = torch.from_numpy(train_eeg[:740])
        val_image = torch.from_numpy(train_img_feature[:740])

        train_eeg = torch.from_numpy(train_eeg[740:])
        train_image = torch.from_numpy(train_img_feature[740:])

        dataset = torch.utils.data.TensorDataset(train_eeg, train_image)
        if args.debug:
            l.debug("Using only 100 samples from the dataset")
            dataset = Subset(dataset, range(100))
        self.dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=self.batch_size, shuffle=True)
        val_dataset = torch.utils.data.TensorDataset(val_eeg, val_image)
        self.val_dataloader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=self.batch_size, shuffle=False)

        test_eeg = torch.from_numpy(test_eeg)
        test_img_feature = torch.from_numpy(test_img_feature)
        test_center = torch.from_numpy(test_center)
        test_label = torch.from_numpy(test_label)
        test_dataset = torch.utils.data.TensorDataset(test_eeg, test_label)
        self.test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=self.batch_size_test, shuffle=False)

        # Optimizers
        self.optimizer = torch.optim.Adam(itertools.chain(self.Enc_eeg.parameters(), self.Proj_eeg.parameters(), self.Proj_img.parameters(), self.Cross_att.parameters()), 
                                          lr=self.lr, 
                                          betas=(self.b1, self.b2))

        num = 0
        best_loss_val = np.inf

        for e in range(self.n_epochs):
            epoch_losses = []
            epoch_losses_eeg = []
            epoch_losses_img = []

            self.Enc_eeg.train()
            self.Proj_eeg.train()
            self.Proj_img.train()
            self.Cross_att.train()
            starttime_epoch = time.time()

            for i, (eeg, img) in enumerate(self.dataloader):

                # img = Variable(img.cuda().type(self.Tensor))
                eeg = eeg.type(self.Tensor).to(device)
                img_features = img.type(self.Tensor).to(device)
                labels = torch.arange(eeg.shape[0])  # used for the loss
                labels = labels.type(self.LongTensor).to(device)

                # obtain the features
                eeg_features = self.Enc_eeg(eeg)

                # project the features to a multimodal embedding space
                eeg_features = self.Proj_eeg(eeg_features) # shape [batch_size, 768]
                img_features = self.Proj_img(img_features) # shape [batch_size, 768]

                # apply cross-attention
                eeg_features, img_features = self.Cross_att(eeg_features, img_features)
            
                # normalize the features
                eeg_features = eeg_features / eeg_features.norm(dim=1, keepdim=True)
                img_features = img_features / img_features.norm(dim=1, keepdim=True)

                # cosine similarity as the logits
                logit_scale = self.logit_scale.exp()
                logits_per_eeg = logit_scale * eeg_features @ img_features.t()
                logits_per_img = logits_per_eeg.t()

                loss_eeg = self.criterion_cls(logits_per_eeg, labels)
                loss_img = self.criterion_cls(logits_per_img, labels)

                loss_cos = (loss_eeg + loss_img) / 2

                # total loss
                loss = loss_cos
                epoch_losses.append(loss.item())
                epoch_losses_eeg.append(loss_eeg.item())
                epoch_losses_img.append(loss_img.item())

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            # Log epoch metrics
            avg_epoch_loss = sum(epoch_losses) / len(epoch_losses)
            avg_epoch_loss_eeg = sum(epoch_losses_eeg) / len(epoch_losses_eeg)
            avg_epoch_loss_img = sum(epoch_losses_img) / len(epoch_losses_img)
            wandb.log({
                "epoch": e + 1,
                f"train/loss/subj{self.nSub}": avg_epoch_loss,
                f"train/loss_eeg/subj{self.nSub}": avg_epoch_loss_eeg,
                f"train/loss_img/subj{self.nSub}": avg_epoch_loss_img
            })

            if (e + 1) % 1 == 0:
                self.Enc_eeg.eval()
                self.Proj_eeg.eval()
                self.Proj_img.eval()
                self.Cross_att.eval()

                with torch.no_grad():
                    # * validation part
                    val_losses = []
                    val_losses_eeg = []
                    val_losses_img = []
                    for i, (veeg, vimg) in enumerate(self.val_dataloader):

                        veeg = veeg.type(self.Tensor).to(device)
                        vimg_features = vimg.type(self.Tensor).to(device)
                        vlabels = torch.arange(veeg.shape[0])
                        vlabels = vlabels.type(self.LongTensor).to(device)

                        veeg_features = self.Enc_eeg(veeg)
                        veeg_features = self.Proj_eeg(veeg_features)
                        vimg_features = self.Proj_img(vimg_features)
                        
                        # Adding cross att
                        veeg_features, vimg_features = self.Cross_att(veeg_features, vimg_features)

                        veeg_features = veeg_features / veeg_features.norm(dim=1, keepdim=True)
                        vimg_features = vimg_features / vimg_features.norm(dim=1, keepdim=True)

                        logit_scale = self.logit_scale.exp()
                        vlogits_per_eeg = logit_scale * veeg_features @ vimg_features.t()
                        vlogits_per_img = vlogits_per_eeg.t()

                        vloss_eeg = self.criterion_cls(vlogits_per_eeg, vlabels)
                        vloss_img = self.criterion_cls(vlogits_per_img, vlabels)

                        vloss = (vloss_eeg + vloss_img) / 2
                        val_losses.append(vloss.item())
                        val_losses_eeg.append(vloss_eeg.item())
                        val_losses_img.append(vloss_img.item())

                    avg_val_loss = sum(val_losses) / len(val_losses)
                    avg_val_loss_eeg = sum(val_losses_eeg) / len(val_losses_eeg)
                    avg_val_loss_img = sum(val_losses_img) / len(val_losses_img)
                    wandb.log({
                        "epoch": e + 1,
                        f"val/loss/subj{self.nSub}": avg_val_loss,
                        f"val/loss_eeg/subj{self.nSub}": avg_val_loss_eeg,
                        f"val/loss_img/subj{self.nSub}": avg_val_loss_img
                        })

                    if vloss <= best_loss_val:
                        best_loss_val = vloss
                        best_epoch = e + 1
                        os.makedirs(model_checkpoint_path, exist_ok=True)
                        print(f"New best epoch - {best_epoch}")
                        # Save models, handling both DataParallel and non-DataParallel cases
                        save_model(self.Enc_eeg, model_checkpoint_path, model_idx)
                        save_model(self.Cross_att, model_checkpoint_path, model_idx)
                        save_model(self.Proj_eeg, model_checkpoint_path, model_idx)
                        save_model(self.Proj_img, model_checkpoint_path, model_idx)

                print('Epoch:', e,
                      '  Cos eeg: %.4f' % loss_eeg.detach().cpu().numpy(),
                      '  Cos img: %.4f' % loss_img.detach().cpu().numpy(),
                      '  loss val: %.4f' % vloss.detach().cpu().numpy(),
                      )
            
            endtime_epoch = time.time()
            print(f"Epoch {e} took {endtime_epoch - starttime_epoch} seconds")

        # * test part
        all_center = test_center
        total = 0
        top1 = 0
        top3 = 0
        top5 = 0

        self.Enc_eeg = load_model(self.Enc_eeg, model_checkpoint_path, model_idx)
        self.Cross_att = load_model(self.Cross_att, model_checkpoint_path, model_idx)
        self.Proj_eeg = load_model(self.Proj_eeg, model_checkpoint_path, model_idx)
        self.Proj_img = load_model(self.Proj_img, model_checkpoint_path, model_idx)

        self.Enc_eeg.eval()
        self.Cross_att.eval()
        self.Proj_eeg.eval()
        self.Proj_img.eval()

        with torch.no_grad():
            for i, (teeg, tlabel) in enumerate(self.test_dataloader):
                teeg = teeg.type(self.Tensor).to(device)
                tlabel = tlabel.type(self.LongTensor).to(device)

                timg = test_img_feature.type(self.Tensor).to(device)

                timg = self.Proj_img(timg)
                tfea = self.Proj_eeg(self.Enc_eeg(teeg))
                tfea, timg = self.Cross_att(tfea, timg)

                tfea = tfea / tfea.norm(dim=1, keepdim=True)
                similarity = (tfea @ timg.t()).softmax(dim=-1)
                _, indices = similarity.topk(5)

                tt_label = tlabel.view(-1, 1)
                total += tlabel.size(0)
                top1 += (tt_label == indices[:, :1]).sum().item()
                top3 += (tt_label == indices[:, :3]).sum().item()
                top5 += (tt_label == indices).sum().item()

            
            top1_acc = float(top1) / float(total)
            top3_acc = float(top3) / float(total)
            top5_acc = float(top5) / float(total)

            wandb.log({
                f"test/top1_accuracy/subj{self.nSub}": top1_acc,
                f"test/top3_accuracy/subj{self.nSub}": top3_acc,
                f"test/top5_accuracy/subj{self.nSub}": top5_acc
            })

        print('The test Top1-%.6f, Top3-%.6f, Top5-%.6f' % (top1_acc, top3_acc, top5_acc))
        print(f"The best epoch is: {best_epoch}")
        
        return top1_acc, top3_acc, top5_acc
        # writer.close()


def main():
    num_sub = args.num_sub   
    cal_num = 0
    aver = []
    aver3 = []
    aver5 = []
    
    for i in range(num_sub):

        cal_num += 1
        starttime = time.time()
        seed_n = np.random.randint(args.seed)

        print('Subject %d' % (i+1))
        ie = IE(args, i + 1)

        Acc, Acc3, Acc5 = ie.train()
        print('THE BEST ACCURACY IS ' + str(Acc))


        endtime = time.time()
        print('subject %d duration: %.2f minutes' % (i+1, (endtime - starttime) / 60))

        aver.append(Acc)
        aver3.append(Acc3)
        aver5.append(Acc5)

    aver.append(np.mean(aver))
    aver3.append(np.mean(aver3))
    aver5.append(np.mean(aver5))

    column = np.arange(1, cal_num+1).tolist()
    column.append('ave')
    pd_all = pd.DataFrame(columns=column, data=[aver, aver3, aver5])
    pd_all.to_csv(os.path.join(result_path, 'train_results.csv'))
    
    wandb.finish()

if __name__ == "__main__":
    print(time.asctime(time.localtime(time.time())))
    main()
    print(time.asctime(time.localtime(time.time())))