import torch.nn as nn
import logging as l
from models.modules import Proj_eeg, Proj_img, Enc_eeg
from torch.nn import init

class SuperNICE(nn.Module):
    def __init__(self, args):
        super(SuperNICE, self).__init__()

        # Config for both Img and EEG depending on the image features type extraced by CLIP
        self.img_projector_input_dim = 768
        if args.eeg_patch_encoder == "tsconv":
            self.eeg_projector_input_dim = 1440
        elif args.eeg_patch_encoder == "multiscale_1block":
            self.eeg_projector_input_dim =  (250 - args.mstc_pool_kernel_size[1]) // args.mstc_pool_stride[1] + 1
            self.eeg_projector_input_dim = 40 * self.eeg_projector_input_dim
        elif args.eeg_patch_encoder == "multiscale_2block":
            self.eeg_projector_input_dim = 1400
        # Standard parameters for both Img and EEG projectors
        self.proj_dim = args.proj_dim
        self.eeg_proj_do = 0.5
        self.img_proj_do = 0.3

        # Extract MSTC parameters if using multiscale encoder
        mstc_kwargs = {}
        if args.eeg_patch_encoder == "multiscale_1block" or args.eeg_patch_encoder == "multiscale_2block":
            mstc_kwargs = {
                'mstc_out_channels': args.mstc_out_channels,
                'mstc_kernel_sizes': args.mstc_kernel_sizes,
                'mstc_dilation_rates': args.mstc_dilation_rates,
                'mstc_pool_kernel_size': args.mstc_pool_kernel_size,
                'mstc_pool_stride': args.mstc_pool_stride,
                'mstc_dropout_p': args.mstc_dropout_p,
                'pe_dropout_p': args.pe_dropout_p
            }
        self.Enc_eeg = Enc_eeg(config=args.config, patch_encoder=args.eeg_patch_encoder, **mstc_kwargs)
        self.Proj_eeg = Proj_eeg(
            input_dim=self.eeg_projector_input_dim,
            proj_dim=self.proj_dim,
            drop_proj=self.eeg_proj_do,
        )
        self.Proj_img = Proj_img(
            input_dim=self.img_projector_input_dim,
            proj_dim=self.proj_dim,
            drop_proj=self.img_proj_do,
            use_image_projector=args.use_image_projector,
        )

        # self.Enc_eeg = nn.DataParallel(self.Enc_eeg, device_ids=[i for i in range(len(gpus))])
        # self.Proj_eeg = nn.DataParallel(self.Proj_eeg, device_ids=[i for i in range(len(gpus))])
        # self.Proj_img = nn.DataParallel(self.Proj_img, device_ids=[i for i in range(len(gpus))])

    def forward(self, eeg, img):
        # obtain the EEG features
        eeg_features = self.Enc_eeg(eeg)

        # project the features to a shared multimodal embedding space
        eeg_features = self.Proj_eeg(eeg_features)  # shape [batch_size, 768]
        img_features = self.Proj_img(img)  # shape [batch_size, 768]

        # normalize the features
        eeg_features = eeg_features / eeg_features.norm(dim=1, keepdim=True)
        img_features = img_features / img_features.norm(dim=1, keepdim=True)
        return eeg_features, img_features

    def init_weights(self):
        self.Enc_eeg.apply(weights_init_normal)
        self.Proj_eeg.apply(weights_init_normal)
        self.Proj_img.apply(weights_init_normal)


def weights_init_normal(m):
    # Conv2d and Linear
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        init.normal_(m.weight, mean=0.0, std=0.02)
        if m.bias is not None:
            init.constant_(m.bias, 0.0)

    # Normalisation layers
    elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
        init.normal_(m.weight, mean=1.0, std=0.02)
        init.constant_(m.bias, 0.0)
