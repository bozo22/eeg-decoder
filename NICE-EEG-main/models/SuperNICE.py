import torch.nn as nn
import logging as l
from models.modules import weights_init_normal, Enc_eeg, Proj_eeg, Proj_img, CrossAttention


class SuperNICE(nn.Module):
    def __init__(self, args):
        super(SuperNICE, self).__init__()

        use_old_image_projector = args.debug_higher_scores in ['old_image_projector', 'both_proj_centers', 'all']

        # Config for both Img and EEG depending on the image features type extraced by CLIP
        self.img_projector_input_dim = 1024 if args.image_features_type == 'hidden_states' else 768
        self.eeg_encoder_flatten = False if args.image_features_type == 'hidden_states' else True
        self.eeg_projector_input_dim = 40 if args.image_features_type == 'hidden_states' else 1440
        # Standard parameters for both Img and EEG projectors
        self.proj_dim = args.proj_dim
        self.eeg_proj_do = 0.5
        self.img_proj_do = 0.3

        # Attention parameters
        self.use_attention = args.use_attn
        self.att_heads = args.att_heads
        self.att_blocks = args.att_blocks
        self.att_dropout = args.att_dropout
        
        self.Enc_eeg = Enc_eeg(flatten=self.eeg_encoder_flatten)
        self.Proj_eeg = Proj_eeg(input_dim = self.eeg_projector_input_dim, proj_dim = self.proj_dim, drop_proj = self.eeg_proj_do)
        self.Proj_img = Proj_img(input_dim = self.img_projector_input_dim, proj_dim = self.proj_dim, drop_proj = self.img_proj_do, 
                                 use_old_image_projector = use_old_image_projector)
        if self.use_attention:
            self.Cross_att = CrossAttention(emb_dim=self.proj_dim, num_heads=self.att_heads, 
                                            n_blocks=self.att_blocks, dropout_p=self.att_dropout)
        else:
            self.Cross_att = nn.Identity() # If no attention, return identity

        # self.Enc_eeg = nn.DataParallel(self.Enc_eeg, device_ids=[i for i in range(len(gpus))])
        # self.Proj_eeg = nn.DataParallel(self.Proj_eeg, device_ids=[i for i in range(len(gpus))])
        # self.Proj_img = nn.DataParallel(self.Proj_img, device_ids=[i for i in range(len(gpus))])
        
    def forward(self, eeg, img):
        # obtain the EEG features
        eeg_features = self.Enc_eeg(eeg)

        # project the features to a shared multimodal embedding space
        eeg_features = self.Proj_eeg(eeg_features) # shape [batch_size, 768]
        img_features = self.Proj_img(img) # shape [batch_size, 768]

        # apply cross-attention
        eeg_features, img_features = self.Cross_att((eeg_features, img_features))
        
        # normalize the features
        eeg_features = eeg_features / eeg_features.norm(dim=1, keepdim=True)
        img_features = img_features / img_features.norm(dim=1, keepdim=True)
        return eeg_features, img_features
        
    def init_weights(self):
        self.Enc_eeg.apply(weights_init_normal)
        self.Proj_eeg.apply(weights_init_normal)
        self.Proj_img.apply(weights_init_normal)
        if self.use_attention:
            self.Cross_att.init_weights()
        
        
        
        
        