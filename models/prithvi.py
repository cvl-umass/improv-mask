import os
import torch
from torch import nn
import matplotlib.pyplot as plt
import numpy as np
import rasterio
import yaml
from prithvi.Prithvi import MaskedAutoencoderViT

class PrithviModel(torch.nn.Module):
    def __init__(self, num_inp_feats=6):
        super(PrithviModel, self).__init__()

        # load weights
        weights_path = "./prithvi/Prithvi_100M.pt"
        checkpoint = torch.load(weights_path)

        # read model config
        model_cfg_path = "./prithvi/Prithvi_100M_config.yaml"
        with open(model_cfg_path) as f:
            model_config = yaml.safe_load(f)

        self.model_args, self.train_args = model_config["model_args"], model_config["train_params"]

        # let us use only 1 frame for now (the model was trained on 3 frames)
        self.model_args["num_frames"] = 1

        # instantiate model
        self.model = MaskedAutoencoderViT(**self.model_args)

        # load weights into model
        # strict=false since we are loading with only 1 frame, but the warning is expected
        del checkpoint['pos_embed']
        del checkpoint['decoder_pos_embed']
        _ = self.model.load_state_dict(checkpoint, strict=False)

        # weights_manager = satlaspretrain_models.Weights()
        self.first = nn.Sequential(
            nn.Conv2d(num_inp_feats, 6, kernel_size=66, stride=2, padding=0), # from 512x512 to 224x224
            # nn.Conv2d(num_inp_feats, 6, kernel_size=5, stride=2, padding=0),
            # nn.Conv2d(6, 6, kernel_size=5, stride=1, padding=0),
        )
        self.backbone_channels = self.model_args["embed_dim"]

    def forward(self, x):
        x = self.first(x)
        # print(f"x: {x.shape}")
        x = torch.unsqueeze(x, 2)
        features, _, _ = self.model.forward_encoder(x, mask_ratio=0)
        # drop cls token
        reshaped_features = features[:, 1:, :]

        # reshape
        feature_img_side_length = int(np.sqrt(reshaped_features.shape[1]))
        reshaped_features = reshaped_features.contiguous().view(-1, feature_img_side_length, feature_img_side_length, self.model_args["embed_dim"])
        # channels first
        reshaped_features = reshaped_features.contiguous().permute(0, 3, 1, 2)
        return reshaped_features
        # print(f"reshaped_features: {reshaped_features.shape}")
        # out = self.segmentation_head(reshaped_features)
        # return out



class PrithviHead(torch.nn.Module):
    def __init__(self, backbone_channels, out_channels):
        super(PrithviHead, self).__init__()

        # read model config
        # model_cfg_path = "./prithvi/Prithvi_100M_config.yaml"
        # with open(model_cfg_path) as f:
        #     model_config = yaml.safe_load(f)

        # self.model_args, self.train_args = model_config["model_args"], model_config["train_params"]
    
        # Segmentation head
        # num_classes = 1
        # in_channels = 768
        upscaling_block = lambda in_channels, out_channels: nn.Sequential(
            nn.Upsample(scale_factor=2), 
            nn.Conv2d(kernel_size=3, in_channels=in_channels, out_channels=out_channels, padding=1), 
            nn.ReLU()
        )
        embed_dims = [backbone_channels // (2**i) for i in range(5)]
        upscaling_block_last = nn.Sequential(
            nn.Upsample(size=(512,512)), 
            nn.Conv2d(kernel_size=3, in_channels=embed_dims[-1], out_channels=embed_dims[-1], padding=1), 
            nn.ReLU()
        )
        self.segmentation_head = nn.Sequential(
            *[
            upscaling_block(embed_dims[i], embed_dims[i+1]) for i in range(4)
            ],
            upscaling_block_last,
            nn.Conv2d(kernel_size=1, in_channels=embed_dims[-1], out_channels=out_channels),
        )

    def forward(self, x):
        return self.segmentation_head(x)