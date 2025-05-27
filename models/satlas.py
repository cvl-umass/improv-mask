import satlaspretrain_models
import torch
from torch import nn
import collections
import torchvision
import requests
from io import BytesIO

# Below function from https://github.com/allenai/satlaspretrain_models/blob/main/satlaspretrain_models/utils.py#L155
def adjust_state_dict_prefix(state_dict, needed, prefix=None, prefix_allowed_count=None):
    """
    Adjusts the keys in the state dictionary by replacing 'backbone.backbone' prefix with 'backbone'.

    Args:
        state_dict (dict): Original state dictionary with 'backbone.backbone' prefixes.

    Returns:
        dict: Modified state dictionary with corrected prefixes.
    """
    new_state_dict = {}
    for key, value in state_dict.items():
        # Assure we're only keeping keys that we need for the current model component. 
        if not needed in key:
            continue

        # Update the key prefixes to match what the model expects.
        if prefix is not None:
            while key.count(prefix) > prefix_allowed_count:
                key = key.replace(prefix, '', 1)

        new_state_dict[key] = value
    return new_state_dict

class SatlasModel(torch.nn.Module):
    def __init__(self, num_inp_feats=6, fpn=True, model_name="Sentinel2_SwinB_SI_RGB"):
        super(SatlasModel, self).__init__()

        weights_manager = satlaspretrain_models.Weights()
        self.first = nn.Conv2d(num_inp_feats, 3, 1) # from 6 channels to 3
        if model_name == "Sentinel2_SwinB_SI_RGB":
            self.backbone = weights_manager.get_pretrained_model(model_identifier=model_name, fpn=fpn)
            self.backbone_channels = self.backbone.upsample.layers[-1][-2].out_channels
        elif model_name == "Sentinel2_SwinT_SI_RGB":
            model = weights_manager.get_pretrained_model(model_identifier=model_name, fpn=False)
            out_channels = [
                [4, 96],
                [8, 192],
                [16, 384],
                [32, 768],
            ]
            model_fpn = FPN(out_channels, 128)
            if fpn: # Download and load weights for FPN
                weights_url = 'https://huggingface.co/allenai/satlas-pretrain/resolve/main/sentinel2_swint_si_rgb.pth?download=true'
                response = requests.get(weights_url)
                if response.status_code == 200:
                    weights_file = BytesIO(response.content)
                weights = torch.load(weights_file)
                fpn_state_dict = adjust_state_dict_prefix(weights, 'fpn', 'intermediates.0.', 0)
                model_fpn.load_state_dict(fpn_state_dict)
            model_upsample = Upsample(model_fpn.out_channels)
            self.backbone = torch.nn.Sequential(
                model,
                model_fpn,
                model_upsample,
            )
            self.backbone_channels = 128
        elif model_name == "Sentinel2_Resnet50_SI_RGB":
            model = weights_manager.get_pretrained_model(model_identifier="Sentinel2_Resnet50_SI_RGB", fpn=False)
            model.backbone.freeze_bn = False    # NOTE: means backbone is not frozen during training
            out_channels = [
                [4, 256],
                [8, 512],
                [16, 1024],
                [32, 2048],
            ]
            model_fpn = FPN(out_channels, 128)
            if fpn:
                weights_url = 'https://huggingface.co/allenai/satlas-pretrain/resolve/main/sentinel2_resnet50_si_rgb.pth?download=true'
                response = requests.get(weights_url)
                if response.status_code == 200:
                    weights_file = BytesIO(response.content)
                weights = torch.load(weights_file)
                fpn_state_dict = adjust_state_dict_prefix(weights, 'fpn', 'intermediates.0.', 0)
                model_fpn.load_state_dict(fpn_state_dict)
            model_upsample = Upsample(model_fpn.out_channels)
            self.backbone = torch.nn.Sequential(
                model,
                model_fpn,
                model_upsample,
            )
            self.backbone_channels = 128

    def forward(self, x):
        x = self.first(x)
        x = self.backbone(x)
        return x[0]


class SatlasHead(torch.nn.Module):
    def __init__(self, backbone_channels, out_channels):
        super(SatlasHead, self).__init__()

        num_layers = 2
        layers = []
        for _ in range(num_layers-1):
            layer = torch.nn.Sequential(
                torch.nn.Conv2d(backbone_channels, backbone_channels, 3, padding=1),
                torch.nn.ReLU(inplace=True),
            )
            layers.append(layer)
        layers.append(torch.nn.Conv2d(backbone_channels, out_channels, 3, padding=1))
        self.head = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.head(x)


# Below classes from https://github.com/allenai/satlaspretrain_models/blob/main/satlaspretrain_models/models/fpn.py#L6
class FPN(torch.nn.Module):
    def __init__(self, backbone_channels, out_channels):    # NOTE: modified out_channels to match checkpoint
        super(FPN, self).__init__()

        # out_channels = backbone_channels[0][1]
        in_channels_list = [ch[1] for ch in backbone_channels]
        self.fpn = torchvision.ops.FeaturePyramidNetwork(in_channels_list=in_channels_list, out_channels=out_channels)

        self.out_channels = [[ch[0], out_channels] for ch in backbone_channels]

    def forward(self, x):
        inp = collections.OrderedDict([('feat{}'.format(i), el) for i, el in enumerate(x)])
        output = self.fpn(inp)
        output = list(output.values())

        return output


class Upsample(torch.nn.Module):
    # Computes an output feature map at 1x the input resolution.
    # It just applies a series of transpose convolution layers on the
    # highest resolution features from the backbone (FPN should be applied first).

    def __init__(self, backbone_channels):
        super(Upsample, self).__init__()
        self.in_channels = backbone_channels

        out_channels = backbone_channels[0][1]
        self.out_channels = [(1, out_channels)] + backbone_channels

        layers = []
        depth, ch = backbone_channels[0]
        while depth > 1:
            next_ch = max(ch//2, out_channels)
            layer = torch.nn.Sequential(
                torch.nn.Conv2d(ch, ch, 3, padding=1),
                torch.nn.ReLU(inplace=True),
                torch.nn.ConvTranspose2d(ch, next_ch, 4, stride=2, padding=1),
                torch.nn.ReLU(inplace=True),
            )
            layers.append(layer)
            ch = next_ch
            depth /= 2

        self.layers = torch.nn.Sequential(*layers)

    def forward(self, x):
        output = self.layers(x[0])
        return [output] + x
