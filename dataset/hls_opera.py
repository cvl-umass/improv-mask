from torch.utils.data.dataset import Dataset

import os
import torch
import fnmatch
import numpy as np
import pdb
import torchvision.transforms as transforms
from PIL import Image
import random
import torch.nn.functional as F
from loguru import logger


class RandomScaleCrop(object):
    """
    Credit to Jialong Wu from https://github.com/lorenmt/mtan/issues/34.
    """
    def __init__(self, scale=[1.0, 1.2, 1.5]):
        self.scale = scale

    def __call__(self, feat, wmask, csmask, cmask, snowice):
        height, width = feat.shape[-2:]
        sc = self.scale[random.randint(0, len(self.scale) - 1)]
        h, w = int(height / sc), int(width / sc)
        i = random.randint(0, height - h)
        j = random.randint(0, width - w)
        feat_ = F.interpolate(feat[None, :, i:i + h, j:j + w], size=(height, width), mode='bilinear', align_corners=True).squeeze(0)
        wmask_ = F.interpolate(wmask[None, None, i:i + h, j:j + w], size=(height, width), mode='nearest').squeeze(0).squeeze(0)
        csmask_ = F.interpolate(csmask[None, None, i:i + h, j:j + w], size=(height, width), mode='nearest').squeeze(0)
        cmask_ = F.interpolate(cmask[None, None, i:i + h, j:j + w], size=(height, width), mode='nearest').squeeze(0)
        snowice_ = F.interpolate(snowice[None, None, i:i + h, j:j + w], size=(height, width), mode='nearest').squeeze(0)
        return feat_, wmask_, csmask_, cmask_, snowice_, [sc, h, w, i, j]


class HLSData(Dataset):
    """
    This file is directly modified from https://pytorch.org/docs/stable/torchvision/datasets.html
    """
    def __init__(self, root, split='train', with_nodata_mask=False, with_fmask=False, augmentation=False, flip=False, normalize=False, nodata_val=-9999):
        self.split = split
        self.root = os.path.expanduser(root)
        self.augmentation = augmentation
        self.flip = flip
        self.normalize = None
        self.nodata_val = nodata_val
        self.with_fmask = with_fmask
        self.with_nodata_mask = with_nodata_mask
        if normalize:
            self.normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

        # R\read the data file
        if split=="train":
            self.data_path = os.path.join(root, 'train.npy')
        elif split=="val":
            self.data_path = os.path.join(root, 'val.npy')
        elif split=="test":
            self.data_path = os.path.join(root, 'test.npy')
        else:
            raise NotImplementedError

        with open(self.data_path, "rb") as f:
            npzfile = np.load(f, allow_pickle=True)
            self.fps = npzfile["fps"]
        self.data_len = len(self.fps)
        # if split == "train":
        # self.data_len = 64

    def __getitem__(self, index):
        # fp = os.path.join(self.data_path, f"{index:06d}.npy")
        # fp = os.path.join(self.data_path, self.fps[index])
        tmp_fp = self.fps[index]
        fp = os.path.join(self.root, "/".join(tmp_fp.split("/")[6:]))
        with open(fp, "rb") as f:
            npzfile = np.load(f)
            features = npzfile["features"]  # (512,512,6)
            snowice_mask = npzfile["snowice_mask"]
            water_mask = npzfile["water_mask"]  # (512,512)
            cloudshadow_mask = npzfile["cloudshadow_mask"]  # (512,512)
            cloud_mask = npzfile["cloud_mask"]  # (512,512)
            sun_mask = npzfile["sun_mask"]  # (512,512)
            fmask = npzfile["fmask_data"]
            # cirrus_mask = npzfile["cirrus_mask"]  # (512,512)
            # site_id = npzfile["site_id"]
            # date_str = npzfile["date_str"]
            # tss_value = npzfile["tss_value"]
        
        nodata_mask = (features==self.nodata_val).astype(int)
        nodata_mask = nodata_mask[:512, :512, 0]    # only get 1st channel of features
        feats_tmp = np.where(features==self.nodata_val, np.nanmax(features), features)
        feats_min = np.nanmin(feats_tmp)
        feats_tmp = np.where(features==self.nodata_val, feats_min, features)    # replace nodata values with minimum
        feats_tmp = np.where(np.isnan(feats_tmp), feats_min, feats_tmp)    # replace nan values with minimum
        feats_tmp = np.nan_to_num(feats_tmp, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        feats_tmp = (feats_tmp - np.nanmin(feats_tmp)) / np.maximum((np.nanmax(feats_tmp) - np.nanmin(feats_tmp)), 1)
        # Random crop
        if self.split == "train":
            data_transforms = transforms.Compose([
                transforms.ToTensor(),
                # transforms.RandomCrop(size=(512,512)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
            ])
        else:
            data_transforms = transforms.Compose([
                transforms.ToTensor(),
            ])
        image = feats_tmp.astype(np.float32)
        s1,s2 = snowice_mask.shape
        snowice_mask = np.reshape(snowice_mask.astype(np.float32), (s1,s2,1))
        water_mask = np.reshape(water_mask.astype(np.float32), (s1,s2,1))
        cloudshadow_mask = np.reshape(cloudshadow_mask.astype(np.float32), (s1,s2,1))
        cloud_mask = np.reshape(cloud_mask.astype(np.float32), (s1,s2,1))
        sun_mask = np.reshape(sun_mask.astype(np.float32), (s1,s2,1))
        all_data = np.concatenate((image, snowice_mask, water_mask, cloudshadow_mask, cloud_mask, sun_mask), axis=-1)
        all_data = data_transforms(all_data)
        all_data = all_data[:,:512, :512]   # only get upper left corner (NOTE: this is to avoid all no data values)
        image = all_data[:6,:,:]
        # logger.debug(f"image: {image.shape}, water_mask: {water_mask.shape} all_data: {all_data.shape}")

        
        # Data Augmentation
        if self.split == "train":
            # Add Random channel mixing
            ccm = torch.eye(6)[None,None,:,:]
            r = torch.rand(3,)*0.25 + torch.Tensor([0,1,0])
            filter = r[None, None, :, None]
            ccm = torch.nn.functional.conv2d(ccm, filter, stride=1, padding="same")
            ccm = torch.squeeze(ccm)
            image = torch.tensordot(ccm, image, dims=([1],[0])) # not exactly the same perhaps

            # Add Gaussian noise
            r = torch.rand(1,1)*0.04
            image = image + torch.normal(mean=0.0, std=r[0][0], size=(6,512,512))

        # Min-max Normalization
        # Normalize data
        if (torch.max(image)-torch.min(image)):
            # feats = (image-np.min(image))/(np.max(image)-np.min(image)) # normalize from 0 to 1
            image = image - torch.min(image)
            image = image / torch.maximum(torch.max(image),torch.tensor(1))
        else:
            logger.warning(f"all zero image. setting all labels to zero. index: {index}. {self.split} {fp}")
            image = torch.zeros_like(image)
            all_data = torch.zeros_like(all_data)
        # logger.debug(f"feats min: {np.min(feats)} feats max: {np.max(feats)}")
        
        
        snowice_mask = all_data[-5,:,:]
        snowice_mask = torch.reshape(snowice_mask, (1,512,-1))
        water_mask = all_data[-4,:,:]
        water_mask = torch.reshape(water_mask, (1,512,-1))
        cloudshadow_mask = all_data[-3,:,:]
        cloudshadow_mask = torch.reshape(cloudshadow_mask, (1,512,-1))
        cloud_mask = all_data[-2,:,:]
        cloud_mask = torch.reshape(cloud_mask, (1,512,-1))
        sun_mask = all_data[-1,:,:]
        sun_mask = torch.reshape(sun_mask, (1,512,-1))


        # logger.debug(f"feats: {torch.max(feats)}, {torch.min(feats)}")
        # logger.debug(f"water_mask: {torch.max(water_mask)}, {torch.min(water_mask)}")
        # logger.debug(f"cloudshadow_mask: {torch.max(cloudshadow_mask)}, {torch.min(cloudshadow_mask)}")
        # logger.debug(f"cloud_mask: {torch.max(cloud_mask)}, {torch.min(cloud_mask)}")
        # logger.debug(f"snowice_mask: {torch.max(snowice_mask)}, {torch.min(snowice_mask)}")
        labels = {
            "water_mask": water_mask.type(torch.FloatTensor),
            "cloudshadow_mask": cloudshadow_mask.type(torch.FloatTensor),
            "cloud_mask": cloud_mask.type(torch.FloatTensor),
            "snowice_mask": snowice_mask.type(torch.FloatTensor),
            "sun_mask": sun_mask.type(torch.FloatTensor),
        }
        if not self.with_fmask:
            return (
                image.type(torch.FloatTensor),
                labels
            )
        else:
            if self.with_nodata_mask:
                return (
                    image.type(torch.FloatTensor),
                    labels,
                    fmask[:512,:512],
                    nodata_mask,
                )
            
            else:
                return (
                    image.type(torch.FloatTensor),
                    labels,
                    fmask[:512,:512],
                )


    def __len__(self):
        return self.data_len

