import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, ConcatDataset, random_split
from torch import Tensor
import pytorch_lightning as pl

import numpy as np
from PIL import Image
from pathlib import Path
import os
import copy
from typing import List
from tqdm import tqdm
import random
from typing import Optional
from itertools import groupby
from operator import itemgetter
from functools import partial
import random
from einops import rearrange

import cv2


class LitDataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.len_train_loader = None
        self.len_test_loader = None
        self.img_size = cfg.Dataset.image_size

        self.norm_transform = lambda x: x * 2. - 1.

        self.train_transform = transforms.Compose([VidPad((0,40,0,40)), VidResize((self.img_size,self.img_size)), VidToTensor(), self.norm_transform])
        self.test_transform = transforms.Compose([VidPad((0,40,0,40)), VidResize((self.img_size,self.img_size)), VidToTensor(), self.norm_transform])
        
        o_resize = None
        p_resize = None
        vp_size = cfg.STDiff.Diffusion.unet_config.sample_size
        vo_size = cfg.STDiff.DiffNet.MotionEncoder.image_size

        if vp_size != self.img_size:
            p_resize = transforms.Resize(vp_size, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True)
        if vo_size != self.img_size:
            o_resize = transforms.Resize(vo_size, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True)

        self.collate_fn = partial(svrfcn, rand_Tp=cfg.Dataset.rand_Tp, rand_predict=cfg.Dataset.rand_predict, o_resize=o_resize, p_resize=p_resize)

    def setup(self, stage: Optional[str] = None):
        self.train_set = None
        self.test_set = None

        if stage in ('train',None):
            TrainData = MyDataset(Path(self.cfg.Dataset.dir), transform=self.train_transform, train=True, num_observed_frames=self.cfg.Dataset.num_observed_frames, num_predict_frames=self.cfg.Dataset.num_predict_frames)
            self.train_set = TrainData()

        if stage in ('test','predict'):
            TestData = MyDataset(Path(self.cfg.Dataset.dir), transform=self.test_transform, train=False, num_observed_frames=self.cfg.Dataset.test_num_observed_frames, num_predict_frames=self.cfg.Dataset.test_num_predict_frames)
            self.test_set = TestData()

    def train_dataloader(self):
        return DataLoader(self.train_set, shuffle=True, batch_size=self.cfg.Dataset.batch_size, num_workers=self.cfg.Dataset.num_workers, drop_last=True, collate_fn=self.collate_fn) if self.train_set else None

    def test_dataloader(self):
        return DataLoader(self.test_set, shuffle=False, batch_size=self.cfg.Dataset.batch_size, num_workers=self.cfg.Dataset.num_workers, drop_last=True, collate_fn=self.collate_fn) if self.test_set else None

def get_lightning_module_dataloader(cfg):
    pl_datamodule = LitDataModule(cfg)
    pl_datamodule.setup(stage=cfg.Dataset.stage)
    return pl_datamodule.train_dataloader(), pl_datamodule.test_dataloader()

class MyDataset(object):
    """
    a wrapper for ClipDataset, inspired by the original implementation of KTH dataset
    the original frame size is (H, W) = (160,240)
    Return the train/test dataset
    """
    def __init__(self, dir, transform, num_observed_frames, num_predict_frames):
        """
        Args:
            dir --- Directory for extracted video frames
            train --- True for training dataset, False for test dataset
            transform --- torchvision transform functions
            num_observed_frames --- number of past frames
            num_predict_frames --- number of future frames
        """
        self.num_observed_frames = num_observed_frames
        self.num_predict_frames = num_predict_frames
        self.clip_length = num_observed_frames + num_predict_frames
        self.transform = transform

        self.path = Path(dir).absolute()

        frame_folders = [self.path.joinpath(s) for s in os.listdir(self.path)]

        self.clips = self.__getClips__(frame_folders)

    def __call__(self):
        """
        Returns:
            clip_set --- ClipDataset object
        """
        clip_set = ClipDataset(self.num_observed_frames, self.num_predict_frames, self.clips, self.transform)
           
        return clip_set
    
    def __getClips__(self, frame_folders):
        clips = []
        for folder in frame_folders:
            key = lambda path:int(''.join(filter(str.isdigit,str(path).split("/")[-1])))
            img_files = sorted(list(folder.glob('*.png')),key=key)
            clip_num = len(img_files) // self.clip_length
            rem_num = len(img_files) % self.clip_length
            img_files = img_files[rem_num // 2 : rem_num//2 + clip_num*self.clip_length]
            for i in range(clip_num):
                clips.append(img_files[i*self.clip_length : (i+1)*self.clip_length])

        return clips
    
class ClipDataset(Dataset):
    """
    Video clips dataset
    """
    def __init__(self, num_observed_frames, num_predict_frames, clips, transform):
        """
        Args:
            num_observed_frames --- number of past frames
            num_predict_frames --- number of future frames
            clips --- List of video clips frames file path
            transform --- torchvision transforms for the image

        Return batched Sample:
            past_clip --- Tensor with shape (batch_size, num_observed_frames, C, H, W)
            future_clip --- Tensor with shape (batch_size, num_predict_frames, C, H, W)
        """
        self.num_observed_frames = num_observed_frames
        self.num_predict_frames = num_predict_frames
        self.clips = clips
        self.transform = transform

    def __len__(self):
        return len(self.clips)
    
    def __getitem__(self, index: int):
        """
        Returns:
            past_clip: Tensor with shape (num_observed_frames, C, H, W)
            future_clip: Tensor with shape (num_predict_frames, C, H, W)
        """
        if torch.is_tensor(index):
            index = index.to_list()
        
        clip_imgs = self.clips[index]
        imgs = []
        for img_path in clip_imgs:
            img = Image.open(img_path.absolute().as_posix()).convert('RGB')
            imgs.append(img)
        
        original_clip = self.transform(imgs)

        past_clip = original_clip[0:self.num_observed_frames, ...]
        future_clip = original_clip[-self.num_predict_frames:, ...]
        return past_clip, future_clip

def svrfcn(batch_data, rand_Tp = 3, rand_predict = True, o_resize = None, p_resize = None):
    """
    Single video dataset random future frames collate function
    batch_data: list of tuples, each tuple is (observe_clip, predict_clip)
    """
    
    observe_clips, predict_clips = zip(*batch_data)
    observe_batch = torch.stack(observe_clips, dim=0)
    predict_batch = torch.stack(predict_clips, dim=0)

    # output the last frame of observation, taken as the first frame of autoregressive prediction
    observe_last_batch = observe_batch[:, -1:, ...]
    
    max_Tp = predict_batch.shape[1]
    if rand_predict:
        assert rand_Tp <= max_Tp, "Invalid rand_Tp"
        rand_idx = np.sort(np.random.choice(max_Tp, rand_Tp, replace=False))
        rand_idx = torch.from_numpy(rand_idx)
        rand_predict_batch = predict_batch[:, rand_idx, ...]
    else:
        rand_idx = torch.linspace(0, max_Tp-1, max_Tp, dtype = torch.int)
        rand_predict_batch = predict_batch
    To = observe_batch.shape[1]
    idx_o = torch.linspace(0, To-1 , To, dtype = torch.int)

    if p_resize is not None:
        N, T, _, _, _ = rand_predict_batch.shape
        rand_predict_batch = p_resize(rand_predict_batch.flatten(0, 1))
        rand_predict_batch = rearrange(rand_predict_batch, "(N T) C H W -> N T C H W", N = N, T=T)
        #also resize the last frame of observation
        observe_last_batch = p_resize(observe_last_batch.flatten(0, 1))
        observe_last_batch = rearrange(observe_last_batch, "(N T) C H W -> N T C H W", N = N, T=1)
        
    if o_resize is not None:
        N, T, _, _, _ = observe_batch.shape
        observe_batch = o_resize(observe_batch.flatten(0, 1))
        observe_batch = rearrange(observe_batch, "(N T) C H W -> N T C H W", N = N, T=T)
    return (observe_batch, rand_predict_batch, observe_last_batch, idx_o.to(torch.float), rand_idx.to(torch.float) + To)

#####################################################################################
class VidResize(object):
    def __init__(self, *args, **resize_kwargs):
        self.resize_kwargs = resize_kwargs
        self.resize_kwargs['antialias'] = True
        self.resize_kwargs['interpolation'] = transforms.InterpolationMode.BICUBIC
        self.args = args

    def __call__(self, clip: List[Image.Image]):
        for i in range(len(clip)):
            clip[i] = transforms.Resize(*self.args, **self.resize_kwargs)(clip[i])

        return clip

class VidRandomHorizontalFlip(object):
    def __init__(self, p: float):
        assert p>=0 and p<=1, "invalid flip probability"
        self.p = p
    
    def __call__(self, clip: List[Image.Image]):
        if np.random.rand() < self.p:
            for i in range(len(clip)):
                clip[i] = transforms.functional.hflip(clip[i])
        return clip

class VidRandomVerticalFlip(object):
    def __init__(self, p: float):
        assert p>=0 and p<=1, "invalid flip probability"
        self.p = p
    
    def __call__(self, clip: List[Image.Image]):
        if np.random.rand() < self.p:
            for i in range(len(clip)):
                clip[i] = transforms.functional.vflip(clip[i])
        return clip

class VidToTensor(object):
    def __call__(self, clip: List[Image.Image]):
        """
        Return: clip --- Tensor with shape (T, C, H, W)
        """
        for i in range(len(clip)):
            clip[i] = transforms.ToTensor()(clip[i])
        clip = torch.stack(clip, dim = 0)

        return clip

class VidPad(object):
    """
    If pad, Do not forget to pass the mask to the transformer encoder.
    """
    def __init__(self, *args, **kwargs):
        self.kwargs = kwargs
        self.args = args

    def __call__(self, clip: List[Image.Image]):
        for i in range(len(clip)):
            clip[i] = transforms.Pad(*self.args, **self.kwargs)(clip[i])

        return clip