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
        self.len_val_loader = None
        self.len_test_loader = None
        self.img_size = cfg.Dataset.image_size

        self.norm_transform = lambda x: x * 2. - 1.

        self.train_transform = transforms.Compose([VidCenterCrop((120, 120)), VidResize((self.img_size, self.img_size)), VidRandomHorizontalFlip(0.5), VidRandomVerticalFlip(0.5), VidToTensor(), self.norm_transform])
        self.test_transform = transforms.Compose([VidCenterCrop((120, 120)), VidResize((self.img_size, self.img_size)), VidToTensor(), self.norm_transform])
        
        o_resize = None
        p_resize = None
        vp_size = cfg.STDiff.Diffusion.unet_config.sample_size
        vo_size = cfg.STDiff.DiffNet.MotionEncoder.image_size
        if vp_size != self.img_size:
            p_resize = transforms.Resize(vp_size, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True)
        if vo_size != self.img_size:
            o_resize = transforms.Resize(vo_size, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True)
        self.collate_fn = partial(svrfcn, rand_Tp=cfg.Dataset.rand_Tp, rand_predict=cfg.Dataset.rand_predict, o_resize=o_resize, p_resize=p_resize, half_fps=cfg.Dataset.half_fps)

    def setup(self, stage: Optional[str] = None):
        # Assign Train/val split(s) for use in Dataloaders
        if stage in (None, "fit"):
            TrainData = Dataset(self.cfg.Dataset.dir, transform = self.train_transform, train = True, val = True, 
                                        num_observed_frames= self.cfg.Dataset.num_observed_frames, num_predict_frames= self.cfg.Dataset.num_predict_frames)
            self.train_set, self.val_set = TrainData()
            
            # Use all training dataset for the final training
            if self.cfg.Dataset.phase == 'deploy':
                self.train_set = ConcatDataset([self.train_set, self.val_set])

            dev_set_size = self.cfg.Dataset.dev_set_size
            if dev_set_size is not None:
                self.train_set, _ = random_split(self.train_set, [dev_set_size, len(self.train_set) - dev_set_size], generator=torch.Generator().manual_seed(2021))
                self.val_set, _ = random_split(self.val_set, [dev_set_size, len(self.val_set) - dev_set_size], generator=torch.Generator().manual_seed(2021))
            
            self.len_train_loader = len(self.train_dataloader())
            self.len_val_loader = len(self.val_dataloader())

        # Assign Test split(s) for use in Dataloaders
        if stage in (None, "test"):
            TestData = Dataset(self.cfg.Dataset.dir, transform = self.test_transform, train = False, val = False, 
                                    num_observed_frames= self.cfg.Dataset.test_num_observed_frames, num_predict_frames= self.cfg.Dataset.test_num_predict_frames)
            self.test_set = TestData()

            dev_set_size = self.cfg.Dataset.dev_set_size
            if dev_set_size is not None:
                self.test_set, _ = random_split(self.test_set, [dev_set_size, len(self.test_set) - dev_set_size], generator=torch.Generator().manual_seed(2021))
            self.len_test_loader = len(self.test_dataloader())

    def train_dataloader(self):
        return DataLoader(self.train_set, shuffle = True, batch_size=self.cfg.Dataset.batch_size, num_workers=self.cfg.Dataset.num_workers, drop_last = True, collate_fn = self.collate_fn)

    def val_dataloader(self):
        return DataLoader(self.val_set, shuffle = False, batch_size=self.cfg.Dataset.batch_size, num_workers=self.cfg.Dataset.num_workers, drop_last = True, collate_fn = self.collate_fn)

    def test_dataloader(self):
        return DataLoader(self.test_set, shuffle = False, batch_size=self.cfg.Dataset.batch_size, num_workers=self.cfg.Dataset.num_workers, drop_last = False, collate_fn = self.collate_fn)


def get_lightning_module_dataloader(cfg):
    pl_datamodule = LitDataModule(cfg)
    pl_datamodule.setup()
    return pl_datamodule.train_dataloader(), pl_datamodule.val_dataloader(), pl_datamodule.test_dataloader()

class Dataset(object):
    """
    a wrapper for ClipDataset, inspired by the original implementation of KTH dataset
    the original frame size is (H, W) = (160,240)
    Split the KTH dataset and return the train and test dataset
    """
    def __init__(self, dir, transform, train, val,
                 num_observed_frames, num_predict_frames):
        """
        Args:
            KTH_dir --- Directory for extracted KTH video frames
            train --- True for training dataset, False for test dataset
            transform --- torchvision transform functions
            num_observed_frames --- number of past frames
            num_predict_frames --- number of future frames
        """
        self.num_observed_frames = num_observed_frames
        self.num_predict_frames = num_predict_frames
        self.clip_length = num_observed_frames + num_predict_frames
        self.transform = transform
        self.color_mode = 'RGB'

        self.path = Path(dir).absolute()
        self.train = train
        self.val = val
        if self.train:
            self.video_ids = list(range(25))
            if self.val:
                self.val_video_ids = [random.randint(0, 16)]
                self.video_ids.remove(self.val_video_ids[0])
        else:
            self.video_ids = list(range(16, 25))

        frame_folders = self.__getFramesFolder__(self.video_ids)
        self.clips = self.__getClips__(frame_folders)
        
        if self.val:
            val_frame_folders = self.__getFramesFolder__(self.val_video_ids)
            self.val_clips = self.__getClips__(val_frame_folders)

    def __call__(self):
        """
        Returns:
            clip_set --- ClipDataset object
        """
        
        clip_set = ClipDataset(self.num_observed_frames, self.num_predict_frames, self.clips, self.transform, self.color_mode)
        if self.val:
            val_clip_set = ClipDataset(self.num_observed_frames, self.num_predict_frames, self.val_clips, self.transform, self.color_mode)
            return clip_set, val_clip_set
        else:
            return clip_set
    
    def __getClips__(self, frame_folders):
        clips = []
        for folder in frame_folders:
            img_files = sorted(list(folder.glob('*')))
            clip_num = len(img_files) // self.clip_length
            rem_num = len(img_files) % self.clip_length
            img_files = img_files[rem_num // 2 : rem_num//2 + clip_num*self.clip_length]
            for i in range(clip_num):
                clips.append(img_files[i*self.clip_length : (i+1)*self.clip_length])

        return clips
    
    def __getFramesFolder__(self, video_ids):
        """
        Get the frames folders for ClipDataset
        Returns:
            return_folders --- the returned video frames folders
        """

        frame_folders = [self.path.joinpath(s) for s in os.listdir(self.path)]

        return_folders = []
        for ff in frame_folders:
            video_id = int(str(ff).strip().split('_')[2][-2:])
            if video_id in video_ids:
                return_folders.append(ff)
        
        return return_folders
    
class ClipDataset(Dataset):
    """
    Video clips dataset
    """
    def __init__(self, num_observed_frames, num_predict_frames, clips, transform, color_mode):
        """
        Args:
            num_observed_frames --- number of past frames
            num_predict_frames --- number of future frames
            clips --- List of video clips frames file path
            transform --- torchvision transforms for the image
            color_mode --- 'RGB' for RGB dataset, 'grey_scale' for grey_scale dataset

        Return batched Sample:
            past_clip --- Tensor with shape (batch_size, num_observed_frames, C, H, W)
            future_clip --- Tensor with shape (batch_size, num_predict_frames, C, H, W)
        """
        self.num_observed_frames = num_observed_frames
        self.num_predict_frames = num_predict_frames
        self.clips = clips
        self.transform = transform
        if color_mode != 'RGB' and color_mode != 'grey_scale':
            raise ValueError("Unsupported color mode!!")
        else:
            self.color_mode = color_mode

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
            if self.color_mode == 'RGB':
                img = Image.open(img_path.absolute().as_posix()).convert('RGB')
            else:
                img = Image.open(img_path.absolute().as_posix()).convert('L')
            imgs.append(img)
        
        original_clip = self.transform(imgs)

        past_clip = original_clip[0:self.num_observed_frames, ...]
        future_clip = original_clip[-self.num_predict_frames:, ...]
        return past_clip, future_clip

    def visualize_clip(self, clip, file_name):
        """
        save a video clip to GIF file
        Args:
            clip: tensor with shape (clip_length, C, H, W)
        """
        imgs = []
        for i in range(clip.shape[0]):
            img = transforms.ToPILImage()(clip[i, ...])
            imgs.append(img)
        
        videodims = img.size
        fourcc = cv2.VideoWriter_fourcc(*'MP4V')    
        video = cv2.VideoWriter(Path(file_name).absolute().as_posix(), fourcc, 10, videodims)
        for img in imgs:
            video.write(cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR))
        video.release()
        #imgs[0].save(str(Path(file_name).absolute()), save_all = True, append_images = imgs[1:])


def svrfcn(batch_data, rand_Tp = 3, rand_predict = True, o_resize = None, p_resize = None, half_fps = False):
    """
    Single video dataset random future frames collate function
    batch_data: list of tuples, each tuple is (observe_clip, predict_clip)
    """
    
    observe_clips, predict_clips = zip(*batch_data)
    observe_batch = torch.stack(observe_clips, dim=0)
    predict_batch = torch.stack(predict_clips, dim=0)

    #output the last frame of observation, taken as the first frame of autoregressive prediction
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

    if half_fps:
        if observe_batch.shape[1] > 2:
            observe_batch = observe_batch[:, ::2, ...]
            idx_o = idx_o[::2, ...]

        rand_predict_batch = rand_predict_batch[:, ::2, ...]
        rand_idx = rand_idx[::2, ...]
        observe_last_batch = observe_batch[:, -1:, ...]

    if p_resize is not None:
        N, T, _, _, _ = rand_predict_batch.shape
        rand_predict_batch = p_resize(rand_predict_batch.flatten(0, 1))
        rand_predict_batch = rearrange(rand_predict_batch, "(N T) C H W -> N T C H W", N = N, T=T)
        #als resize the last frame of observation
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

class VidCenterCrop(object):
    def __init__(self, *args, **kwargs):
        self.kwargs = kwargs
        self.args = args

    def __call__(self, clip: List[Image.Image]):
        for i in range(len(clip)):
            clip[i] = transforms.CenterCrop(*self.args, **self.kwargs)(clip[i])

        return clip

class VidCrop(object):
    def __init__(self, *args, **kwargs):
        self.kwargs = kwargs
        self.args = args

    def __call__(self, clip: List[Image.Image]):
        for i in range(len(clip)):
            clip[i] = transforms.functional.crop(clip[i], *self.args, **self.kwargs)

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

class VidNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
    def __call__(self, clip: Tensor):
        """
        Return: clip --- Tensor with shape (T, C, H, W)
        """
        T, _, _, _ = clip.shape
        for i in range(T):
            clip[i, ...] = transforms.Normalize(self.mean, self.std)(clip[i, ...])

        return clip

class VidReNormalize(object):
    def __init__(self, mean, std):
        try:
            self.inv_std = [1.0/s for s in std]
            self.inv_mean = [-m for m in mean]
            self.renorm = transforms.Compose([transforms.Normalize(mean = [0., 0., 0.],
                                                                std = self.inv_std),
                                            transforms.Normalize(mean = self.inv_mean,
                                                                std = [1., 1., 1.])])
        except TypeError:
            #try normalize for grey_scale images.
            self.inv_std = 1.0/std
            self.inv_mean = -mean
            self.renorm = transforms.Compose([transforms.Normalize(mean = 0.,
                                                                std = self.inv_std),
                                            transforms.Normalize(mean = self.inv_mean,
                                                                std = 1.)])

    def __call__(self, clip: Tensor):
        """
        Return: clip --- Tensor with shape (T, C, H, W)
        """
        T, _, _, _ = clip.shape
        for i in range(T):
            clip[i, ...] = self.renorm(clip[i, ...])

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

def mean_std_compute(dataset, device, color_mode = 'RGB'):
    """
    arguments:
        dataset: pytorch dataloader
        device: torch.device('cuda:0') or torch.device('cpu') for computation
    return:
        mean and std of each image channel.
        std = sqrt(E(x^2) - (E(X))^2)
    """
    data_iter= iter(dataset)
    sum_img = None
    square_sum_img = None
    N = 0

    pgbar = tqdm(desc = 'summarizing...', total = len(dataset))
    for idx, sample in enumerate(data_iter):
        past, future = sample
        clip = torch.cat([past, future], dim = 0)
        N += clip.shape[0]

        img = torch.sum(clip, axis = 0)

        if idx == 0:
            sum_img = img
            square_sum_img = torch.square(img)
            sum_img = sum_img.to(torch.device(device))
            square_sum_img = square_sum_img.to(torch.device(device))
        else:
            img = img.to(device)
            sum_img = sum_img + img
            square_sum_img = square_sum_img + torch.square(img)
        
        pgbar.update(1)
    
    pgbar.close()

    mean_img = sum_img/N
    mean_square_img = square_sum_img/N
    if color_mode == 'RGB':
        mean_r, mean_g, mean_b = torch.mean(mean_img[0, :, :]), torch.mean(mean_img[1, :, :]), torch.mean(mean_img[2, :, :])
        mean_r2, mean_g2, mean_b2 = torch.mean(mean_square_img[0,:,:]), torch.mean(mean_square_img[1,:,:]), torch.mean(mean_square_img[2,:,:])
        std_r, std_g, std_b = torch.sqrt(mean_r2 - torch.square(mean_r)), torch.sqrt(mean_g2 - torch.square(mean_g)), torch.sqrt(mean_b2 - torch.square(mean_b))

        return ([mean_r.cpu().numpy(), mean_g.data.cpu().numpy(), mean_b.cpu().numpy()], [std_r.cpu().numpy(), std_g.cpu().numpy(), std_b.cpu().numpy()])
    else:
        mean = torch.mean(mean_img)
        mean_2 = torch.mean(mean_square_img)
        std = torch.sqrt(mean_2 - torch.square(mean))

        return (mean.cpu().numpy(), std.cpu().numpy())

def get_data_scaler(config):
  """Data normalizer. Assume data are always in [0, 1]."""
  if config.Dataset:
    # Rescale to [-1, 1]
    return lambda x: x * 2. - 1.
  else:
    return lambda x: x

def get_data_inverse_scaler(config):
  """Inverse data normalizer."""
  if config.Dataset.centered:
    # Rescale [-1, 1] to [0, 1]
    return lambda x: (x + 1.) / 2.
  else:
    return lambda x: x

def visualize_batch_clips(gt_past_frames_batch, gt_future_frames_batch, pred_frames_batch, file_dir, renorm_transform = None, desc = None):
    """
        pred_frames_batch: tensor with shape (N, future_clip_length, C, H, W)
        gt_future_frames_batch: tensor with shape (N, future_clip_length, C, H, W)
        gt_past_frames_batch: tensor with shape (N, past_clip_length, C, H, W)
    """
    if not Path(file_dir).exists():
        Path(file_dir).mkdir(parents=True, exist_ok=True) 
    def save_clip(clip, file_name):
        imgs = []
        if renorm_transform is not None:
            clip = renorm_transform(clip)
            clip = torch.clamp(clip, min = 0., max = 1.0)
        for i in range(clip.shape[0]):
            img = transforms.ToPILImage()(clip[i, ...])
            imgs.append(img)

        imgs[0].save(str(Path(file_name).absolute()), save_all = True, append_images = imgs[1:], loop = 0)
    
    def append_frames(batch, max_clip_length):
        d = max_clip_length - batch.shape[1]
        batch = torch.cat([batch, batch[:, -2:-1, :, :, :].repeat(1, d, 1, 1, 1)], dim = 1)
        return batch
    max_length = max(gt_future_frames_batch.shape[1], gt_past_frames_batch.shape[1])
    max_length = max(max_length, pred_frames_batch.shape[1])
    if gt_past_frames_batch.shape[1] < max_length:
        gt_past_frames_batch = append_frames(gt_past_frames_batch, max_length)
    if gt_future_frames_batch.shape[1] < max_length:
        gt_future_frames_batch = append_frames(gt_future_frames_batch, max_length)
    if pred_frames_batch.shape[1] < max_length:    
        pred_frames_batch = append_frames(pred_frames_batch, max_length)

    batch = torch.cat([gt_past_frames_batch, gt_future_frames_batch, pred_frames_batch], dim = -1) #shape (N, clip_length, C, H, 3W)
    batch = batch.cpu()
    N = batch.shape[0]
    for n in range(N):
        clip = batch[n, ...]
        file_name = file_dir.joinpath(f'{desc}_clip_{n}.gif')
        save_clip(clip, file_name)