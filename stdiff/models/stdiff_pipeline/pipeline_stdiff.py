# Copyright 2023 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from typing import List, Optional, Tuple, Union

import torch
import torch
import torch.nn.functional as F
from torch import Tensor

from diffusers import utils
from diffusers import DiffusionPipeline, ImagePipelineOutput
import torchvision.transforms as transforms
from math import exp

from einops import rearrange

class STDiffPipeline(DiffusionPipeline):
    """
    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)

    Parameters:
        unet ([`UNet2DModel`]): U-Net architecture to denoise the encoded image.
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `unet` to denoise the encoded image. Can be one of
            [`DDPMScheduler`], or [`DDIMScheduler`].
    """

    def __init__(self, stdiff, scheduler):
        super().__init__()
        self.register_modules(stdiff=stdiff, scheduler=scheduler)
        
    @torch.no_grad()
    def __call__(
        self,
        Vo,
        Vo_last_frame,
        idx_o,
        idx_p,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        num_inference_steps: int = 1000,
        output_type: Optional[str] = "pil",
        to_cpu=True,
        fix_init_noise=None
    ) -> Union[ImagePipelineOutput, Tuple]:
        r"""
        Args:
            batch_size (`int`, *optional*, defaults to 1):
                The number of images to generate.
            generator (`torch.Generator`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            num_inference_steps (`int`, *optional*, defaults to 1000):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.ImagePipelineOutput`] instead of a plain tuple.

        Returns:
            [`~pipelines.ImagePipelineOutput`] or `tuple`: [`~pipelines.utils.ImagePipelineOutput`] if `return_dict` is
            True, otherwise a `tuple. When returning a tuple, the first element is a list with the generated images.
        """
        #set default value for fix_init_noise
        if fix_init_noise is None:
            if not self.stdiff.autoreg:
                fix_init_noise = True
            else:
                fix_init_noise = False

        if not self.stdiff.autoreg:
            # Sample gaussian noise to begin loop
            if fix_init_noise:
                image_shape = (Vo.shape[0], self.stdiff.diffusion_unet.in_channels, self.stdiff.diffusion_unet.sample_size, self.stdiff.diffusion_unet.sample_size)
            else:
                batch_size = Vo.shape[0]*idx_p.shape[0]
                image_shape = (batch_size, self.stdiff.diffusion_unet.in_channels, self.stdiff.diffusion_unet.sample_size, self.stdiff.diffusion_unet.sample_size)
                
            image = self.init_noise(image_shape, generator)
            if fix_init_noise:
                image = image.unsqueeze(1).repeat(1, idx_p.shape[0], 1, 1, 1).flatten(0, 1)
            # set step values
            self.scheduler.set_timesteps(num_inference_steps)

            # manually extract the future motion feature
            #vo: (N, To, C, H, W), idx_o: (To, ), idx_p: (Tp, ), noisy_Vp: (N*Tp, C, H, W)
            m_context = self.stdiff.tde_model.context_encode(Vo, idx_o) #(N, C, H, W)
                
            #use ode/sde to predict the future motion features
            m_future = self.stdiff.tde_model.future_predict(m_context, torch.cat([idx_o[-1:], idx_p])) #(Tp, N, C, H, W)

            for t in self.progress_bar(self.scheduler.timesteps):
                # 1. predict noise model_output
                model_output = self.stdiff.diffusion_unet(image, t, m_feat = m_future.permute(1, 0, 2, 3, 4).flatten(0, 1)).sample

                # 2. compute previous image: x_t -> x_t-1
                #image = self.scheduler.step(model_output, t, image, generator=generator).prev_sample
                image = self.scheduler.step(model_output, t, image).prev_sample
        
        else:
            # Sample gaussian noise to begin loop
            image_shape = (Vo.shape[0], self.stdiff.diffusion_unet.out_channels, self.stdiff.diffusion_unet.sample_size, self.stdiff.diffusion_unet.sample_size)
            
            # set step values
            self.scheduler.set_timesteps(num_inference_steps)

            # manually extract the future motion feature
            #vo: (N, To, C, H, W), idx_o: (To, ), idx_p: (Tp, ), noisy_Vp: (N*Tp, C, H, W)
            m_context = self.stdiff.tde_model.context_encode(Vo, idx_o) #(N, C, H, W)

            m_future = self.stdiff.tde_model.future_predict(m_context, torch.cat([idx_o[-1:], idx_p])) #(Tp, N, C, H, W)

            #for the super resolution training
            Ho, Wo = Vo.shape[3], Vo.shape[4]
            Hp, Wp = image_shape[2], image_shape[3]
            down_sample = lambda x: x
            up_sample = lambda x: x
            if self.stdiff.super_res_training:
                if Ho < Hp or Wo < Wp:
                    down_sample= transforms.Resize((Ho, Wo), interpolation=transforms.InterpolationMode.BICUBIC, antialias=True)
                    up_sample = transforms.Resize((Hp, Wp), interpolation=transforms.InterpolationMode.BICUBIC, antialias=True)
            
            image = self.init_noise(image_shape, generator)
            imgs = []
            for tp in range(idx_p.shape[0]):
                if tp == 0:
                    if self.stdiff.super_res_training:
                        Vo_last_frame = up_sample(Vo[:, -1, ...])
                        prev_frame = Vo_last_frame
                    else:
                        prev_frame = Vo_last_frame[:, -1, ...]
                else:
                    if self.stdiff.super_res_training:
                        prev_frame = up_sample(down_sample(imgs[-1]))
                    else:
                        prev_frame = imgs[-1]
                
                if not fix_init_noise:
                    image = self.init_noise(image_shape, generator)
                
                for t in self.progress_bar(self.scheduler.timesteps):
                    # 1. predict noise model_output
                    model_output = self.stdiff.diffusion_unet(torch.cat([image, prev_frame.clamp(-1, 1)], dim = 1), t, m_feat = m_future[tp, ...]).sample

                    # 2. compute previous image: x_t -> x_t-1
                    #image = self.scheduler.step(model_output, t, image, generator=generator).prev_sample
                    image = self.scheduler.step(model_output, t, image).prev_sample

                imgs.append(image)

            image = torch.stack(imgs, dim = 1).flatten(0, 1)
            
        image = (image / 2 + 0.5).clamp(0, 1)
        if output_type == "numpy":
            image = image.cpu().permute(0, 2, 3, 1).numpy()
            return ImagePipelineOutput(images=image)
        else:
            image = rearrange(image, '(N T) C H W -> N T C H W', N = Vo.shape[0], T = idx_p.shape[0])
            if to_cpu:
                image = image.cpu()
            return image
    
    def init_noise(self, image_shape, generator):
        if self.device.type == "mps":
            # randn does not work reproducibly on mps
            image = utils.randn_tensor(image_shape, generator=generator)
            image = image.to(self.device)
        else:
            image = utils.randn_tensor(image_shape, generator=generator, device=self.device)
        
        return image
    
    def disable_pgbar(self):
        self.progress_bar = lambda x: x