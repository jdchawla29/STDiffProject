
from utils import LitDataModule

from utils import get_lightning_module_dataloader
import torchvision.transforms as transforms
from torchvision.utils import make_grid

import torch
import torch.nn as nn

import hydra
from hydra import compose, initialize
from omegaconf import DictConfig, omegaconf
from einops import rearrange

from utils import visualize_batch_clips
from pathlib import Path
import argparse
from models import STDiffPipeline, STDiffDiffusers
from diffusers import DDPMScheduler, DPMSolverMultistepScheduler, PNDMScheduler, DDIMScheduler
from accelerate import Accelerator
from tqdm.auto import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description=globals()['__doc__'])
    parser.add_argument('--test_config', type=str, required=True, help='Path to configuration file')
    args = parser.parse_args()
    return args.test_config

def main(cfg : DictConfig) -> None:
    accelerator = Accelerator()
    device = accelerator.device
    ckpt_path = cfg.TestCfg.ckpt_path
    r_save_path = cfg.TestCfg.test_results_path
    if not Path(r_save_path).exists():
        Path(r_save_path).mkdir(parents=True, exist_ok=True) 

    #load stdiff model
    stdiff = STDiffDiffusers.from_pretrained(ckpt_path, subfolder='stdiff').eval()
    #Print the number of parameters
    num_params = sum(p.numel() for p in stdiff.parameters() if p.requires_grad)
    print('Number of parameters is: ', num_params)

    #init scheduler
    scheduler = DDPMScheduler.from_pretrained(ckpt_path, subfolder = 'scheduler')

    stdiff_pipeline = STDiffPipeline(stdiff, scheduler).to(device)
    if not accelerator.is_main_process:
        stdiff_pipeline.disable_pgbar()
    _, test_loader = get_lightning_module_dataloader(cfg)
    stdiff_pipeline, test_loader = accelerator.prepare(stdiff_pipeline, test_loader)

    To = 11
    Tp = 11
    idx_o = torch.linspace(0, 10 , 11).to(device)
    idx_p = torch.linspace(11, 21, 11).to(device)

    if accelerator.is_main_process:
        print('idx_o', idx_o)
        print('idx_p', idx_p)
        test_config = {'cfg': cfg, 'idx_o': idx_o.to('cpu'), 'idx_p': idx_p.to('cpu')}
        torch.save(test_config, f = Path(r_save_path).joinpath('TestConfig.pt'))
    
    def get_resume_batch_idx(r_save_path):
        save_path = Path(r_save_path)
        saved_preds = sorted(list(save_path.glob('Preds_*')))
        saved_batches = sorted([int(str(p.name).split('_')[1].split('.')[0]) for p in saved_preds])
        try:
            return saved_batches[-1]
        except IndexError:
            return -1
    resume_batch_idx = get_resume_batch_idx(r_save_path)
    print('number of test batches: ', len(test_loader))
    print('resume batch index: ', resume_batch_idx)

    #Predict and save the predictions to disk for evaluation
    with torch.no_grad():
        progress_bar = tqdm(total=len(test_loader), disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Testing...") 
        for idx, batch in enumerate(test_loader):
            if idx > resume_batch_idx: #resume test
                Vo, Vp, Vo_last_frame, _, _ = batch

                filter_first_out = stdiff_pipeline.filter_best_first_pred(10, Vo.clone(), 
                                                                            Vo_last_frame, Vp[:, 0:1, ...], idx_o, idx_p, 
                                                                            num_inference_steps = 100,
                                                                            fix_init_noise=False,
                                                                            bs = 4)
                Vo_input = Vo.clone()
                pred_clip = stdiff_pipeline.pred_remainig_frames(*(filter_first_out + (False, "pil", False)))
                Vo_input = pred_clip[:, -To:, ...]*2. - 1.
                Vo_last_frame = pred_clip[:, -1:, ...]*2. -1.

                Vo = (Vo / 2 + 0.5).clamp(0, 1)
                Vp = (Vp / 2 + 0.5).clamp(0, 1)

                g_pred = accelerator.gather(pred_clip)
                g_Vo = accelerator.gather(Vo)
                g_Vp = accelerator.gather(Vp)

                if accelerator.is_main_process:
                    dump_obj = {'Vo': g_Vo.detach().cpu(), 'g_Vp': g_Vp.detach().cpu(), 'g_Pred': g_pred.detach().cpu()}
                    torch.save(dump_obj, f=Path(r_save_path).joinpath(f'Pred_{idx}.pt'))
                    progress_bar.update(1)
                    visualize_batch_clips(Vo, Vp, pred_clip, file_dir=Path(r_save_path),idx=idx)

                    del g_Vo
                    del g_Vp
                    del g_pred
    print("Inference finished")
if __name__ == '__main__':
    config_path = Path(parse_args())
    initialize(version_base=None, config_path=str(config_path.parent))
    cfg = compose(config_name=str(config_path.name))

    main(cfg)