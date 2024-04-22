
from utils import get_lightning_module_dataloader

import torch

from hydra import compose, initialize
from omegaconf import DictConfig

from pathlib import Path
import argparse
from models import STDiffPipeline, STDiffDiffusers
from diffusers import DDPMScheduler
from accelerate import Accelerator
from tqdm.auto import tqdm
from PIL import Image

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

    # Load stdiff model
    stdiff = STDiffDiffusers.from_pretrained(ckpt_path, subfolder='stdiff').eval()

    # Print the number of parameters
    num_params = sum(p.numel() for p in stdiff.parameters() if p.requires_grad)
    print('Number of parameters is: ', num_params)

    # Init scheduler
    scheduler = DDPMScheduler.from_pretrained(ckpt_path, subfolder = 'scheduler')

    stdiff_pipeline = STDiffPipeline(stdiff, scheduler).to(device)
    
    if not accelerator.is_main_process:
        stdiff_pipeline.disable_pgbar()

    _, test_loader = get_lightning_module_dataloader(cfg)
    stdiff_pipeline, test_loader = accelerator.prepare(stdiff_pipeline, test_loader)

    To = cfg.Dataset.num_observed_frames
    Tp = cfg.Dataset.num_predict_frames

    idx_o = torch.linspace(0, To-1, To).to(device)
    idx_p = torch.linspace(To, Tp+To-1, Tp).to(device)
    
    def get_resume_batch_idx(r_save_path):
        save_path = Path(r_save_path)
        saved_preds = sorted(list(save_path.glob('*_0_0.png')))
        saved_batches = [int(str(p.name).split('_')[1]) for p in saved_preds]
        try:
            return saved_batches[-1]
        except IndexError:
            return -1
    resume_batch_idx = get_resume_batch_idx(r_save_path)

    print('number of test batches: ', len(test_loader))
    print('resume batch index: ', resume_batch_idx)

    with torch.no_grad():
        progress_bar = tqdm(total=len(test_loader))
        progress_bar.set_description(f"Testing...") 

        for idx, batch in enumerate(test_loader):
            if idx > resume_batch_idx: # Resume test
                Vo, _, Vo_last_frame, _, _ = batch

                images = stdiff_pipeline(
                    Vo,
                    Vo_last_frame,
                    idx_o,
                    idx_p,
                    num_inference_steps=100,
                    output_type="numpy"
                ).images

                images_processed = (images * 255).round().astype("uint8")

                for i in range(images_processed.shape[0]):
                    img = Image.fromarray(images_processed[i])
                    if (i+1) % Tp == 0:
                        img.save(Path(r_save_path).joinpath(f'Pred_{idx}_{i//Tp}.png'))
                progress_bar.update(1)
        
        progress_bar.close()
                
    print("Inference finished")
    
if __name__ == '__main__':
    config_path = Path(parse_args())
    initialize(version_base=None, config_path=str(config_path.parent))
    cfg = compose(config_name=str(config_path.name))

    main(cfg)