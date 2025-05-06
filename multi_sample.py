import argparse
import os
import random
import torch
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm
from einops import rearrange
from torchvision.utils import make_grid
from multiprocessing import Process, Queue, set_start_method

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler


torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def load_model_from_config(config, ckpt, device, verbose=False):
    print(f"Loading model from {ckpt} on device {device}")
    pl_sd = torch.load(ckpt, map_location="cpu", weights_only=False)
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)
    model.to(device)
    model.eval()
    return model

def generate_samples(rank, gpu_id, opt, prompts_slice, quantize):
    device = torch.device(f"cuda:{gpu_id}")
    torch.cuda.set_device(device)

    config = OmegaConf.load(opt.cfg)
    model = load_model_from_config(config, opt.ckpt, device=device)

    if opt.plms:
        sampler = PLMSSampler(model)
    else:
        sampler = DDIMSampler(model)

    outpath = opt.outdir
    sample_path = os.path.join(outpath, "samples")
    os.makedirs(sample_path, exist_ok=True)

    if opt.seed is not None:
        torch.manual_seed(opt.seed + rank)
        torch.cuda.manual_seed_all(opt.seed + rank)
        random.seed(opt.seed + rank)
        np.random.seed(opt.seed + rank)

    base_count = rank * len(prompts_slice)  

    with torch.no_grad():
        with model.ema_scope():
            uc = None
            if opt.scale != 1.0:
                uc = model.get_learned_conditioning(opt.batch_size_per_gpu * [""])

            for n in tqdm(range(opt.n_iter), desc=f"Sampling on GPU {gpu_id}"):
                for i in range(0, len(prompts_slice), opt.batch_size_per_gpu):
                    current_prompts = prompts_slice[i:i + opt.batch_size_per_gpu]
                    actual_batch_size = len(current_prompts)

                    c = model.get_learned_conditioning(current_prompts)
                    if opt.sd:
                        shape = [4, opt.H // 8, opt.W // 8]
                    else:
                        shape = [3, opt.H // 16, opt.W // 16]
                    samples_ddim, _ = sampler.sample(
                        S=opt.ddim_steps,
                        conditioning=c,
                        batch_size=actual_batch_size,
                        shape=shape,
                        verbose=False,
                        unconditional_guidance_scale=opt.scale,
                        unconditional_conditioning=uc[:actual_batch_size] if uc is not None else None,
                        eta=opt.ddim_eta,
                        quantize_x0=quantize,
                    )

                    x_samples_ddim = model.decode_first_stage(samples_ddim)
                    x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)

                    for j, x_sample in enumerate(x_samples_ddim):
                        x_sample = 255.0 * rearrange(x_sample.cpu().numpy(), "c h w -> h w c")
                        Image.fromarray(x_sample.astype(np.uint8)).save(
                            os.path.join(sample_path, f"{base_count + i + j:06}.png")
                        )
            print(f"GPU {gpu_id} finished processing.")

if __name__ == "__main__":
    try:
        set_start_method('spawn')
    except RuntimeError:
        pass

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--prompt",
        type=str,
        nargs="?",
        required=True,
        help="the prompt to render",
    )

    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        help="dir to write results to",
        required=True,
    )
    parser.add_argument(
        "--ddim_steps",
        type=int,
        default=200,
        help="number of ddim sampling steps",
    )

    parser.add_argument(
        "--plms",
        action="store_true",
        help="use plms sampling",
    )

    parser.add_argument(
        "--ddim_eta",
        type=float,
        default=0.0,
        help="ddim eta (eta=0.0 corresponds to deterministic sampling)",
    )
    parser.add_argument(
        "--n_iter",
        type=int,
        default=1,
        help="sample this often",
    )

    parser.add_argument(
        "--H",
        type=int,
        default=512,
        help="image height, in pixel space",
    )

    parser.add_argument(
        "--W",
        type=int,
        default=1024,
        help="image width, in pixel space",
    )

    parser.add_argument(
        "--batch_size_per_gpu",
        type=int,
        default=1,
        help="batch size per GPU",
    )

    parser.add_argument(
        "--scale",
        type=float,
        default=5.0,
        help="unconditional guidance scale",
    )
    parser.add_argument(
        "--cfg",
        type=str,
        required=True,
        help="path to model config",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        required=True,
        help="path to model checkpoint",
    )
    parser.add_argument(
        "--sd",
        action="store_true",
        help="flag for sd2 model",
    )
    parser.add_argument(
        "--quantize",
        action="store_true",
        help="flag for quantize",
    )
    parser.add_argument(
        "--gpus",
        type=str,
        default="0",
        help="comma-separated list of GPU ids to use, e.g., '0,1,2,3'",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="random seed",
    )
    opt = parser.parse_args()

    if os.path.isfile(opt.prompt):
        with open(opt.prompt, "r") as f:
            prompts = f.read().splitlines()
    else:
        prompts = [opt.prompt]

    assert len(prompts) > 0, "Please provide a prompt to generate samples from."

    os.makedirs(opt.outdir, exist_ok=True)
    outpath = opt.outdir

    gpu_list = [int(x) for x in opt.gpus.split(',')]
    num_gpus = len(gpu_list)

    total_prompts = len(prompts)
    prompts_per_gpu = total_prompts // num_gpus
    remainder = total_prompts % num_gpus

    processes = []

    start = 0
    for rank, gpu_id in enumerate(gpu_list):
        end = start + prompts_per_gpu
        if rank < remainder:
            end += 1
        prompts_slice = prompts[start:end]
        start = end

        p = Process(target=generate_samples, args=(rank, gpu_id, opt, prompts_slice, opt.quantize))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    print(f"Your samples are ready and waiting for you here: \n{outpath} \nEnjoy.")
