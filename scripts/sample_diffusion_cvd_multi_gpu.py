import argparse, os, sys, glob, datetime, yaml
import torch
import time
import numpy as np

import re
import tqdm
from omegaconf import OmegaConf
from PIL import Image
from torch_utils import distributed as dist

from ldm.models.diffusion.ddim_cvd_guided import DDIMCVDSampler
from ldm.models.diffusion.ddim_cvd_doodl_only import DDIMCVDDOODLSampler
from ldm.util import instantiate_from_config
import pickle

rescale = lambda x: (x + 1.) / 2.


# ----------------------------------------------------------------------------

class StackedRandomGenerator:
    def __init__(self, device, seeds):
        super().__init__()
        self.generators = [torch.Generator(device).manual_seed(int(seed) % (1 << 32)) for seed in seeds]

    def randn(self, size, **kwargs):
        assert size[0] == len(self.generators)
        return torch.stack([torch.randn(size[1:], generator=gen, **kwargs) for gen in self.generators])

    def randn_like(self, input):
        return self.randn(input.shape, dtype=input.dtype, layout=input.layout, device=input.device)

    def randint(self, *args, size, **kwargs):
        assert size[0] == len(self.generators)
        return torch.stack([torch.randint(*args, size=size[1:], generator=gen, **kwargs) for gen in self.generators])


# ----------------------------------------------------------------------------


def custom_to_pil(x):
    x = x.detach().cpu()
    x = torch.clamp(x, -1., 1.)
    x = (x + 1.) / 2.
    x = x.permute(1, 2, 0).numpy()
    x = (255 * x).astype(np.uint8)
    x = Image.fromarray(x)
    if not x.mode == "RGB":
        x = x.convert("RGB")
    return x


def prepare_output(x):
    # saves the batch in adm style as in https://github.com/openai/guided-diffusion/blob/main/scripts/image_sample.py
    sample = x.detach().cpu()
    sample = ((sample + 1) * 127.5).clamp(0, 255).to(torch.uint8)
    sample = sample.permute(0, 2, 3, 1)
    sample = sample.contiguous()
    return sample


def logs2pil(logs, keys=["sample"]):
    imgs = dict()
    for k in logs:
        try:
            if len(logs[k].shape) == 4:
                img = custom_to_pil(logs[k][0, ...])
            elif len(logs[k].shape) == 3:
                img = custom_to_pil(logs[k])
            else:
                dist.print0(f"Unknown format for key {k}. ")
                img = None
        except:
            img = None
        imgs[k] = img
    return imgs


@torch.no_grad()
def convsample(model, shape, return_intermediates=True,
               verbose=True,
               make_prog_row=False):
    if not make_prog_row:
        return model.p_sample_loop(None, shape,
                                   return_intermediates=return_intermediates, verbose=verbose)
    else:
        return model.progressive_denoising(
            None, shape, verbose=True
        )


@torch.no_grad()
def convsample_ddim(model, steps, shape, eta=1.0, cvd_args=None, rnd=None, initial_noise_path=None):
    if cvd_args['run_doodl']:
        ddim_cvd = DDIMCVDDOODLSampler(model, cvd_args=cvd_args, rnd=rnd)
    else:
        network_pkl = ""
        if network_pkl != "":
            with open(network_pkl, 'rb') as f:
                D = pickle.load(f)['D']  # torch.nn.Module
        else:
            D = None
        ddim_cvd = DDIMCVDSampler(model, cvd_args=cvd_args, rnd=rnd, D=D)

    bs = shape[0]
    shape = shape[1:]
    samples, intermediates = ddim_cvd.sample(steps, batch_size=bs, shape=shape, eta=eta, verbose=False, initial_noise_path=initial_noise_path)
    return samples, intermediates


@torch.no_grad()
def make_convolutional_sample(model, batch_size, cvd_args=None, vanilla=False, custom_steps=None, eta=1.0, rnd=None, initial_noise_path=None):
    log = dict()

    shape = [batch_size,
             model.model.diffusion_model.in_channels,
             model.model.diffusion_model.image_size,
             model.model.diffusion_model.image_size]

    with model.ema_scope("Plotting"):
        t0 = time.time()
        if vanilla:
            sample, progrow = convsample(model, shape,
                                         make_prog_row=True)
        else:
            sample, intermediates = convsample_ddim(model, cvd_args=cvd_args, steps=custom_steps, shape=shape,
                                                    eta=eta, rnd=rnd, initial_noise_path=initial_noise_path)

        t1 = time.time()

    x_sample = model.decode_first_stage(sample)

    log["sample"] = x_sample
    log["time"] = t1 - t0
    log['throughput'] = sample.shape[0] / (t1 - t0)
    dist.print0(f'Throughput for this batch: {log["throughput"]}')
    return log


def run(model, logdir, cvd_args=None, vanilla=False, custom_steps=None, eta=None, n_samples=50000,
        nplog=None, dist=None, rank_batches=None, device=torch.device('cuda'), initial_noise_path=None):
    if vanilla:
        dist.print0(f'Using Vanilla DDPM sampling with {model.num_timesteps} sampling steps.')
    else:
        dist.print0(f'Using DDIM sampling with {custom_steps} sampling steps and eta={eta}')

    tstart = time.time()
    n_saved = len(glob.glob(os.path.join(logdir, '*.png'))) - 1
    # path = logdir
    if model.cond_stage_model is None:
        all_images = []

        dist.print0(f"Running unconditional sampling for {n_samples} samples")
        for batch_seeds in tqdm.tqdm(rank_batches, unit='batch', disable=(dist.get_rank() != 0)):
            torch.distributed.barrier()
            batch_size = len(batch_seeds)
            # print(f"dist rank: {dist.get_rank()}: batch of {batch_seeds}")

            if batch_size == 0:
                continue

            rnd = StackedRandomGenerator(device, batch_seeds)
            logs = make_convolutional_sample(model, cvd_args=cvd_args, batch_size=batch_size,
                                             vanilla=vanilla, custom_steps=custom_steps,
                                             eta=eta, rnd=rnd, initial_noise_path=initial_noise_path)

            if cvd_args['init_latent_load_offset'] >= 0:
                cvd_args['init_latent_load_offset'] += batch_size

            n_saved = save_logs(logs, logdir, n_saved=n_saved, key="sample", dist=dist)
            all_images.extend([prepare_output(logs["sample"])])

            if n_saved >= n_samples:
                dist.print0(f'Finish after generating {n_saved} samples')
                break

        all_img = np.concatenate(all_images, axis=0)
        all_img = all_img[:n_samples]
        shape_str = "x".join([str(x) for x in all_img.shape])
        nppath = os.path.join(nplog, f"{shape_str}-samples.npz")
        np.savez(nppath, all_img)

    else:
        raise NotImplementedError('Currently only sampling for unconditional models supported.')

    dist.print0(f"sampling of {n_saved} images finished in {(time.time() - tstart) / 60.:.2f} minutes.")


def save_logs(logs, path, n_saved=0, key="sample", np_path=None, dist=None):
    for k in logs:
        if k == key:
            batch = logs[key]
            if np_path is None:
                for x in batch:
                    img = custom_to_pil(x)
                    imgpath = os.path.join(path, f"{key}_{n_saved:06}_{dist.get_rank()}.png")
                    img.save(imgpath)
                    n_saved += 1
            else:
                npbatch = prepare_output(batch)
                shape_str = "x".join([str(x) for x in npbatch.shape])
                nppath = os.path.join(np_path, f"{n_saved}-{shape_str}-samples.npz")
                np.savez(nppath, npbatch)
                n_saved += npbatch.shape[0]
    return n_saved


def get_guidance_args(opt):
    if opt.cvd_degree < 0 or opt.cvd_degree > 1:
        raise ValueError("cvd_degree should be in the range [0, 1]")

    cvd_args = {"cvd_degree": opt.cvd_degree,
                "cvd_type": opt.cvd_type,
                "cvd_alpha": opt.cvd_alpha,
                "gs": opt.gs,
                "backward_iters": opt.backward_iters,
                "backward_lr": opt.backward_lr,
                "init_latent_load_offset": opt.init_latent_load_offset,
                "run_doodl": opt.run_doodl}

    return cvd_args


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-r",
        "--ckpt",
        type=str,
        nargs="?",
        help="load from logdir or checkpoint in logdir",
    )
    parser.add_argument(
        "-n",
        "--n_samples",
        type=int,
        nargs="?",
        help="number of samples to draw - no need to specify for cvd guided sampling",
        default=50000
    )
    parser.add_argument(
        "-e",
        "--eta",
        type=float,
        nargs="?",
        help="eta for ddim sampling (0.0 yields deterministic sampling)",
        default=0.0
    )
    parser.add_argument(
        "-v",
        "--vanilla_sample",
        default=False,
        action='store_true',
        help="vanilla sampling (default option is DDIM sampling)?",
    )
    parser.add_argument(
        "-l",
        "--logdir",
        type=str,
        nargs="?",
        help="extra logdir, where outputs are stored",
        default="none"
    )
    parser.add_argument(
        "-c",
        "--custom_steps",
        type=int,
        nargs="?",
        help="number of steps for ddim and fastdpm sampling",
        default=50
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        nargs="?",
        help="the bs",
        default=1
    )
    parser.add_argument(
        "--seeds",
        type=parse_int_list,
        default='0-479',
        help="the seeds (for reproducible sampling)",
    )

    # ==================================================
    # CVD Guidance Arguments
    # ==================================================

    parser.add_argument(
        "--cvd_degree",
        type=float,
        default=0.0,
        help="the degree (severity) of color vision deficiency",
    )
    parser.add_argument(
        "--cvd_type",
        type=str,
        default="PROTAN",
        choices=["PROTAN", "DEUTAN"],
        help="the type of color vision deficiency, either PROTAN or DEUTAN",
    )
    parser.add_argument(
        "--cvd_alpha",
        type=float,
        default=0.3,
        help="CVD interpolation weight for the color information loss component",
    )
    parser.add_argument(
        "--gs",
        type=int,
        default=70,
        help="cvd guidance scale",
    )
    parser.add_argument(
        "--backward_iters",
        type=int,
        default=0,
        help="No. of iterations for backward guidance",
    )
    parser.add_argument(
        "--backward_lr",
        type=float,
        default=2e-4,
        help="Learning rate for backward guidance",
    )
    parser.add_argument(
        "--init_latent_load_offset",
        type=int,
        default=-1,
        help="Accumulative offset for loading latents from the corresponding pt files. Default -1 means do not load from existing pt files",
    )
    parser.add_argument(
        "--run_doodl",
        type=bool,
        default=False,
        help="Whether run doodl, batch size will be forced to 1 if True",
    )
    parser.add_argument(
        "--initial_noise_path",
        type=str,
        help="path to the initial latent noise for ddim sampling, should be a .pt file",
    )
    return parser


def load_model_from_config(config, sd):
    model = instantiate_from_config(config)
    model.load_state_dict(sd, strict=False)
    model.to(torch.device('cuda'))
    model.eval()
    return model


def load_model(config, ckpt):
    if ckpt:
        dist.print0(f"Loading model from {ckpt}")
        pl_sd = torch.load(ckpt, map_location="cpu")
        global_step = pl_sd["global_step"]
    else:
        pl_sd = {"state_dict": None}
        global_step = None
    model = load_model_from_config(config.model,
                                   pl_sd["state_dict"])

    return model, global_step


def sanity_check_and_setup(opt):
    if not os.path.exists(opt.ckpt):
        raise ValueError("Cannot find {}".format(opt.ckpt))

    if not os.path.exists(opt.ckpt):
        raise ValueError("Cannot find {}".format(opt.ckpt))

    if opt.run_doodl:
        opt.batch_size = 1

    return opt


def parse_int_list(s):
    if isinstance(s, list): return s
    ranges = []
    range_re = re.compile(r'^(\d+)-(\d+)$')
    for p in s.split(','):
        m = range_re.match(p)
        if m:
            ranges.extend(range(int(m.group(1)), int(m.group(2)) + 1))
        else:
            ranges.append(int(p))
    return ranges


def main():
    now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    sys.path.append(os.getcwd())

    parser = get_parser()
    opt, unknown = parser.parse_known_args()

    opt = sanity_check_and_setup(opt)

    dist.init()
    num_batches = ((len(opt.seeds) - 1) // (opt.batch_size * dist.get_world_size()) + 1) * dist.get_world_size()
    all_batches = torch.as_tensor(opt.seeds).tensor_split(num_batches)
    rank_batches = all_batches[dist.get_rank():: dist.get_world_size()]

    # Rank 0 goes first.
    if dist.get_rank() != 0:
        torch.distributed.barrier()

    if os.path.isfile(opt.ckpt):
        # paths = opt.ckpt.split("/")
        try:
            logdir = '/'.join(opt.ckpt.split('/')[:-1])
            # idx = len(paths)-paths[::-1].index("logs")+1
            dist.print0(f'Logdir is {logdir}')
        except ValueError:
            paths = opt.ckpt.split("/")
            idx = -2  # take a guess: path/to/logdir/checkpoints/model.ckpt
            logdir = "/".join(paths[:idx])
        ckpt = opt.ckpt
    else:
        assert os.path.isdir(opt.ckpt), f"{opt.ckpt} is not a directory"
        logdir = opt.ckpt.rstrip("/")
        ckpt = os.path.join(logdir, "model.ckpt")

    base_configs = sorted(glob.glob(os.path.join(logdir, "config.yaml")))
    opt.base = base_configs

    configs = [OmegaConf.load(cfg) for cfg in opt.base]
    cli = OmegaConf.from_dotlist(unknown)
    config = OmegaConf.merge(*configs, cli)

    if opt.logdir != "none":
        locallog = logdir.split(os.sep)[-1]
        if locallog == "": locallog = logdir.split(os.sep)[-2]
        dist.print0(f"Switching logdir from '{logdir}' to '{os.path.join(opt.logdir, locallog)}'")
        logdir = os.path.join(opt.logdir, locallog)

    dist.print0(config)

    model, global_step = load_model(config, ckpt)

    dist.print0(f"global step: {global_step}")

    dist.print0(75 * "=")
    dist.print0("logging to:")
    logdir = os.path.join(logdir, "samples", f"{global_step:08}", now)
    imglogdir = os.path.join(logdir, "img")
    numpylogdir = os.path.join(logdir, "numpy")

    if dist.get_rank() == 0:
        os.makedirs(imglogdir)
        os.makedirs(numpylogdir)

    torch.distributed.barrier()

    dist.print0(logdir)
    dist.print0(75 * "=")

    cvd_args = get_guidance_args(opt)

    # write config out
    sampling_file = os.path.join(logdir, "sampling_config.yaml")
    sampling_conf = vars(opt)

    if dist.get_rank() == 0:
        with open(sampling_file, 'w') as f:
            yaml.dump(sampling_conf, f, default_flow_style=False)
        dist.print0(sampling_conf)

    torch.distributed.barrier()

    run(model, imglogdir, cvd_args=cvd_args, eta=opt.eta,
        vanilla=opt.vanilla_sample, n_samples=opt.n_samples, custom_steps=opt.custom_steps,
        nplog=numpylogdir, dist=dist, rank_batches=rank_batches, initial_noise_path=opt.initial_noise_path)

    if dist.get_rank() == 0:
        torch.distributed.barrier()

    dist.print0("done.")


if __name__ == "__main__":
    main()
