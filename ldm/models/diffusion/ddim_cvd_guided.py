"""SAMPLING ONLY."""

import torch
import numpy as np
from tqdm import tqdm
from torch.optim import Adam, AdamW
from PIL import Image

from ldm.modules.diffusionmodules.util import make_ddim_sampling_parameters, make_ddim_timesteps, noise_like
from ldm.modules.losses.cvd_loss import *
from ldm.modules.doodl.helper_functions import *
import os
import re
from torch_utils import misc
import PIL
import kornia.color as kcolor


class DDIMCVDSampler(object):
    def __init__(self, model, cvd_args, schedule="linear", rnd=None, D=None):
        super().__init__()
        self.model = model
        self.ddpm_num_timesteps = model.num_timesteps
        self.schedule = schedule
        self.cvdkwargs = cvd_args
        self.rnd = rnd
        self.iter = 0

        if D is not None:
            self.D = D.to(self.model.device)
        else:
            self.D = None

    def register_buffer(self, name, attr):
        if type(attr) == torch.Tensor:
            if attr.device != torch.device("cuda"):
                attr = attr.to(torch.device("cuda"))
        setattr(self, name, attr)

    def make_schedule(self, ddim_num_steps, ddim_discretize="uniform", ddim_eta=0., verbose=True):
        self.ddim_timesteps = make_ddim_timesteps(ddim_discr_method=ddim_discretize, num_ddim_timesteps=ddim_num_steps,
                                                  num_ddpm_timesteps=self.ddpm_num_timesteps, verbose=verbose)
        alphas_cumprod = self.model.alphas_cumprod
        assert alphas_cumprod.shape[0] == self.ddpm_num_timesteps, 'alphas have to be defined for each timestep'
        to_torch = lambda x: x.clone().detach().to(torch.float32).to(self.model.device)

        self.register_buffer('betas', to_torch(self.model.betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(self.model.alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod.cpu())))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod.cpu())))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod.cpu())))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod.cpu())))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod.cpu() - 1)))

        # ddim sampling parameters
        ddim_sigmas, ddim_alphas, ddim_alphas_prev = make_ddim_sampling_parameters(alphacums=alphas_cumprod.cpu(),
                                                                                   ddim_timesteps=self.ddim_timesteps,
                                                                                   eta=ddim_eta, verbose=verbose)
        self.register_buffer('ddim_sigmas', ddim_sigmas)
        self.register_buffer('ddim_alphas', ddim_alphas)
        self.register_buffer('ddim_alphas_prev', ddim_alphas_prev)
        self.register_buffer('ddim_sqrt_one_minus_alphas', np.sqrt(1. - ddim_alphas))
        sigmas_for_original_sampling_steps = ddim_eta * torch.sqrt(
            (1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod) * (
                    1 - self.alphas_cumprod / self.alphas_cumprod_prev))
        self.register_buffer('ddim_sigmas_for_original_num_steps', sigmas_for_original_sampling_steps)

    def sample(self,
               S,
               batch_size,
               shape,
               conditioning=None,
               callback=None,
               img_callback=None,
               quantize_x0=False,
               eta=0.,
               mask=None,
               x0=None,
               temperature=1.,
               noise_dropout=0.,
               score_corrector=None,
               corrector_kwargs=None,
               verbose=True,
               x_T=None,
               log_every_t=100,
               unconditional_guidance_scale=1.,
               unconditional_conditioning=None,
               initial_noise_path=None,
               # this has to come in the same format as the conditioning, # e.g. as encoded tokens, ...
               **kwargs
               ):
        if conditioning is not None:
            if isinstance(conditioning, dict):
                cbs = conditioning[list(conditioning.keys())[0]].shape[0]
                if cbs != batch_size:
                    print(f"Warning: Got {cbs} conditionings but batch-size is {batch_size}")
            else:
                if conditioning.shape[0] != batch_size:
                    print(f"Warning: Got {conditioning.shape[0]} conditionings but batch-size is {batch_size}")

        self.make_schedule(ddim_num_steps=S, ddim_eta=eta, verbose=verbose)
        # sampling
        C, H, W = shape
        size = (batch_size, C, H, W)
        print(f'Data shape for DDIM sampling is {size}, eta {eta}')

        samples, intermediates = self.ddim_sampling(conditioning, size,
                                                    callback=callback,
                                                    img_callback=img_callback,
                                                    quantize_denoised=quantize_x0,
                                                    mask=mask, x0=x0,
                                                    ddim_use_original_steps=False,
                                                    noise_dropout=noise_dropout,
                                                    temperature=temperature,
                                                    score_corrector=score_corrector,
                                                    corrector_kwargs=corrector_kwargs,
                                                    x_T=x_T,
                                                    log_every_t=log_every_t,
                                                    unconditional_guidance_scale=unconditional_guidance_scale,
                                                    unconditional_conditioning=unconditional_conditioning,
                                                    initial_noise_path=initial_noise_path,
                                                    )
        return samples, intermediates

    def ddim_sampling(self, cond, shape,
                      x_T=None, ddim_use_original_steps=False,
                      callback=None, timesteps=None, quantize_denoised=False,
                      mask=None, x0=None, img_callback=None, log_every_t=100,
                      temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                      unconditional_guidance_scale=1., unconditional_conditioning=None, initial_noise_path=None):
        device = self.model.betas.device
        b = shape[0]

        if self.cvdkwargs['init_latent_load_offset'] >= 0:
            img_cvd = self.load_tensors("./doodl_inits", self.cvdkwargs['init_latent_load_offset'], b).to(device)
        elif initial_noise_path is not None:
            img_cvd = torch.load(initial_noise_path).to(device)
        elif x_T is None:
            if self.rnd is not None:
                img_cvd = self.rnd.randn(shape, device=device)
            else:
                img_cvd = torch.randn(shape, device=device)
        else:
            img_cvd = x_T

        img_pivot = img_cvd.clone()
        if timesteps is None:
            timesteps = self.ddpm_num_timesteps if ddim_use_original_steps else self.ddim_timesteps
        elif timesteps is not None and not ddim_use_original_steps:
            subset_end = int(min(timesteps / self.ddim_timesteps.shape[0], 1) * self.ddim_timesteps.shape[0]) - 1
            timesteps = self.ddim_timesteps[:subset_end]

        intermediates = {'x_inter': [img_cvd], 'pred_x0': [img_cvd]}
        time_range = reversed(range(0, timesteps)) if ddim_use_original_steps else np.flip(timesteps)
        total_steps = timesteps if ddim_use_original_steps else timesteps.shape[0]
        print(f"Running DDIM Sampling with {total_steps} timesteps")

        iterator = tqdm(time_range, desc='DDIM Sampler', total=total_steps)

        for i, step in enumerate(iterator):
            index = total_steps - i - 1
            ts = torch.full((b,), step, device=device, dtype=torch.long)

            if mask is not None:
                assert x0 is not None
                img_orig = self.model.q_sample(x0, ts)  # TODO: deterministic forward pass?
                img_cvd = img_orig * mask + (1. - mask) * img_cvd

            outs = self.p_sample_ddim(img_cvd, img_pivot, cond, ts, index=index,
                                      use_original_steps=ddim_use_original_steps,
                                      quantize_denoised=quantize_denoised, temperature=temperature,
                                      noise_dropout=noise_dropout, score_corrector=score_corrector,
                                      corrector_kwargs=corrector_kwargs,
                                      unconditional_guidance_scale=unconditional_guidance_scale,
                                      unconditional_conditioning=unconditional_conditioning)

            img_cvd, pred_x0_cvd, img_pivot = outs
            self.iter += 1
            if callback: callback(i)
            if img_callback: img_callback(pred_x0_cvd, i)

            if index % log_every_t == 0 or index == total_steps - 1:
                intermediates['x_inter'].append(img_cvd)
                intermediates['pred_x0'].append(pred_x0_cvd)

        return img_cvd, intermediates

    def p_sample_ddim(self, x, x_pivot, c, t, index, repeat_noise=False, use_original_steps=False,
                      quantize_denoised=False,
                      temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                      unconditional_guidance_scale=1., unconditional_conditioning=None):
        b, *_, device = *x.shape, x.device

        torch.set_grad_enabled(True)
        x_cvd = x.detach().requires_grad_(True)
        x_cvd_pivot = x_pivot.detach()

        if unconditional_conditioning is None or unconditional_guidance_scale == 1.:
            e_t_cvd_pivot, attn_maps = self.model.apply_model(x_cvd_pivot, t, c)
            e_t_cvd, _ = self.model.apply_model(x_cvd, t, c, replace_attns=attn_maps)
        else:
            raise NotImplementedError

        if score_corrector is not None:
            assert self.model.parameterization == "eps"
            e_t_cvd = score_corrector.modify_score(self.model, e_t_cvd, x_cvd, t, c, **corrector_kwargs)
            e_t_cvd_pivot = score_corrector.modify_score(self.model, e_t_cvd_pivot, x_cvd_pivot, t, c,
                                                         **corrector_kwargs)

        alphas = self.model.alphas_cumprod if use_original_steps else self.ddim_alphas
        alphas_prev = self.model.alphas_cumprod_prev if use_original_steps else self.ddim_alphas_prev
        sqrt_one_minus_alphas = self.model.sqrt_one_minus_alphas_cumprod if use_original_steps else self.ddim_sqrt_one_minus_alphas
        sigmas = self.model.ddim_sigmas_for_original_num_steps if use_original_steps else self.ddim_sigmas
        # select parameters corresponding to the currently considered timestep
        a_t = torch.full((b, 1, 1, 1), alphas[index], device=device)
        a_prev = torch.full((b, 1, 1, 1), alphas_prev[index], device=device)
        sigma_t = torch.full((b, 1, 1, 1), sigmas[index], device=device)
        sqrt_one_minus_at = torch.full((b, 1, 1, 1), sqrt_one_minus_alphas[index], device=device)

        if self.cvdkwargs['backward_iters'] > 0:
            x_cvd = x_cvd.detach().requires_grad_(True)
            optimizer_backward = AdamW([x_cvd], lr=self.cvdkwargs['backward_lr'])

            for i in range(self.cvdkwargs['backward_iters']):
                optimizer_backward.zero_grad()
                x0_full = self.decode_xt(x_cvd, e_t_cvd.detach(), sqrt_one_minus_at, a_t)
                x0_full_pivot = self.decode_xt(x_cvd_pivot.detach(), e_t_cvd_pivot.detach(), sqrt_one_minus_at, a_t)
                cvd_loss = self.loss_fn(x0_full, x0_full_pivot)
                cvd_loss.backward()
                optimizer_backward.step()

            e_t_cvd, _ = self.model.apply_model(x_cvd, t, c, replace_attns=attn_maps)

        cvd_guidance = self.cond_fn(x_cvd, e_t_cvd, sqrt_one_minus_at, a_t, x_cvd_pivot, e_t_cvd_pivot)
        e_t_cvd = e_t_cvd - a_t.sqrt() * cvd_guidance

        torch.set_grad_enabled(False)
        with torch.no_grad():
            pred_x0_cvd = (x_cvd - sqrt_one_minus_at * e_t_cvd) / a_t.sqrt()
            pred_x0_cvd_pivot = (x_cvd_pivot - sqrt_one_minus_at * e_t_cvd_pivot) / a_t.sqrt()

            if quantize_denoised:
                pred_x0_cvd, _, *_ = self.model.first_stage_model.quantize(pred_x0_cvd)
                pred_x0_cvd_pivot, _, *_ = self.model.first_stage_model.quantize(pred_x0_cvd_pivot)

            # direction pointing to x_t
            dir_xt_cvd = (1. - a_prev - sigma_t ** 2).sqrt() * e_t_cvd
            dir_xt_cvd_pivot = (1. - a_prev - sigma_t ** 2).sqrt() * e_t_cvd_pivot

            noise = sigma_t * noise_like(x_cvd.shape, device, repeat_noise) * temperature
            if noise_dropout > 0.:
                noise = torch.nn.functional.dropout(noise, p=noise_dropout)

            x_prev_cvd = a_prev.sqrt() * pred_x0_cvd + dir_xt_cvd + noise
            x_prev_cvd_pivot = a_prev.sqrt() * pred_x0_cvd_pivot + dir_xt_cvd_pivot + noise

        return x_prev_cvd, pred_x0_cvd, x_prev_cvd_pivot

    def cond_fn(self, x_cvd, e_t_cvd, sqrt_one_minus_at, a_t, x_cvd_pivot, e_t_cvd_pivot, retain_graph=True):
        with torch.enable_grad():
            x_var_d = self.decode_xt(x_cvd, e_t_cvd, sqrt_one_minus_at, a_t)
            x_pivot_d = self.decode_xt(x_cvd_pivot, e_t_cvd_pivot, sqrt_one_minus_at, a_t)

        # def convert(images):
        #     return (images * 255).clip(0, 255).to(torch.uint8).permute(0, 2, 3, 1).cpu()

        # PIL.Image.fromarray(convert(x_pivot_d)[0].numpy(), 'RGB').save(f'/root/inm/x_pivot_{self.iter}.png')
        # PIL.Image.fromarray(convert(x_var_d)[0].numpy(), 'RGB').save(f'/root/inm/x_cvd_{self.iter}.png')

        return self.grad_fn(x_cvd, x_var_d, x_pivot_d, retain_graph=retain_graph)

    def grad_fn(self, x_cvd, x_cvd_d, x_pivot_d, retain_graph=True):
        cvd_loss = self.loss_fn(x_cvd_d, x_pivot_d)
        grad = torch.autograd.grad(-cvd_loss, x_cvd, retain_graph=retain_graph)[0].detach()

        return grad * self.cvdkwargs['gs']

    def rgb2lab(self, imgs):
        lab_images = kcolor.rgb_to_lab(imgs)
        L_channel = lab_images[:, 0:1, :, :]
        a_channel = lab_images[:, 1:2, :, :]
        b_channel = lab_images[:, 2:3, :, :]

        L_channel = (L_channel / 100).clip(0, 1)
        a_channel = ((a_channel * 1 + 128) / 255).clip(0, 1)
        b_channel = ((b_channel * 1 + 128) / 255).clip(0, 1)

        return torch.cat((L_channel, a_channel, b_channel), dim=1)


    def loss_fn(self, x_cvd_d, x_pivot_d):
        x_cvd_d_sim = run_sim(x_cvd_d, self.cvdkwargs['cvd_degree'], x_cvd_d.device,
                              cvd_type=self.cvdkwargs['cvd_type'])

        x_pivot_d_sim = run_sim(x_pivot_d, self.cvdkwargs['cvd_degree'], x_cvd_d.device,
                        cvd_type=self.cvdkwargs['cvd_type'])

        alpha = self.cvdkwargs['cvd_alpha']
        #cvd_loss = alpha * color_info_loss(x_cvd_d, x_cvd_d_sim) + (1 - alpha) * MS_SSIM_loss(x_cvd_d, x_cvd_d_sim).sum()
        cvd_loss = alpha * hist_loss(x_cvd_d.device, x_cvd_d, x_cvd_d_sim) + (1 - alpha) * MS_SSIM_loss(x_cvd_d, x_cvd_d_sim).sum()
        # d_logits = self.run_D(x_cvd_d)
        # d_loss = torch.nn.functional.softplus(-d_logits)
        return cvd_loss# + 1e-2 * d_loss.sum()

    def decode_xt(self, x, e, sqrt_one_minus_at, a_t):
        pred_x0 = (x - sqrt_one_minus_at * e) / a_t.sqrt()
        return self.decode_x0(pred_x0)

    def decode_x0(self, x0):
        # decode first stage vae
        x_var_d = self.model.decode_first_stage_with_grad(x0)

        return ((x_var_d + 1) / 2).clamp(0, 1)

    def load_tensors(self, directory, offset, bs):
        # Initialize an empty list to store loaded tensors
        loaded_tensors = []

        # Calculate the ending index
        end = offset + bs

        # Load tensors from start_x to end_x - 1
        for x in range(offset, end):
            filename = f"doodl_inits_{x}.pt"
            file_path = os.path.join(directory, filename)
            if os.path.exists(file_path):
                tensor = torch.load(file_path)
                loaded_tensors.append(tensor)
            else:
                raise ValueError(f"File {file_path} does not exist when loading pre-existing tensors.")

        return torch.cat(loaded_tensors)

    def run_D(self, img, c=None):
        if c is None:
            c = torch.empty(img.shape[0], 0, device=img.device, dtype=img.dtype)

        return self.D(img, c)

