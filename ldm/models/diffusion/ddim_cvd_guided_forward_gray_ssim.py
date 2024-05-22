"""SAMPLING ONLY."""

import torch
import numpy as np
import PIL
from tqdm import tqdm
from torch.optim import AdamW, Adam
import kornia
from PIL import Image
from torch import autocast

from ldm.modules.diffusionmodules.util import make_ddim_sampling_parameters, make_ddim_timesteps, noise_like
from ldm.modules.losses.cvd_loss import *
from ldm.modules.doodl.helper_functions import *
from ldm.modules.doodl import memcnn


class DDIMCVDSampler(object):
    def __init__(self, model, cvd_args, schedule="linear", rnd=None, D=None):
        super().__init__()
        self.model = model
        self.ddpm_num_timesteps = model.num_timesteps
        self.schedule = schedule
        self.cvdkwargs = cvd_args
        self.rnd = rnd
        self.D = D.to(self.model.device)
        self.iter = 0

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
                                                    )
        return samples, intermediates

    def ddim_sampling(self, cond, shape,
                      x_T=None, ddim_use_original_steps=False,
                      callback=None, timesteps=None, quantize_denoised=False,
                      mask=None, x0=None, img_callback=None, log_every_t=100,
                      temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                      unconditional_guidance_scale=1., unconditional_conditioning=None, ):
        device = self.model.betas.device
        b = shape[0]
        if x_T is None:
            if self.rnd is not None:
                img_cvd = self.rnd.randn(shape, device=device)
            else:
                img_cvd = torch.randn(shape, device=device)
        else:
            img_cvd = x_T

        #img_cvd = self.doodl(img_cvd)
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
            e_t_cvd, attn_maps_cvd = self.model.apply_model(x_cvd, t, c)
            e_t_cvd_pivot, attn_maps = self.model.apply_model(x_cvd_pivot, t, c)
        else:
            x_cvd_in = torch.cat([x_cvd] * 2)
            t_in = torch.cat([t] * 2)
            c_in = torch.cat([unconditional_conditioning, c])
            e_t_uncond_cvd, e_t_cvd = self.model.apply_model(x_cvd_in, t_in, c_in).chunk(2)
            e_t_cvd = e_t_uncond_cvd + unconditional_guidance_scale * (e_t_cvd - e_t_uncond_cvd)

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

        cvd_guidance = self.cond_fn(x_cvd, e_t_cvd, sqrt_one_minus_at, a_t, x_cvd_pivot, e_t_cvd_pivot)
        e_t_cvd = e_t_cvd - a_t * cvd_guidance

        # ==================================================================
        # ==================================================================

        e_t_cvd = e_t_cvd.clone().detach().requires_grad_(True)
        x_cvd = x_cvd.requires_grad_(False)
        optimizer = Adam([e_t_cvd], lr=3e-2 * ((self.iter + 1) / 50.))

        for j in range(50):
            optimizer.zero_grad()
            x_var_d = self.decode_xt(x_cvd.detach(), e_t_cvd, sqrt_one_minus_at, a_t)
            x_var_d_pivot = self.decode_xt(x_cvd_pivot.detach(), e_t_cvd_pivot.detach(), sqrt_one_minus_at, a_t)

            x_var_d_gray = kornia.color.rgb_to_grayscale(x_var_d)
            x_var_d_pivot_gray = kornia.color.rgb_to_grayscale(x_var_d_pivot)

            pivot_loss = structural_correction_loss(x_var_d_pivot_gray, x_var_d_gray)
            pivot_loss.backward()
            #print(f"pivot loss: {pivot_loss}")
            optimizer.step()

        # ==================================================================
        # PRINT GRAYSCALE IMAGES
        # ==================================================================

        x_var_d = self.decode_xt(x_cvd.detach(), e_t_cvd, sqrt_one_minus_at, a_t)
        x_var_d_pivot = self.decode_xt(x_cvd_pivot.detach(), e_t_cvd_pivot.detach(), sqrt_one_minus_at, a_t)

        x_var_d_gray = kornia.color.rgb_to_grayscale(x_var_d)
        x_var_d_pivot_gray = kornia.color.rgb_to_grayscale(x_var_d_pivot)

        def convert_to_grayscale(images):
            return (images * 255).clip(0, 255).to(torch.uint8).squeeze(1).cpu()
        PIL.Image.fromarray(convert_to_grayscale(x_var_d_pivot_gray)[0].numpy(), 'L').save(f'/root/inm/x_pivot_gray_{self.iter}.png')
        PIL.Image.fromarray(convert_to_grayscale(x_var_d_gray)[0].numpy(), 'L').save(f'/root/inm/x_cvd_gray_{self.iter}.png')

        # ==================================================================
        # ==================================================================
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
            x_var_d_pivot = self.decode_xt(x_cvd_pivot, e_t_cvd_pivot, sqrt_one_minus_at, a_t)

            def convert(images):
                return (images * 255).clip(0, 255).to(torch.uint8).permute(0, 2, 3, 1).cpu()

            PIL.Image.fromarray(convert(x_var_d_pivot)[0].numpy(), 'RGB').save(f'/root/inm/x_pivot_{self.iter}.png')
            PIL.Image.fromarray(convert(x_var_d)[0].numpy(), 'RGB').save(f'/root/inm/x_cvd_{self.iter}.png')
            # PIL.Image.fromarray(convert(x_cvd_d_sim)[0].numpy(), 'RGB').save(f'/root/inm/x_cvd_sim_{self.it}.png')

        return self.grad_fn(x_cvd, x_var_d, retain_graph=retain_graph)

    def grad_fn(self, x_cvd, x_cvd_d, retain_graph=True):
        cvd_loss = self.loss_fn(x_cvd_d)
        grad = torch.autograd.grad(-cvd_loss, x_cvd, retain_graph=retain_graph)[0].detach()

        return grad * self.cvdkwargs['gs']

    def loss_fn(self, x_cvd_d):
        x_cvd_d_sim = run_sim(x_cvd_d, self.cvdkwargs['cvd_degree'], x_cvd_d.device,
                              cvd_type=self.cvdkwargs['cvd_type'])

        cvd_loss = 5 * color_info_loss(x_cvd_d, x_cvd_d_sim) + 50 * MS_SSIM_loss(x_cvd_d, x_cvd_d_sim).sum()
        return cvd_loss

    def decode_xt(self, x, e, sqrt_one_minus_at, a_t):
        pred_x0 = (x - sqrt_one_minus_at * e) / a_t.sqrt()
        return self.decode_x0(pred_x0)

    def decode_x0(self, x0):
        # decode first stage vae
        x_var_d = self.model.decode_first_stage_with_grad(x0)

        return ((x_var_d + 1) / 2).clamp(0, 1)

    def doodl(self,
            latent,
            inference_steps=50,
            embedding_conditional=None,
            embedding_unconditional=None,
            grad_scale=1,
            unconditional_guidance_scale=7.5,
            mix_weight=0.93,  # .7
            num_traversal_steps=50,
            tied_latents=True,
            use_momentum=True,
            use_nesterov=False,
            renormalize_latents=True,
            optimize_first_edict_image_only=False,
            perturb_grad_scale=1e-4,
            clip_grad_val=1e-3,
            ddim_use_original_steps=False,
            device='cuda',
    ):
        latent_pair = torch.cat([latent.clone(), latent.clone()])

        if renormalize_latents:  # if renormalize_latents then get original norm value
            orig_norm = latent.norm().item()

        mix = MixingLayer(mix_weight)
        mix = memcnn.InvertibleModuleWrapper(mix, keep_input=True, keep_input_inverse=True, num_bwd_passes=1)

        s = SteppingLayer(self,
                          embedding_conditional,
                          embedding_unconditional,
                          num_timesteps=inference_steps,
                          unconditional_guidance_scale=unconditional_guidance_scale)
        s = memcnn.InvertibleModuleWrapper(s, keep_input=True,
                                           keep_input_inverse=True,
                                           num_bwd_passes=1)

        # SGD boiler plate
        if use_momentum: prev_b_arr = [None, None]

        """
        PERFORM GRADIENT DESCENT DIRECTLY ON LATENTS USING GRADIENT CALCUALTED THROUGH WHOLE CHAIN
        """
        # turn on gradient calculation
        with torch.enable_grad(): # important b/c don't have on by default in module
            for m in range(num_traversal_steps): # This is # of optimization steps
                print(f"Optimization Step {m}")

                # Get clone of latent pair
                orig_latent_pair = latent_pair.clone().detach().requires_grad_(True)
                input_latent_pair = orig_latent_pair.clone()
                with autocast(device):
                    timesteps = self.ddpm_num_timesteps if ddim_use_original_steps else self.ddim_timesteps
                    time_range = reversed(range(0, timesteps)) if ddim_use_original_steps else np.flip(timesteps)
                    total_steps = timesteps if ddim_use_original_steps else timesteps.shape[0]
                    print(f"DOODL: Running DDIM Sampling with {total_steps} timesteps.")

                    iterator = tqdm(time_range, desc='DDIM Sampler', total=total_steps)

                    for i, step in enumerate(iterator):
                        ts = torch.full((b,), step, device=device, dtype=torch.long)

                        i = torch.tensor([i], device=latent_pair.device, dtype=torch.float32)
                        input_latent_pair = s(i, ts, input_latent_pair)
                        input_latent_pair = mix(input_latent_pair)

                    # Get the images that the latents yield
                    ims = [self.decode_x0(l) for l in input_latent_pair.chunk(2)]  # in [0-1]

                # save images and compute loss
                # save image to ims/{save_str.replace(.png, _m.png)
                losses = []
                for im_i, im in enumerate(ims):
                    # If guiding then compute loss
                    if grad_scale != 0:
                        loss = self.loss_fn(im)
                        losses.append(loss)
                        if optimize_first_edict_image_only: break

                sum_loss = sum(losses)

                # Backward pass
                sum_loss.backward()
                # Access latent gradient directly
                grad = -0.5 * orig_latent_pair.grad
                # Average gradients if tied_latents
                if tied_latents:
                    grad = grad.mean(dim=0, keepdim=True)
                    grad = grad.repeat(2, 1, 1, 1)

                new_latents = []
                # doing perturbation linked as well
                # Perturbation is just random noise added
                perturbation = perturb_grad_scale * torch.randn_like(orig_latent_pair[0]) if perturb_grad_scale else 0

                # SGD step (S=stochastic from multicrop, can also just be GD)
                # Iterate through latents/grads
                for grad_idx, (g, l) in enumerate(zip(grad.chunk(2), orig_latent_pair.chunk(2))):

                    # Clip max magnitude
                    if clip_grad_val is not None:
                        g = g.clip(-clip_grad_val, clip_grad_val)

                    # SGD code
                    # https://pytorch.org/docs/stable/generated/torch.optim.SGD.html
                    if use_momentum:
                        mom = 0.9
                        # LR is grad scale
                        # sticking with generic 0.9 momentum for now, no dampening
                        if m == 0:
                            b = g
                        else:
                            b = mom * prev_b_arr[grad_idx] + g
                        if use_nesterov:
                            g = g + mom * b
                        else:
                            g = b
                        prev_b_arr[grad_idx] = b.clone()
                    new_l = l + g + perturbation
                    new_latents.append(new_l.clone())
                if tied_latents:  # don't think is needed with other tied_latent logic but just being safe
                    combined_l = 0.5 * (new_latents[0] + new_latents[1])
                    latent_pair = combined_l.repeat(2, 1, 1, 1)
                else:
                    latent_pair = torch.cat(new_latents)

                if renormalize_latents: # Renormalize latents
                    for norm_i in range(2):
                        latent_pair[norm_i] = latent_pair[norm_i] * orig_norm / latent_pair[norm_i].norm().item()

        return latent_pair.chunk(2)[0]
