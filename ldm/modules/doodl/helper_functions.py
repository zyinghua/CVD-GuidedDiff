import torch
from torch import nn


class SteppingLayer(nn.Module):
    """
    This is a layer that performs DDIM stepping that will be wrapped
    by memcnn to be invertible
    """

    def __init__(self, scheduler,  # ddim self
                 embedding_c=None,
                 embedding_uc=None,
                 num_timesteps=50, # inference steps
                 unconditional_guidance_scale=7.5,
                 ):
        super(SteppingLayer, self).__init__()
        self.scheduler = scheduler
        self.emb_c = embedding_c
        self.emb_uc = embedding_uc
        self.num_timesteps = num_timesteps
        self.unconditional_guidance_scale = unconditional_guidance_scale

    def forward(self, i, t, latent_pair,
                reverse=False):
        """
        Run an EDICT step
        """
        for base_latent_i in range(2):
            # Need to alternate order compatibly forward and backward
            if reverse:
                orig_i = self.num_timesteps - (i + 1)
                offset = (orig_i + 1) % 2
                latent_i = (base_latent_i + offset) % 2
            else:
                offset = i % 2
                latent_i = (base_latent_i + offset) % 2

            # leapfrog steps/run baseline logic hardcoded here
            latent_j = ((latent_i + 1) % 2)

            latent_i = latent_i.long()
            latent_j = latent_j.long()

            # select latent model input
            if base_latent_i == 0:
                latent_model_input = latent_pair.index_select(0, latent_j)
            else:
                latent_model_input = first_output
            latent_base = latent_pair.index_select(0, latent_i)

            #cfg
            if self.emb_uc is None or self.guidance_scale == 1.:
                noise_pred, _ = self.scheduler.model.apply_model(latent_model_input, t, self.emb_c)
            else:
                x_in = torch.cat([latent_model_input] * 2)
                t_in = torch.cat([t] * 2)
                c_in = torch.cat([self.emb_uc, self.emb_c])
                e_t_uncond, e_t = self.scheduler.model.apply_model(x_in, t_in, c_in).chunk(2)
                noise_pred = e_t_uncond + self.unconditional_guidance_scale * (e_t - e_t_uncond)

            # Going forward or backward?
            step_call = reverse_step if reverse else forward_step
            # Step
            new_latent = step_call(self.scheduler,
                                   noise_pred,
                                   self.num_timesteps - int(i.item()) - 1,
                                   latent_base)
            new_latent = new_latent.to(latent_base.dtype)

            if base_latent_i == 0:  # first pass
                first_output = new_latent
            else:  # second pass
                second_output = new_latent
                if latent_i == 1:  # so normal order
                    combined_outputs = torch.cat([first_output, second_output])
                else:  # Offset so did in reverse
                    combined_outputs = torch.cat([second_output, first_output])

        return combined_outputs

    def inverse(self, i, t, latent_pair):
        # Inverse method for memcnn
        output = self.forward(i, t, latent_pair, reverse=True)
        return output


class MixingLayer(nn.Module):
    """
    This does the mixing layer of EDICT
    https://arxiv.org/abs/2211.12446
    Equations 12/13
    """

    def __init__(self, mix_weight=0.93):
        super(MixingLayer, self).__init__()
        self.p = mix_weight

    def forward(self, input_x):
        input_x0, input_x1 = input_x[:1], input_x[1:]
        x0 = self.p * input_x0 + (1 - self.p) * input_x1
        x1 = (1 - self.p) * x0 + self.p * input_x1
        return torch.cat([x0, x1])

    def inverse(self, input_x):
        input_x0, input_x1 = input_x.split(1)
        x1 = (input_x1 - (1 - self.p) * input_x0) / self.p
        x0 = (input_x0 - (1 - self.p) * x1) / self.p
        return torch.cat([x0, x1])


# forward DDIM step
def forward_step(
        self,
        model_output,
        index,  # small to large
        sample,
        use_original_steps=False,
):
    b, *_, device = *model_output.shape, model_output.device

    alphas = self.model.alphas_cumprod if use_original_steps else self.ddim_alphas
    alphas_prev = self.model.alphas_cumprod_prev if use_original_steps else self.ddim_alphas_prev
    # select parameters corresponding to the currently considered timestep
    a_t = torch.full((b, 1, 1, 1), alphas[index], device=device)
    a_prev = torch.full((b, 1, 1, 1), alphas_prev[index], device=device)

    alpha_prod_t, beta_prod_t = a_t, 1 - a_t
    alpha_prod_t_prev, _ = a_prev, 1 - a_prev

    alpha_quotient = ((alpha_prod_t / alpha_prod_t_prev) ** 0.5)

    first_term = (1. / alpha_quotient) * sample
    second_term = (1. / alpha_quotient) * (beta_prod_t ** 0.5) * model_output
    third_term = ((1 - alpha_prod_t_prev) ** 0.5) * model_output
    return first_term - second_term + third_term


# reverse ddim step
def reverse_step(
        self,
        model_output,
        index,  # small to large
        sample,
        use_original_steps=False,
):
    b, *_, device = *model_output.shape, model_output.device

    alphas = self.model.alphas_cumprod if use_original_steps else self.ddim_alphas
    alphas_prev = self.model.alphas_cumprod_prev if use_original_steps else self.ddim_alphas_prev
    # select parameters corresponding to the currently considered timestep
    a_t = torch.full((b, 1, 1, 1), alphas[index], device=device)
    a_prev = torch.full((b, 1, 1, 1), alphas_prev[index], device=device)

    alpha_prod_t, beta_prod_t = a_t, 1 - a_t
    alpha_prod_t_prev, _ = a_prev, 1 - a_prev

    alpha_quotient = ((alpha_prod_t / alpha_prod_t_prev) ** 0.5)

    first_term = alpha_quotient * sample
    second_term = ((beta_prod_t) ** 0.5) * model_output
    third_term = alpha_quotient * ((1 - alpha_prod_t_prev) ** 0.5) * model_output
    return first_term + second_term - third_term
