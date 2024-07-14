import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm


def extract(v, t, x_shape):
    """
    Extract some coefficients at specified timesteps, then reshape to
    [batch_size, 1, 1, 1, 1, ...] for broadcasting purposes.
    """
    out = torch.gather(v, index=t, dim=0).float()
    return out.view([t.shape[0]] + [1] * (len(x_shape) - 1))


class GaussionDiffusion(nn.Module):
    def __init__(self, model, image_size, image_channel, beta_1, beta_T, T):
        super(GaussionDiffusion, self).__init__()
        self.model = model
        self.image_size = image_size
        self.image_channel = image_channel
        self.T = T

        betas = torch.linspace(beta_1, beta_T, T).double()
        alphas = 1. - betas
        alphas_bar = torch.cumprod(alphas, dim=0)

        self.register_buffer("betas", betas)
        self.register_buffer("sqrt_alphas_bar", torch.sqrt(alphas_bar))
        self.register_buffer("sqrt_one_minus_alphas_bar", torch.sqrt(1. - alphas_bar))
        self.register_buffer("remove_noise_coef", betas/torch.sqrt(1-alphas_bar))
        self.register_buffer("reciprocal_sqrt_alphas", 1./torch.sqrt(alphas))
        self.register_buffer("sigma", torch.sqrt(betas))
        self.register_buffer("alpha_t_bar", alphas_bar)

    def forward(self, x_0):
        t = torch.randint(self.T, size=(x_0.shape[0], ), device=x_0.device)
        noise = torch.randn_like(x_0)
        x_t = (
            extract(self.sqrt_alphas_bar, t, x_0.shape) * x_0 +
            extract(self.sqrt_one_minus_alphas_bar, t, x_0.shape) * noise)
        loss = F.mse_loss(self.model(x_t, t), noise, reduction='mean')
        return loss


class DDPMSampler(nn.Module):
    def __init__(self, model, image_size, image_channel, beta_1, beta_T, T):
        super(DDPMSampler, self).__init__()
        self.model = model
        self.image_size = image_size
        self.image_channel = image_channel
        self.T = T

        betas = torch.linspace(beta_1, beta_T, T).double()
        alphas = 1. - betas
        alphas_bar = torch.cumprod(alphas, dim=0)

        self.register_buffer("remove_noise_coef", betas/torch.sqrt(1-alphas_bar))
        self.register_buffer("reciprocal_sqrt_alphas", 1./torch.sqrt(alphas))
        self.register_buffer("sigma", torch.sqrt(betas))

    @torch.no_grad()
    def forward(self, batch_size, device, **kwargs):
        x = torch.randn(batch_size, self.image_channel, self.image_size, self.image_size, device=device)

        with tqdm(reversed(range(0, self.T)), total=self.T) as sampling_steps:
            for t in sampling_steps:
                t_batch = torch.tensor([t], device=device).repeat(batch_size)
                x = (x - extract(self.remove_noise_coef, t_batch, x.shape) * self.model(x, t_batch)) * \
                    extract(self.reciprocal_sqrt_alphas, t_batch, x.shape)

                if t > 0:
                    x += extract(self.sigma, t_batch, x.shape) * torch.randn_like(x)
                sampling_steps.set_description("sample")
                sampling_steps.set_postfix(ordered_dict={"step": t + 1})
            return x


class DDIMSampler(nn.Module):
    def __init__(self, model, image_size, image_channel, beta_1, beta_T, T):
        super(DDIMSampler, self).__init__()
        self.model = model
        self.image_size = image_size
        self.image_channel = image_channel
        self.T = T
        betas = torch.linspace(beta_1, beta_T, T).double()
        alphas = 1. - betas
        alphas_bar = torch.cumprod(alphas, dim=0)
        self.register_buffer("alpha_t_bar", alphas_bar)

    @torch.no_grad()
    def sample_one_step(self, x_t, time_step: int, prev_time_step: int, eta: float):
        t = torch.full((x_t.shape[0],), time_step, device=x_t.device, dtype=torch.long)
        prev_t = torch.full((x_t.shape[0],), prev_time_step, device=x_t.device, dtype=torch.long)

        # get current and previous alpha_cumprod
        alpha_t = extract(self.alpha_t_bar, t, x_t.shape)
        alpha_t_prev = extract(self.alpha_t_bar, prev_t, x_t.shape)

        # predict noise using model
        epsilon_theta_t = self.model(x_t, t)

        # calculate x_{t-1}
        sigma_t = eta * torch.sqrt((1 - alpha_t_prev) / (1 - alpha_t) * (1 - alpha_t / alpha_t_prev))
        epsilon_t = torch.randn_like(x_t)
        x_t_minus_one = (
                torch.sqrt(alpha_t_prev / alpha_t) * x_t +
                (torch.sqrt(1 - alpha_t_prev - sigma_t ** 2) - torch.sqrt(
                    (alpha_t_prev * (1 - alpha_t)) / alpha_t)) * epsilon_theta_t +
                sigma_t * epsilon_t
        )
        return x_t_minus_one

    @torch.no_grad()
    def forward(self, batch_size, steps, device):
        a = self.T // steps
        time_steps = np.asarray(list(range(0, self.T, a)))

        time_steps = time_steps + 1

        time_steps_prev = np.concatenate([[0], time_steps[:-1]])

        x_t = torch.randn((batch_size, self.image_channel, self.image_size, self.image_size), device=device)
        x = [x_t]
        eta = 0.0
        with tqdm(reversed(range(0, steps)), total=steps) as sampling_steps:
            for i in sampling_steps:
                x_t = self.sample_one_step(x_t, time_steps[i], time_steps_prev[i], eta)

                sampling_steps.set_postfix(ordered_dict={"step": i + 1})

            return x_t