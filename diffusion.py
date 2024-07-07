import torch
import torch.nn as nn
import torch.nn.functional as F


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

    def forward(self, x_0):
        t = torch.randint(self.T, size=(x_0.shape[0], ), device=x_0.device)
        noise = torch.randn_like(x_0)
        x_t = (
            extract(self.sqrt_alphas_bar, t, x_0.shape) * x_0 +
            extract(self.sqrt_one_minus_alphas_bar, t, x_0.shape) * noise)
        loss = F.mse_loss(self.model(x_t, t), noise, reduction='mean')
        return loss

    def sample(self, batch_size, device):
        x = torch.randn(batch_size, self.image_channel, self.image_size, self.image_size, device=device)

        for t in reversed(range(self.T)):
            # t = x_t.new_ones([x_t.shape[0], ], dtype=torch.long) * time_step
            t_batch = torch.tensor([t], device=device).repeat(batch_size)
            x = (x - extract(self.remove_noise_coef, t_batch, x.shape) * self.model(x, t_batch)) * \
                extract(self.reciprocal_sqrt_alphas, t_batch, x.shape)

            if t > 0:
                x += extract(self.sigma, t_batch, x.shape) * torch.randn_like(x)
        return x
