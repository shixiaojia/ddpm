import torch
import argparse
from torchvision.utils import save_image
from model import UNet
from diffusion import DDPMSampler, DDIMSampler


def args_parser():
    parser = argparse.ArgumentParser(description="Parameters of training vae model")
    parser.add_argument("-b", "--batch_size", type=int, default=64)
    parser.add_argument("-i", "--in_channels", type=int, default=3)
    parser.add_argument("-n", "--num_samples", type=int, default=64)
    parser.add_argument("-p", "--path", type=str, default="./results_linear")
    parser.add_argument("--T", type=int, default=1000)
    parser.add_argument("--ch", type=int, default=64)
    parser.add_argument("--ch_ratio", type=list, default=[1, 2, 2, 2])
    parser.add_argument("--num_res_block", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument('--beta_1', type=float, default=1e-4, help='start beta value')
    parser.add_argument('--beta_T', type=float, default=0.02, help='end beta value')
    parser.add_argument("--image_size", type=int, default=32)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--sampler", type=str, choices=["ddpm", "ddim"], default="ddpm")

    return parser.parse_args()


if __name__ == '__main__':
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    opt = args_parser()

    model = UNet(T=opt.T, ch=opt.ch, ch_ratio=opt.ch_ratio, num_res_block=opt.num_res_block, dropout=opt.dropout)
    model.load_state_dict(torch.load("./model_step_7.pth"))
    model.eval()

    if opt.sampler == "ddpm":
        sampler = DDPMSampler(model, image_size=opt.image_size, image_channel=opt.in_channels, beta_1=opt.beta_1,
                              beta_T=opt.beta_T, T=opt.T).to(DEVICE)
    elif opt.sampler == "ddim":
        sampler = DDIMSampler(model, image_size=opt.image_size, image_channel=opt.in_channels, beta_1=opt.beta_1,
                              beta_T=opt.beta_T, T=opt.T).to(DEVICE)
    else:
        raise ValueError(f"invalid sampler: {opt.sampler}")

    images = sampler(batch_size=opt.num_samples, steps=opt.steps, device=DEVICE)
    fname = './my_generated-images.png'
    save_image(images, fname, nrow=8)


