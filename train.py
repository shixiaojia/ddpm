import torch
import argparse
from torch.utils.data import DataLoader
from torch.optim import Adam
from utils import MyDataset
from torchvision.utils import save_image
from tqdm import tqdm
from model import UNet
from diffusion import GaussionDiffusion


def args_parser():
    parser = argparse.ArgumentParser(description="Parameters of training vae model")
    parser.add_argument("-b", "--batch_size", type=int, default=64)
    parser.add_argument("-d", "--latent_dim", type=int, default=64)
    parser.add_argument("-l", "--lr", type=float, default=1e-4)
    parser.add_argument("-w", "--weight_decay", type=float, default=1e-5)
    parser.add_argument("-e", "--epoch", type=int, default=500)
    parser.add_argument("-v", "--snap_epoch", type=int, default=1)
    parser.add_argument("-n", "--num_samples", type=int, default=64)
    parser.add_argument("--T", type=int, default=1000)
    parser.add_argument("--ch", type=int, default=64)
    parser.add_argument("--ch_ratio", type=list, default=[1, 2, 2, 2])
    parser.add_argument("--num_res_block", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument('--beta_1', type=float, default=1e-4, help='start beta value')
    parser.add_argument('--beta_T', type=float, default=0.02, help='end beta value')
    parser.add_argument("--image_size", type=int, default=32)
    parser.add_argument("--image_channels", type=int, default=3)
    return parser.parse_args()


if __name__ == '__main__':
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    opt = args_parser()

    dataset = MyDataset(img_path="../faces/", device=DEVICE)
    train_loader = DataLoader(dataset=dataset, batch_size=opt.batch_size, shuffle=True, num_workers=0)
    model = UNet(T=opt.T, ch=opt.ch, ch_ratio=opt.ch_ratio, num_res_block=opt.num_res_block, dropout=opt.dropout)

    diffusion = GaussionDiffusion(model, opt.image_size, opt.image_channels, opt.beta_1, opt.beta_T, opt.T).to(DEVICE)
    optimizer = Adam(diffusion.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)

    for epoch in range(opt.epoch):
        diffusion.train()

        for step, x_0 in enumerate(train_loader):
            pbar = tqdm(total=len(train_loader), desc=f"Epoch {step + 1}/{len(train_loader)}")
            optimizer.zero_grad()
            loss = diffusion(x_0.to(DEVICE))
            loss.backward()
            optimizer.step()

        if epoch % opt.snap_epoch == 0 or epoch == opt.epoch - 1:
            diffusion.eval()
            with torch.no_grad():
                images = diffusion.sample(opt.num_samples, device=DEVICE)
                imgs = images.detach().cpu().numpy()
                fname = './my_generated-images-epoch_{0:0=4d}.png'.format(epoch)
                save_image(images, fname, nrow=8)
                torch.save(diffusion.state_dict(), f"./model_step_{epoch}.pth")
