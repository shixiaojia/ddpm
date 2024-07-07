import os
from torch.utils.data import Dataset
from torchvision.transforms import transforms
import glob
import cv2


class MyDataset(Dataset):
    def __init__(self, img_path, device):
        super(MyDataset, self).__init__()
        self.device = device
        self.fnames = glob.glob(os.path.join(img_path+"*.jpg"))
        self.transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((32, 32)),
            # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def __getitem__(self, idx):
        fname = self.fnames[idx]
        img = cv2.imread(fname, cv2.IMREAD_COLOR)
        img = self.transforms(img)
        img = img.to(self.device)
        return img

    def __len__(self):
        return len(self.fnames)
