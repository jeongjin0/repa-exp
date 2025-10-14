import os
import numpy as np
from PIL import Image
from tqdm import tqdm
from scipy import linalg

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image



# -------------------------
# Feature Extractor (InceptionV3)
# -------------------------
class InceptionV3FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        inception = models.inception_v3(pretrained=True, transform_input=False)
        inception.Mixed_7c.register_forward_hook(self.output_hook)
        self.inception = inception
        self.out = None

    def output_hook(self, module, input, output):
        self.out = F.adaptive_avg_pool2d(output, output_size=(1, 1))

    def forward(self, x):
        _ = self.inception(x)
        return self.out.view(x.size(0), -1)


# -------------------------
# Image Loader
# -------------------------
def get_image_paths(folder):
    return [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(".jpg")]


class ImageFolderDataset(Dataset):
    def __init__(self, folder, transform=None):
        self.paths = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(".jpg")]
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img


# -------------------------
# Compute activations
# -------------------------
def get_activations(loader, model, device="cuda"):
    model.eval()
    feats = []
    with torch.no_grad():
        for batch in tqdm(loader):
            batch = batch.to(device)
            pred = model(batch)
            feats.append(pred.cpu().numpy())
    feats = np.concatenate(feats, axis=0)
    return feats


# -------------------------
# FID Calculation
# -------------------------
def calculate_fid(mu1, sigma1, mu2, sigma2, eps=1e-6):
    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        covmean = covmean.real
        covmean = covmean + np.eye(sigma1.shape[0]) * eps
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = diff.dot(diff) + np.trace(sigma1 + sigma2 - 2 * covmean)
    return fid


# -------------------------
# Main
# -------------------------
def compute_fid(folder1, folder2, device="cuda"):
    # Transform
    transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])
    ])

    # Model
    model = InceptionV3FeatureExtractor().to(device)

    # Load datasets
    paths1 = get_image_paths(folder1)
    paths2 = get_image_paths(folder2)

    dataset1 = ImageFolderDataset(folder1, transform=transform)
    dataset2 = ImageFolderDataset(folder2, transform=transform)

    loader1 = DataLoader(dataset1, batch_size=32, shuffle=False, num_workers=4)
    loader2 = DataLoader(dataset2, batch_size=32, shuffle=False, num_workers=4)

    # Features
    act1 = get_activations(loader1, model, device)
    act2 = get_activations(loader2, model, device)

    mu1, sigma1 = act1.mean(axis=0), np.cov(act1, rowvar=False)
    mu2, sigma2 = act2.mean(axis=0), np.cov(act2, rowvar=False)

    fid_value = calculate_fid(mu1, sigma1, mu2, sigma2)
    return fid_value


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--folder1", type=str, required=True, help="Path to the first image folder")
    parser.add_argument("--folder2", type=str, required=True, help="Path to the second image folder")

    args = parser.parse_args()

    fid = compute_fid(args.folder1, args.folder2, device="cuda:0" if torch.cuda.is_available() else "cpu")
    print(args.folder1, args.folder2)
    print("FID:", fid)