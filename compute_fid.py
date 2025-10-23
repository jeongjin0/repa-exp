import os
import numpy as np
from PIL import Image
from tqdm import tqdm
from scipy import linalg
import hashlib

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
# Cache Management
# -------------------------
def get_folder_hash(folder):
    """폴더의 고유 식별자를 생성합니다."""
    # 폴더 경로의 절대 경로를 해시화
    abs_path = os.path.abspath(folder)
    # 폴더 내 파일 목록과 수정 시간을 포함하여 해시 생성
    files = sorted([f for f in os.listdir(folder) if f.endswith(".jpg")])
    
    hash_str = abs_path + str(len(files))
    # 첫 5개와 마지막 5개 파일의 이름과 크기를 해시에 포함
    sample_files = files[:5] + files[-5:] if len(files) > 10 else files
    for f in sample_files:
        fpath = os.path.join(folder, f)
        hash_str += f + str(os.path.getsize(fpath))
    
    return hashlib.md5(hash_str.encode()).hexdigest()


def get_cache_path(folder, cache_dir=".fid_cache"):
    """캐시 파일 경로를 반환합니다."""
    os.makedirs(cache_dir, exist_ok=True)
    folder_hash = get_folder_hash(folder)
    cache_filename = f"activations_{folder_hash}.npz"
    return os.path.join(cache_dir, cache_filename)


def load_cached_activations(folder, cache_dir=".fid_cache"):
    """캐시된 활성화 값을 불러옵니다."""
    cache_path = get_cache_path(folder, cache_dir)
    if os.path.exists(cache_path):
        print(f"Loading cached activations for {folder}")
        data = np.load(cache_path)
        return data['activations']
    return None


def save_cached_activations(folder, activations, cache_dir=".fid_cache"):
    """활성화 값을 캐시에 저장합니다."""
    cache_path = get_cache_path(folder, cache_dir)
    print(f"Saving activations to cache: {cache_path}")
    np.savez_compressed(cache_path, activations=activations)


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


def get_activations_with_cache(folder, model, transform, device="cuda", cache_dir=".fid_cache", use_cache=True):
    """캐시를 사용하여 활성화 값을 가져옵니다."""
    # 캐시 확인
    if use_cache:
        cached_act = load_cached_activations(folder, cache_dir)
        if cached_act is not None:
            return cached_act
    
    # 캐시가 없으면 계산
    print(f"Computing activations for {folder}")
    dataset = ImageFolderDataset(folder, transform=transform)
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)
    activations = get_activations(loader, model, device)
    
    # 캐시 저장
    if use_cache:
        save_cached_activations(folder, activations, cache_dir)
    
    return activations


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
def compute_fid(folder1, folder2, device="cuda", cache_dir=".fid_cache", use_cache=True):
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

    # Get activations with cache
    act1 = get_activations_with_cache(folder1, model, transform, device, cache_dir, use_cache)
    act2 = get_activations_with_cache(folder2, model, transform, device, cache_dir, use_cache)

    # Calculate statistics
    mu1, sigma1 = act1.mean(axis=0), np.cov(act1, rowvar=False)
    mu2, sigma2 = act2.mean(axis=0), np.cov(act2, rowvar=False)

    fid_value = calculate_fid(mu1, sigma1, mu2, sigma2)
    return fid_value


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--folder1", type=str, required=True, help="Path to the first image folder")
    parser.add_argument("--folder2", type=str, required=True, help="Path to the second image folder")
    parser.add_argument("--cache-dir", type=str, default=".fid_cache", help="Directory to store cache files")
    parser.add_argument("--no-cache", action="store_true", help="Disable caching")

    args = parser.parse_args()

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    fid = compute_fid(
        args.folder1, 
        args.folder2, 
        device=device,
        cache_dir=args.cache_dir,
        use_cache=not args.no_cache
    )
    
    print(args.folder1, args.folder2)
    print("FID:", fid)