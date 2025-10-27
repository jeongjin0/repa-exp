import os
import torch
from torchvision import datasets, transforms
from models.dit import DiT_models
from models.dit_repa_independent import DiTZeroflowintegrated_independent_t
from tqdm import tqdm
import torch.nn.functional as F


def compute_dinov2_embeddings(images, dinov2_model, device):
    with torch.no_grad():
        # [-1, 1] → [0, 1]
        img_normalized = (images + 1) / 2
        
        # 224x224로 resize
        img_resized = F.interpolate(
            img_normalized,
            size=(224, 224),
            mode='bicubic',
            align_corners=False
        )
        
        # DINOv2 normalize
        mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
        img_dino = (img_resized - mean) / std
        
        # DINOv2 forward
        features = dinov2_model(img_dino)
        features = F.normalize(features, dim=-1)

        # 디버깅
        print(f"[DEBUG] features.shape: {features.shape}")
        print(f"[DEBUG] features.mean(): {features.mean():.4f}")
        print(f"[DEBUG] features.std(): {features.std():.4f}")
        print(f"[DEBUG] Per-sample norm: {torch.norm(features, p=2, dim=1).mean():.4f}")
        print(f"[DEBUG] Per-sample norm (min/max): {torch.norm(features, p=2, dim=1).min():.4f} / {torch.norm(features, p=2, dim=1).max():.4f}")
    
    return features


def add_noise_at_t(clean_image, noise, t):
    """
    Clean image에 시간 t에 해당하는 noise를 추가
    x_t = t * noise + (1 - t) * clean_image
    """
    return t * noise + (1 - t) * clean_image


def one_step_dino_prediction(dinov2_model, dataset, t1=0.9, t2=0.0,
                             num_samples=10000, batch_size=32, device="cuda"):
    """
    t1=0.9, t2=0.0에서 one-step으로 DINO embedding 예측
    predicted_dino = noise_y + model_output_y
    """
    noise_dim_size = 384
    
    total_batches = (num_samples + batch_size - 1) // batch_size
    all_mse_errors = []

    dinos =[]
    noises =[]

    images = []
    imagenoises = []
    
    for b in tqdm(range(total_batches), desc=f"One-step DINO prediction (t1={t1}, t2={t2})"):
        current_batch_size = min(batch_size, num_samples - b * batch_size)
        
        # 실제 CIFAR-10 이미지 샘플링
        indices = torch.randint(0, len(dataset), (current_batch_size,))
        clean_images = []
        for idx in indices:
            img, _ = dataset[idx]
            clean_images.append(img)
        clean_images = torch.stack(clean_images).to(device)
        
        # Noise 생성
        noise_x = torch.randn_like(clean_images, device=device)
        
        # t1 시점의 noisy image 생성
        t_tensor = torch.full((current_batch_size, 1, 1, 1), t1, device=device)
        noisy_images = add_noise_at_t(clean_images, noise_x, t_tensor)
        
        # t2=0일 때의 초기 noise vector
        noise_y = torch.randn(current_batch_size, noise_dim_size, device=device)
        
        t1_tensor = torch.full((current_batch_size,), t1, device=device)
        t2_tensor = torch.full((current_batch_size,), t2, device=device)
        y = torch.tensor(0, device=device).expand(current_batch_size)
        
        # 실제 DINO embedding 계산 (clean image로부터)
        actual_dino = compute_dinov2_embeddings(clean_images, dinov2_model, device)

        compute_dinov2_embeddings(clean_images, dinov2_model, device)
        
        dinos.append(actual_dino.cpu())
        noises.append(noise_y.cpu())
        images.append(clean_images.cpu())
        imagenoises.append(noise_x.cpu())

        if len(dinos) == 100:
            #visualize
            #distribution
            dinos_tensor = torch.cat(dinos, dim=0)
            noises_tensor = torch.cat(noises, dim=0)
            images_tensor = torch.cat(images, dim=0)
            imagenoises_tensor = torch.cat(imagenoises, dim=0)

            print(f"After { (b+1)*batch_size} samples:")
            print(f"DINO embedding - Mean: {dinos_tensor.mean():.4f}, Std: {dinos_tensor.std():.4f}, Norm: {torch.norm(dinos_tensor, p=2, dim=1).mean():.4f}")
            print(f"Noise_y - Mean: {noises_tensor.mean():.4f}, Std: {noises_tensor.std():.4f}, Norm: {torch.norm(noises_tensor, p=2, dim=1).mean():.4f}")
            print(f"Images - Mean: {images_tensor.mean():.4f}, Std: {images_tensor.std():.4f}, Norm: {torch.norm(images_tensor.reshape(images_tensor.size(0), -1), p=2, dim=1).mean():.4f}")
            print(f"Image Noises - Mean: {imagenoises_tensor.mean():.4f}, Std: {imagenoises_tensor.std():.4f}, Norm: {torch.norm(imagenoises_tensor.reshape(imagenoises_tensor.size(0), -1), p=2, dim=1).mean():.4f}")
        
    return 0



if __name__ == "__main__":
    device = "cuda:0"
    noise_dim_size = 384

    # Load CIFAR-10 dataset
    print("Loading CIFAR-10 dataset...")
    dataset = datasets.CIFAR10(
        root="./data/cifar10", 
        train=True, 
        download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
    )

    # Load DINOv2 model
    print("Loading DINOv2 model...")
    dinov2_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
    dinov2_model = dinov2_model.to(device)
    dinov2_model.eval()


    # One-step prediction: t1=0.9, t2=0.0
    results = one_step_dino_prediction(
        dinov2_model=dinov2_model,
        dataset=dataset,
        t1=0.9,
        t2=0.0,
        num_samples=10000,
        batch_size=32,
        device=device
    )
