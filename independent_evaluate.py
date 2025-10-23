import os
import torch
from torchvision.utils import save_image
from torchvision import datasets, transforms
from models.dit import DiT_models
from models.dit_repa_independent import DiTZeroflowintegrated_independent_t
from tqdm import tqdm
import torch.nn.functional as F

# -------------------------
# Unified Euler Sampler (x와 y_t 함께 업데이트)
# -------------------------
def euler_sampler_unified(model, x0, t1_start, t2_start, steps, device="cuda", noise_dim_size=384):
    """
    t1_start에서 시작하여 x를 denoise하고
    t2_start에서 시작하여 y_t (DINO embedding)를 생성
    둘 다 같은 step으로 t=1까지 진행
    """
    dt1 = (1.0 - t1_start) / steps
    dt2 = (1.0 - t2_start) / steps
    
    x = x0.to(device)
    v_noise = torch.randn(x.size(0), noise_dim_size, device=device)
    y_t = v_noise

    for i in range(steps):
        t1_current = t1_start + i * dt1
        t2_current = t2_start + i * dt2
        
        t1 = torch.full((x.size(0),), t1_current, device=device)
        t2 = torch.full((x.size(0),), t2_current, device=device)
        y = torch.tensor(0, device=device).expand(x.size(0))

        with torch.no_grad():
            v, v_noise, _ = model(x, t1, t2, y=y, noise_vector=[y_t])

        x = x + v * dt1
        y_t = y_t + v_noise * dt2

    return x, y_t


def compute_dinov2_embeddings(images, dinov2_model, device):
    """
    생성된 이미지들의 실제 DINOv2 embedding 계산
    images: [B, 3, 32, 32], range [-1, 1]
    """
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
    
    return features


def add_noise_at_t(clean_image, noise, t):
    """
    Clean image에 시간 t에 해당하는 noise를 추가
    x_t = t * noise + (1 - t) * clean_image
    """
    return t * noise + (1 - t) * clean_image


# -------------------------
# Denoising MSE 평가
# -------------------------
def evaluate_denoising_mse(model, dinov2_model, dataset, t1_start=0.9, t2_start=0.0,
                          steps=50, num_samples=1000, batch_size=32, device="cuda"):
    """
    t1_start에서 denoising 시작, t2_start에서 DINO embedding 생성 시작
    """
    noise_dim_size = 384
    
    total_batches = (num_samples + batch_size - 1) // batch_size
    all_mse_errors = []
    all_image_mse_errors = []
    
    for b in tqdm(range(total_batches), desc=f"Evaluating t1={t1_start}, t2={t2_start}"):
        current_batch_size = min(batch_size, num_samples - b * batch_size)
        
        # 실제 CIFAR-10 이미지 샘플링
        indices = torch.randint(0, len(dataset), (current_batch_size,))
        clean_images = []
        for idx in indices:
            img, _ = dataset[idx]
            clean_images.append(img)
        clean_images = torch.stack(clean_images).to(device)
        
        # Noise 생성
        noise = torch.randn_like(clean_images, device=device)
        
        # t1_start 시점의 noisy image 생성
        t_tensor = torch.full((current_batch_size, 1, 1, 1), t1_start, device=device)
        noisy_images = add_noise_at_t(clean_images, noise, t_tensor)
        
        # Unified sampling: x는 t1_start에서, y_t는 t2_start에서 시작
        denoised_images, predicted_y_t = euler_sampler_unified(
            model, noisy_images, t1_start=t1_start, t2_start=t2_start,
            steps=steps, device=device, noise_dim_size=noise_dim_size
        )
        
        # 실제 DINO embedding 계산 (denoised image로부터)
        actual_dino = compute_dinov2_embeddings(denoised_images, dinov2_model, device)
        
        # DINO embedding MSE 계산
        dino_mse = F.mse_loss(predicted_y_t, actual_dino, reduction='none').mean(dim=1)
        all_mse_errors.append(dino_mse.cpu())
        
        # Image MSE 계산 (denoised vs clean)
        image_mse = F.mse_loss(denoised_images, clean_images, reduction='none').mean(dim=[1,2,3])
        all_image_mse_errors.append(image_mse.cpu())
    
    # DINO MSE 통계
    all_mse_errors = torch.cat(all_mse_errors, dim=0)
    mean_mse = all_mse_errors.mean().item()
    std_mse = all_mse_errors.std().item()
    min_mse = all_mse_errors.min().item()
    max_mse = all_mse_errors.max().item()
    
    # Image MSE 통계
    all_image_mse_errors = torch.cat(all_image_mse_errors, dim=0)
    mean_img_mse = all_image_mse_errors.mean().item()
    std_img_mse = all_image_mse_errors.std().item()
    
    print(f"\n=== Results for t1_start={t1_start}, t2_start={t2_start} ===")
    print(f"DINO Embedding MSE:")
    print(f"  Mean: {mean_mse:.6f}")
    print(f"  Std:  {std_mse:.6f}")
    print(f"  Min:  {min_mse:.6f}")
    print(f"  Max:  {max_mse:.6f}")
    print(f"\nImage Reconstruction MSE:")
    print(f"  Mean: {mean_img_mse:.6f}")
    print(f"  Std:  {std_img_mse:.6f}\n")
    
    return {
        't1_start': t1_start,
        't2_start': t2_start,
        'dino_mean_mse': mean_mse,
        'dino_std_mse': std_mse,
        'dino_min_mse': min_mse,
        'dino_max_mse': max_mse,
        'image_mean_mse': mean_img_mse,
        'image_std_mse': std_img_mse,
        'all_dino_mse': all_mse_errors,
        'all_image_mse': all_image_mse_errors
    }


# -------------------------
# 실행
# ------------------------- 
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

    # 160k checkpoint만 실험
    weight_path = 'exps/independent_dino/checkpoints/0160000.pt'
    
    print(f"\n{'='*60}")
    print(f"Loading weights from: {weight_path}")
    print(f"{'='*60}")

    original_dit_model = DiT_models['DiT-S/2'](
        input_size=32,
        num_classes=10,
        in_channels=3,
        learn_sigma=False,
    ).to(device)

    model = DiTZeroflowintegrated_independent_t(
        original_dit_model, 
        noise_dim=384, 
        output_noise_dim=384
    ).to(device)

    state_dict = torch.load(weight_path, map_location=device)
    ckpt = state_dict['ema_model']
    new_state_dict = {k.replace("module.", ""): v for k, v in ckpt.items()}

    model.load_state_dict(new_state_dict)
    model.eval()

    # t1=0.9, t2=0.0으로 실험
    results = evaluate_denoising_mse(
        model=model,
        dinov2_model=dinov2_model,
        dataset=dataset,
        t1_start=0.9,
        t2_start=0.0,
        steps=5,
        num_samples=10000,
        batch_size=32,
        device=device
    )

    # Save results
    torch.save(results, 'denoising_mse_t1_0.9_160k.pt')
    print(f"\nResults saved to 'denoising_mse_t1_0.9_160k.pt'")