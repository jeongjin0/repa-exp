import os
import torch
from torchvision import datasets, transforms
from models.dit import DiT_models
from models.dit_repa_independent import DiTZeroflowintegrated_independent_t
from tqdm import tqdm
import torch.nn.functional as F


def compute_dinov2_embeddings(images, dinov2_model, device):
    """
    이미지들의 실제 DINOv2 embedding 계산
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


def one_step_dino_prediction(model, dinov2_model, dataset, t1=0.9, t2=0.0,
                             num_samples=10000, batch_size=32, device="cuda"):
    """
    t1=0.9, t2=0.0에서 one-step으로 DINO embedding 예측
    predicted_dino = noise_y + model_output_y
    """
    noise_dim_size = 384
    
    total_batches = (num_samples + batch_size - 1) // batch_size
    all_mse_errors = []
    
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
        
        # One-step model forward
        t1_tensor = torch.full((current_batch_size,), t1, device=device)
        t2_tensor = torch.full((current_batch_size,), t2, device=device)
        y = torch.tensor(0, device=device).expand(current_batch_size)
        
        with torch.no_grad():
            _, model_output_y, _ = model(noisy_images, t1_tensor, t2_tensor, 
                                         y=y, noise_vector=[noise_y])
        
        # predicted_dino = noise_y + model_output_y
        predicted_dino = noise_y + model_output_y
        
        # 실제 DINO embedding 계산 (clean image로부터)
        actual_dino = compute_dinov2_embeddings(clean_images, dinov2_model, device)
        
        # MSE 계산
        mse = F.mse_loss(predicted_dino, actual_dino, reduction='none').mean(dim=1)
        all_mse_errors.append(mse.cpu())
    
    # 통계 계산
    all_mse_errors = torch.cat(all_mse_errors, dim=0)
    mean_mse = all_mse_errors.mean().item()
    std_mse = all_mse_errors.std().item()
    min_mse = all_mse_errors.min().item()
    max_mse = all_mse_errors.max().item()
    
    print(f"\n=== One-Step DINO Prediction (t1={t1}, t2={t2}) ===")
    print(f"Mean MSE: {mean_mse:.6f}")
    print(f"Std MSE:  {std_mse:.6f}")
    print(f"Min MSE:  {min_mse:.6f}")
    print(f"Max MSE:  {max_mse:.6f}\n")
    
    return {
        't1': t1,
        't2': t2,
        'mean_mse': mean_mse,
        'std_mse': std_mse,
        'min_mse': min_mse,
        'max_mse': max_mse,
        'all_mse': all_mse_errors
    }


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

    # 160k checkpoint 실험
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

    # One-step prediction: t1=0.9, t2=0.0
    results = one_step_dino_prediction(
        model=model,
        dinov2_model=dinov2_model,
        dataset=dataset,
        t1=0.9,
        t2=0.0,
        num_samples=10000,
        batch_size=32,
        device=device
    )

    # Save results
    torch.save(results, 'onestep_dino_mse_t1_0.9_t2_0.0_160k.pt')
    print(f"\nResults saved to 'onestep_dino_mse_t1_0.9_t2_0.0_160k.pt'")