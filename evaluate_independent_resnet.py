import argparse
import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image, make_grid
import numpy as np
from tqdm import tqdm
import json

from models.dit import DiT_models
from models.dit_repa_independent import DiTZeroflowintegrated_independent_multitoken
from models.resnet import ResNet18


def load_resnet_encoder(checkpoint_path, device):
    """Load frozen ResNet encoder"""
    print(f"Loading ResNet from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    args = checkpoint['args']
    resnet = ResNet18(
        num_classes=args.num_classes,
        embedding_dim=384
    ).to(device)
    
    resnet.load_state_dict(checkpoint['resnet_state_dict'])
    resnet.eval()
    
    for param in resnet.parameters():
        param.requires_grad = False
    
    print(f"ResNet loaded from epoch {checkpoint['epoch']}, acc: {checkpoint.get('test_acc', 'N/A'):.2f}%")
    return resnet


def load_dit_model(checkpoint_path, device):
    """Load DiT model"""
    print(f"Loading DiT model from {checkpoint_path}...")
    
    original_model = DiT_models['DiT-S/2'](
        input_size=32,
        num_classes=10,
        in_channels=3,
        learn_sigma=False,
    ).to(device)
    
    model = DiTZeroflowintegrated_independent_multitoken(
        original_model, 
        noise_dim=384, 
        output_noise_dim=384
    ).to(device)
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Handle both 'net_model' and 'ema_model' keys
    if 'ema_model' in checkpoint:
        model.load_state_dict(checkpoint['ema_model'])
        print(f"Loaded EMA model from step {checkpoint.get('step', 'unknown')}")
    elif 'net_model' in checkpoint:
        model.load_state_dict(checkpoint['net_model'])
        print(f"Loaded model from step {checkpoint.get('step', 'unknown')}")
    else:
        raise KeyError("Checkpoint must contain 'ema_model' or 'net_model' key")
    
    model.eval()
    return model


def preprocess_for_resnet(x):
    """
    Preprocess for ResNet
    x: [-1, 1] range images
    """
    mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1).to(x.device)
    std = torch.tensor([0.2023, 0.1994, 0.2010]).view(1, 3, 1, 1).to(x.device)
    
    # [-1, 1] -> [0, 1]
    x = (x + 1) / 2.0
    
    # Normalize
    x = (x - mean) / std
    
    return x


def add_noise_image(x, t, device):
    """Add noise to image at timestep t"""
    noise = torch.randn_like(x)
    alpha_t = torch.sqrt(1 - t).view(-1, 1, 1, 1)
    sigma_t = torch.sqrt(t).view(-1, 1, 1, 1)
    x_noised = alpha_t * x + sigma_t * noise
    return x_noised


def add_noise_embedding(emb, t, device):
    """Add noise to embedding at timestep t"""
    noise = torch.randn_like(emb)
    alpha_t = torch.sqrt(1 - t).view(-1, 1)
    sigma_t = torch.sqrt(t).view(-1, 1)
    emb_noised = alpha_t * emb + sigma_t * noise
    return emb_noised


@torch.no_grad()
def generate_with_euler(model, x_start, emb_start, y, num_steps=100, device='cuda'):
    """
    Euler integration for joint generation
    
    Args:
        model: DiT model
        x_start: initial noised image (B, 3, 32, 32)
        emb_start: initial noised embedding (B, 384) or None for random
        y: class labels (B,)
        num_steps: number of integration steps
    
    Returns:
        x_final: generated image
        emb_final: generated embedding
    """
    dt = 1.0 / num_steps
    
    x = x_start.clone()
    emb = emb_start.clone()
    
    for i in range(num_steps):
        t = i / num_steps
        t_batch = torch.full((x.size(0),), t, device=device)
        
        # Predict velocity
        v_x, v_emb = model(x, t_batch, y=y, y_embedding=emb)
        
        # Euler step
        x = x + v_x * dt
        emb = emb + v_emb * dt
    
    return x, emb


@torch.no_grad()
def evaluate_generation(args):
    """Evaluate generation quality"""
    
    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    print(f"Using device: {device}")
    
    # Load models
    resnet = load_resnet_encoder(args.resnet_checkpoint, device)
    dit_model = load_dit_model(args.dit_checkpoint, device)
    
    # Load test dataset
    print("\nLoading CIFAR-10 test dataset...")
    test_dataset = datasets.CIFAR10(
        root="./data/cifar10",
        train=False,
        download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Metrics
    total_mse_embedding = 0.0
    total_mse_image = 0.0
    num_samples = 0
    
    all_generated_embeddings = []
    all_target_embeddings = []
    
    print(f"\nStarting evaluation...")
    print(f"Image noise level (t_image): {args.t_image}")
    print(f"Embedding starts from: {'pure noise (t=1.0)' if args.t_embedding == 1.0 else f't={args.t_embedding}'}")
    print(f"Number of integration steps: {args.num_steps}")
    print("-" * 60)
    
    # Evaluate
    for batch_idx, (images, labels) in enumerate(tqdm(test_loader, desc="Evaluating")):
        if batch_idx >= args.max_batches:
            break
        
        images = images.to(device)
        labels = labels.to(device)
        batch_size = images.size(0)
        
        # Get target embeddings from ResNet
        images_preprocessed = preprocess_for_resnet(images)
        _, target_embeddings = resnet(images_preprocessed, return_embedding=True)
        
        # Add noise to images (small noise)
        t_image = torch.full((batch_size,), args.t_image, device=device)
        images_noised = add_noise_image(images, t_image, device)
        
        # Start embeddings from noise (or partially noised)
        if args.t_embedding == 1.0:
            # Pure noise
            embeddings_start = torch.randn(batch_size, 384, device=device)
        else:
            # Partially noised target embeddings
            t_emb = torch.full((batch_size,), args.t_embedding, device=device)
            embeddings_start = add_noise_embedding(target_embeddings, t_emb, device)
        
        # Generate
        generated_images, generated_embeddings = generate_with_euler(
            dit_model,
            images_noised,
            embeddings_start,
            labels,
            num_steps=args.num_steps,
            device=device
        )
        
        # Compute metrics
        mse_embedding = F.mse_loss(generated_embeddings, target_embeddings, reduction='sum').item()
        mse_image = F.mse_loss(generated_images, images, reduction='sum').item()
        
        total_mse_embedding += mse_embedding
        total_mse_image += mse_image
        num_samples += batch_size
        
        # Store embeddings for further analysis
        all_generated_embeddings.append(generated_embeddings.cpu())
        all_target_embeddings.append(target_embeddings.cpu())
        
        # Save sample images (first batch only)
        if batch_idx == 0:
            # Original images
            original_grid = make_grid(
                (images[:16] + 1) / 2,  # [-1, 1] -> [0, 1]
                nrow=4,
                normalize=False
            )
            save_image(original_grid, os.path.join(args.output_dir, 'original.png'))
            
            # Noised images
            noised_grid = make_grid(
                (images_noised[:16] + 1) / 2,
                nrow=4,
                normalize=False
            )
            save_image(noised_grid, os.path.join(args.output_dir, 'noised_images.png'))
            
            # Generated images
            generated_grid = make_grid(
                (generated_images[:16].clamp(-1, 1) + 1) / 2,
                nrow=4,
                normalize=False
            )
            save_image(generated_grid, os.path.join(args.output_dir, 'generated.png'))
            
            # Comparison
            comparison = torch.cat([
                (images[:16] + 1) / 2,
                (images_noised[:16] + 1) / 2,
                (generated_images[:16].clamp(-1, 1) + 1) / 2
            ])
            comparison_grid = make_grid(comparison, nrow=16, normalize=False)
            save_image(comparison_grid, os.path.join(args.output_dir, 'comparison.png'))
    
    # Compute average metrics
    avg_mse_embedding = total_mse_embedding / num_samples
    avg_mse_image = total_mse_image / num_samples
    
    # Concatenate all embeddings
    all_generated_embeddings = torch.cat(all_generated_embeddings, dim=0).numpy()
    all_target_embeddings = torch.cat(all_target_embeddings, dim=0).numpy()
    
    # Compute per-dimension statistics
    per_dim_mse = np.mean((all_generated_embeddings - all_target_embeddings) ** 2, axis=0)
    
    # Compute cosine similarity
    generated_norm = all_generated_embeddings / (np.linalg.norm(all_generated_embeddings, axis=1, keepdims=True) + 1e-8)
    target_norm = all_target_embeddings / (np.linalg.norm(all_target_embeddings, axis=1, keepdims=True) + 1e-8)
    cosine_sim = np.sum(generated_norm * target_norm, axis=1)
    avg_cosine_sim = np.mean(cosine_sim)
    
    # Print results
    print("\n" + "=" * 60)
    print("Evaluation Results")
    print("=" * 60)
    print(f"Number of samples: {num_samples}")
    print(f"\nImage Metrics:")
    print(f"  MSE (image): {avg_mse_image:.6f}")
    print(f"\nEmbedding Metrics:")
    print(f"  MSE (embedding): {avg_mse_embedding:.6f}")
    print(f"  MSE per dimension (mean): {per_dim_mse.mean():.6f}")
    print(f"  MSE per dimension (std): {per_dim_mse.std():.6f}")
    print(f"  Cosine similarity: {avg_cosine_sim:.6f}")
    print(f"\nEmbedding Statistics:")
    print(f"  Generated mean: {all_generated_embeddings.mean():.6f}")
    print(f"  Generated std: {all_generated_embeddings.std():.6f}")
    print(f"  Target mean: {all_target_embeddings.mean():.6f}")
    print(f"  Target std: {all_target_embeddings.std():.6f}")
    print("=" * 60)
    
    # Save results to JSON
    results = {
        'num_samples': num_samples,
        'config': {
            't_image': args.t_image,
            't_embedding': args.t_embedding,
            'num_steps': args.num_steps,
        },
        'metrics': {
            'mse_image': float(avg_mse_image),
            'mse_embedding': float(avg_mse_embedding),
            'mse_per_dim_mean': float(per_dim_mse.mean()),
            'mse_per_dim_std': float(per_dim_mse.std()),
            'cosine_similarity': float(avg_cosine_sim),
        },
        'embedding_stats': {
            'generated_mean': float(all_generated_embeddings.mean()),
            'generated_std': float(all_generated_embeddings.std()),
            'target_mean': float(all_target_embeddings.mean()),
            'target_std': float(all_target_embeddings.std()),
        }
    }
    
    results_path = os.path.join(args.output_dir, 'evaluation_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"\nResults saved to {results_path}")
    
    # Save embeddings for further analysis
    np.save(os.path.join(args.output_dir, 'generated_embeddings.npy'), all_generated_embeddings)
    np.save(os.path.join(args.output_dir, 'target_embeddings.npy'), all_target_embeddings)
    print(f"Embeddings saved to {args.output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate DiT generation with ResNet embeddings')
    
    # Model checkpoints
    parser.add_argument('--resnet-checkpoint', type=str, default='checkpoint_epoch_50.pt',
                        help='Path to ResNet checkpoint')
    parser.add_argument('--dit-checkpoint', type=str, required=True,
                        help='Path to DiT checkpoint')
    
    # Evaluation settings
    parser.add_argument('--t-image', type=float, default=0.3,
                        help='Noise level for images (0=clean, 1=pure noise)')
    parser.add_argument('--t-embedding', type=float, default=1.0,
                        help='Noise level for embeddings (1.0=pure noise)')
    parser.add_argument('--num-steps', type=int, default=100,
                        help='Number of integration steps')
    
    # Data
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Batch size for evaluation')
    parser.add_argument('--max-batches', type=int, default=100,
                        help='Maximum number of batches to evaluate')
    
    # Output
    parser.add_argument('--output-dir', type=str, default='./eval_results',
                        help='Directory to save results')
    
    # Device
    parser.add_argument('--no-cuda', action='store_true',
                        help='Disable CUDA')
    
    args = parser.parse_args()
    
    evaluate_generation(args)


if __name__ == '__main__':
    main()