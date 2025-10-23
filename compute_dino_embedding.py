"""
DINOv2를 사용하여 CIFAR10 클래스별 prototype embedding을 계산하고 저장하는 스크립트
"""

import os
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from absl import app, flags

FLAGS = flags.FLAGS

flags.DEFINE_string("output_path", "./prototypes/dinov2_cifar10_prototypes.pt", 
                    help="path to save computed prototypes")
flags.DEFINE_integer("num_samples_per_class", 1000, 
                     help="number of samples per class to compute prototype")
flags.DEFINE_string("device", "0", help="gpu device to use")


def compute_dinov2_class_prototypes(dataset, device, num_samples_per_class=1000):
    """
    DINOv2를 사용하여 각 클래스의 prototype embedding 계산
    
    Args:
        dataset: CIFAR10 dataset
        device: torch device
        num_samples_per_class: 각 클래스당 사용할 샘플 수
    
    Returns:
        prototypes: [10, 384] tensor
    """
    print("Loading DINOv2 model...")
    dinov2_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
    dinov2_model = dinov2_model.to(device)
    dinov2_model.eval()
    
    # DINOv2 transform
    dino_transform = transforms.Compose([
        transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    class_features = {i: [] for i in range(10)}
    
    print("Computing class prototypes...")
    with torch.no_grad():
        for idx, (img, label) in enumerate(dataset):
            if len(class_features[label]) >= num_samples_per_class:
                if all(len(class_features[i]) >= num_samples_per_class for i in range(10)):
                    break
                continue
            
            # CIFAR 이미지는 [-1, 1] normalized → [0, 1]로 변환
            img_normalized = (img + 1) / 2
            
            # 224x224로 resize
            img_resized = F.interpolate(
                img_normalized.unsqueeze(0),
                size=(224, 224),
                mode='bicubic',
                align_corners=False
            ).squeeze(0)
            
            # DINOv2 normalization 적용
            img_dino = dino_transform(img_resized)
            
            # Forward pass
            features = dinov2_model(img_dino.unsqueeze(0).to(device))
            class_features[label].append(features.cpu())
            
            # Progress
            if sum(len(v) for v in class_features.values()) % 500 == 0:
                print(f"Processed {sum(len(v) for v in class_features.values())} images...")
    
    # 각 클래스의 평균 계산
    prototypes = []
    for class_idx in range(10):
        features_list = class_features[class_idx]
        if len(features_list) == 0:
            print(f"Warning: No samples for class {class_idx}")
            prototypes.append(torch.randn(384))
        else:
            prototype = torch.stack(features_list).mean(dim=0).squeeze()
            prototypes.append(prototype)
            print(f"Class {class_idx}: {len(features_list)} samples, "
                  f"prototype norm: {torch.norm(prototype).item():.2f}")
    
    prototypes = torch.stack(prototypes)  # [10, 384]
    
    # Cleanup
    del dinov2_model
    torch.cuda.empty_cache()
    
    return prototypes


def main(argv):
    use_cuda = torch.cuda.is_available()
    device = torch.device(f"cuda:{FLAGS.device}" if use_cuda else "cpu")
    
    print(f"Using device: {device}")
    
    # CIFAR10 dataset 로드
    print("Loading CIFAR10 dataset...")
    dataset = datasets.CIFAR10(
        root="./data/cifar10", 
        train=True, 
        download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
    )
    
    # Prototype 계산
    prototypes = compute_dinov2_class_prototypes(
        dataset, 
        device, 
        num_samples_per_class=FLAGS.num_samples_per_class
    )
    
    # 저장 디렉토리 생성
    os.makedirs(os.path.dirname(FLAGS.output_path), exist_ok=True)
    
    # Prototype 저장
    print(f"\nSaving prototypes to {FLAGS.output_path}...")
    torch.save({
        'prototypes': prototypes,
        'num_classes': 10,
        'embedding_dim': 384,
        'num_samples_per_class': FLAGS.num_samples_per_class,
    }, FLAGS.output_path)
    
    print("Done! Prototypes saved successfully.")
    print(f"Prototype shape: {prototypes.shape}")


if __name__ == "__main__":
    app.run(main)