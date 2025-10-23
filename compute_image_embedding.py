# compute_image_embeddings.py
"""
CIFAR10 전체 이미지에 대한 DINOv2 embedding을 미리 계산하고 저장
"""

import os
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from tqdm import tqdm

def compute_all_dinov2_embeddings(dataset, device, output_path):
    """
    모든 CIFAR10 이미지의 DINOv2 embedding 계산
    """
    print("Loading DINOv2 model...")
    dinov2_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
    dinov2_model = dinov2_model.to(device)
    dinov2_model.eval()
    
    embeddings_list = []
    labels_list = []
    
    print("Computing embeddings for all images...")
    with torch.no_grad():
        for idx in tqdm(range(len(dataset))):
            img, label = dataset[idx]
            
            # [-1, 1] → [0, 1]
            img_normalized = (img + 1) / 2
            
            # 224x224로 resize 및 normalize
            img_resized = F.interpolate(
                img_normalized.unsqueeze(0),
                size=(224, 224),
                mode='bicubic',
                align_corners=False
            )
            
            mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
            img_dino = (img_resized - mean) / std
            
            # DINOv2 forward
            features = dinov2_model(img_dino.to(device))
            embeddings_list.append(features.cpu())
            labels_list.append(label)
    
    embeddings = torch.cat(embeddings_list, dim=0)  # [N, 384]
    labels = torch.tensor(labels_list)
    
    # 저장
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save({
        'embeddings': embeddings,
        'labels': labels,
        'embedding_dim': 384,
    }, output_path)
    
    print(f"Saved {len(embeddings)} embeddings to {output_path}")
    print(f"Embedding shape: {embeddings.shape}")
    
    del dinov2_model
    torch.cuda.empty_cache()
    
    return embeddings, labels


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Train set
    train_dataset = datasets.CIFAR10(
        root="./data/cifar10", 
        train=True, 
        download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
    )
    
    compute_all_dinov2_embeddings(
        train_dataset, 
        device, 
        "./embeddings/dinov2_cifar10_train.pt"
    )