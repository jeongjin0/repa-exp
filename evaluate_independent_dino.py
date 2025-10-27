import os
import torch
from torchvision import datasets, transforms
from torchvision.models import resnet18
from models.dit import DiT_models
from models.dit_repa_independent import DiTZeroflowintegrated_independent_t
from tqdm import tqdm
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim


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


class DinoClassifier(nn.Module):
    """
    DINO embedding (384차원)을 입력받아 클래스를 예측하는 ResNet18 기반 classifier
    """
    def __init__(self, input_dim=384, num_classes=10):
        super().__init__()
        # ResNet18의 앞부분만 사용하거나, 간단한 MLP 사용
        # 여기서는 MLP로 구현
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        return self.classifier(x)


def train_dino_classifier(predicted_dinos, labels, num_epochs=50, batch_size=128, 
                         learning_rate=0.001, device="cuda"):
    """
    Predicted DINO embedding으로부터 label을 예측하는 classifier 학습
    
    Args:
        predicted_dinos: [N, 384] predicted DINO embeddings
        labels: [N] 실제 클래스 레이블
        num_epochs: 학습 에포크 수
        batch_size: 배치 크기
        learning_rate: 학습률
        device: 디바이스
    
    Returns:
        classifier: 학습된 classifier
        train_acc: 최종 train accuracy
    """
    num_samples = predicted_dinos.shape[0]
    dataset = torch.utils.data.TensorDataset(predicted_dinos, labels)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Classifier 생성
    classifier = DinoClassifier(input_dim=384, num_classes=10).to(device)
    optimizer = optim.Adam(classifier.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    
    print(f"\n{'='*60}")
    print(f"Training DINO Classifier")
    print(f"Dataset size: {num_samples}")
    print(f"Num epochs: {num_epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {learning_rate}")
    print(f"{'='*60}\n")
    
    classifier.train()
    for epoch in range(num_epochs):
        total_loss = 0
        correct = 0
        total = 0
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for batch_dinos, batch_labels in progress_bar:
            batch_dinos = batch_dinos.to(device)
            batch_labels = batch_labels.to(device)
            
            optimizer.zero_grad()
            outputs = classifier(batch_dinos)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += batch_labels.size(0)
            correct += predicted.eq(batch_labels).sum().item()
            
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100.*correct/total:.2f}%'
            })
        
        epoch_loss = total_loss / len(dataloader)
        epoch_acc = 100. * correct / total
        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%")
    
    # 최종 accuracy 계산
    classifier.eval()
    with torch.no_grad():
        all_outputs = []
        for batch_dinos, _ in torch.utils.data.DataLoader(dataset, batch_size=batch_size):
            batch_dinos = batch_dinos.to(device)
            outputs = classifier(batch_dinos)
            all_outputs.append(outputs.cpu())
        
        all_outputs = torch.cat(all_outputs, dim=0)
        _, predicted = all_outputs.max(1)
        train_acc = 100. * predicted.eq(labels.cpu()).sum().item() / num_samples
    
    print(f"\nFinal Training Accuracy: {train_acc:.2f}%\n")
    
    return classifier, train_acc


def one_step_dino_prediction_with_classifier(model, dinov2_model, dataset, t1=0.9, t2=0.0,
                                             num_samples=10000, batch_size=32, device="cuda"):
    """
    t1=0.9, t2=0.0에서 one-step으로 DINO embedding 예측
    predicted_dino = noise_y + model_output_y
    
    그리고 predicted DINO로부터 클래스를 예측하는 classifier를 학습
    """
    noise_dim_size = 384
    
    total_batches = (num_samples + batch_size - 1) // batch_size
    all_mse_errors = []
    all_predicted_dinos = []
    all_actual_dinos = []
    all_labels = []
    
    for b in tqdm(range(total_batches), desc=f"One-step DINO prediction (t1={t1}, t2={t2})"):
        current_batch_size = min(batch_size, num_samples - b * batch_size)
        
        # 실제 CIFAR-10 이미지 샘플링
        indices = torch.randint(0, len(dataset), (current_batch_size,))
        clean_images = []
        labels = []
        for idx in indices:
            img, label = dataset[idx]
            clean_images.append(img)
            labels.append(label)
        clean_images = torch.stack(clean_images).to(device)
        labels = torch.tensor(labels, device=device)
        
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
        
        # 저장
        all_predicted_dinos.append(predicted_dino.cpu())
        all_actual_dinos.append(actual_dino.cpu())
        all_labels.append(labels.cpu())
    
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
    
    # Classifier 학습
    all_predicted_dinos = torch.cat(all_predicted_dinos, dim=0)
    all_actual_dinos = torch.cat(all_actual_dinos, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    
    print(f"\n{'='*60}")
    print("Training classifier on PREDICTED DINO embeddings")
    print(f"{'='*60}")
    classifier_pred, train_acc_pred = train_dino_classifier(
        all_predicted_dinos, all_labels, 
        num_epochs=50, batch_size=128, learning_rate=0.001, device=device
    )
    
    # 비교를 위해 actual DINO embedding으로도 classifier 학습
    print(f"\n{'='*60}")
    print("Training classifier on ACTUAL DINO embeddings (baseline)")
    print(f"{'='*60}")
    classifier_actual, train_acc_actual = train_dino_classifier(
        all_actual_dinos, all_labels, 
        num_epochs=50, batch_size=128, learning_rate=0.001, device=device
    )
    
    print(f"\n{'='*60}")
    print("CLASSIFIER TRAINING RESULTS")
    print(f"{'='*60}")
    print(f"Predicted DINO → Accuracy: {train_acc_pred:.2f}%")
    print(f"Actual DINO → Accuracy: {train_acc_actual:.2f}%")
    print(f"Accuracy Gap: {train_acc_actual - train_acc_pred:.2f}%")
    print(f"{'='*60}\n")
    
    return {
        't1': t1,
        't2': t2,
        'mean_mse': mean_mse,
        'std_mse': std_mse,
        'min_mse': min_mse,
        'max_mse': max_mse,
        'all_mse': all_mse_errors,
        'train_acc_predicted': train_acc_pred,
        'train_acc_actual': train_acc_actual,
        'classifier_pred': classifier_pred,
        'classifier_actual': classifier_actual
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
    weight_path = 'exps2/independent_dino/checkpoints/0160000.pt'
    weight_path = 'exps2/exps/independent_dino/checkpoints/0160000.pt'
    
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

    # One-step prediction with classifier training
    results = one_step_dino_prediction_with_classifier(
        model=model,
        dinov2_model=dinov2_model,
        dataset=dataset,
        t1=0.9,
        t2=0.0,
        num_samples=10000,
        batch_size=32,
        device=device
    )