from torchvision.datasets import CIFAR10
from torchvision.utils import save_image
from torchvision import transforms
import os

save_dir = "cifar10_images"
os.makedirs(save_dir, exist_ok=True)

dataset = CIFAR10(root=".", train=True, download=True, transform=transforms.ToTensor())

for i, (img, _) in enumerate(dataset):
    save_image(img, f"{save_dir}/{i:05d}.jpg")
