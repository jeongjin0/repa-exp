import os
import torch
from torchvision.utils import save_image
from models.dit import DiT_models
from models.dit_repa_independent import DiTZeroflowintegrated_independent_t, DiTZeroflowintegrated_independent_multitoken
from tqdm import tqdm
import re

# -------------------------
# Euler 기반 ODE Solver
# -------------------------
def euler_sampler_iterative_y(model, x0, steps=10, device="cuda"):
    dt = 1.0 / steps
    x = x0.to(device)
    v_noise = torch.randn(x.size(0), noise_dim_size, device=device)
    y_t = v_noise

    for i in range(steps):
        t1 = torch.full((x.size(0),), i * dt, device=device)
        t2 = torch.full((x.size(0),), i * dt, device=device)

        y = torch.tensor(0, device=device).expand(x.size(0))

        t1_ = t1.view(-1, 1)
        t2_ = t2.view(-1, 1)
        noise_vector = torch.randn(x.size(0), noise_dim_size, device=device)

        #y_t = [t2_ * v_noise + (1 - t2_) * noise_vector]

        with torch.no_grad():
            v, v_noise, _ = model(x, t1, t2, y=y, noise_vector=[y_t])

        x = x + v * dt  # Euler step
        y_t = y_t + v_noise * dt

    return x, y_t

def euler_sampler_zerot2(model, x0, steps=10, device="cuda"):
    dt = 1.0 / steps
    x = x0.to(device)
    v_noise = torch.randn(x.size(0), noise_dim_size, device=device)
    y_t = [v_noise]

    for i in range(steps):
        t1 = torch.full((x.size(0),), i * dt, device=device)
        t2 = torch.zeros((x.size(0),), device=device)

        y = torch.tensor(0, device=device).expand(x.size(0))

        t1_ = t1.view(-1, 1)
        t2_ = t2.view(-1, 1)
        noise_vector = torch.randn(x.size(0), noise_dim_size, device=device)


        with torch.no_grad():
            v, v_noise, _ = model(x, t1, t2, y=y, noise_vector=[noise_vector])

        x = x + v * dt  # Euler step

    return x, y_t
    


# -------------------------
# 생성 코드 (batch 단위)
# -------------------------
def generate_images(model, num_samples=64, save_folder="folder2", device="cuda", batch_size=64):
    os.makedirs(save_folder, exist_ok=True)
    total_batches = (num_samples + batch_size - 1) // batch_size
    idx_start = 0

    for b in tqdm(range(total_batches), desc="Generating batches"):
        current_batch_size = min(batch_size, num_samples - idx_start)
        noise = torch.randn(current_batch_size, 3, 32, 32, device=device)
        samples, _ = euler_sampler_iterative_y(model, noise, steps=50, device=device)
        samples = (samples.clamp(-1, 1) + 1) / 2  # [-1,1] → [0,1]

        for i, img in enumerate(samples):
            save_path = os.path.join(save_folder, f"{idx_start + i}.jpg")
            save_image(img, save_path)

        idx_start += current_batch_size

    print(f"총 {num_samples}개의 생성 이미지가 {save_folder}에 저장됨")


# -------------------------
# 실행
# ------------------------- 
if __name__ == "__main__":
    device = "cuda:0"

    weight_list = [f'exps/repa_noencoder/checkpoints/0{step}0000.pt' for step in [ '04']]

    for weight_path in weight_list:
        print(f"Loading weights from: {weight_path}")

        original_dit_model = DiT_models['DiT-S/2'](
            input_size=32,
            num_classes=10,
            in_channels=3,
            learn_sigma=False,
        ).to(device)
        model = original_dit_model

        noise_dim_size = 384
        #noise_dim_size = 1024
        #model = DiTZeroflowintegrated(original_dit_model, noise_dim=noise_dim_size, output_noise_dim=noise_dim_size).to(device)
        #model = DiTZeroflow(original_dit_model, noise_dim=noise_dim_size, output_noise_dim=10).to(device)
        model = DiTZeroflowintegrated_independent_t(original_dit_model, noise_dim=384, output_noise_dim=384).to(device)
        model = DiTZeroflowintegrated_independent_multitoken(original_dit_model, noise_dim=384, output_noise_dim=384).to(device)


        state_dict = torch.load(weight_path, map_location=device)
        ckpt = state_dict['ema_model']
        new_state_dict = {k.replace("module.", ""): v for k, v in ckpt.items()}

        model.load_state_dict(new_state_dict)
        model.to(device)
        model.eval()

        name = re.search(r"exps/([^/]+)/", weight_path).group(1)
        save_folder = os.path.join(name, '_generated_images')
        save_folder = os.path.join('generated_images', save_folder)
        generate_images(model, num_samples=10000, device=device, batch_size=32, save_folder=save_folder)