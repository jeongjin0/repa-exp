import torch

# 사용 가능한 모든 GPU 확인
for i in range(torch.cuda.device_count()):
    print(f"Trying GPU {i}: {torch.cuda.get_device_name(i)}")
    try:
        with torch.cuda.device(i):
            t = torch.tensor([1, 2]).cuda(i)
            print(f"✓ GPU {i} works!")
            break
    except Exception as e:
        print(f"✗ GPU {i} failed: {e}")
