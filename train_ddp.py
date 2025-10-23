import argparse
import copy
from copy import deepcopy
import logging
import os
from pathlib import Path
from collections import OrderedDict
import json

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.cuda.amp import autocast, GradScaler

from models.sit import SiT_models
from models.dit import DiT_models
from loss import SILoss, FlowMatchingWithProjectionLoss
from utils import load_encoders

from dataset import CustomDataset
from diffusers.models import AutoencoderKL
import wandb
import math
from torchvision.utils import make_grid
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torchvision.transforms import Normalize

from torchvision.utils import save_image
from torchdyn.core import NeuralODE

CLIP_DEFAULT_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_DEFAULT_STD = (0.26862954, 0.26130258, 0.27577711)

def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    logging.basicConfig(
        level=logging.INFO,
        format='[\033[34m%(asctime)s\033[0m] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
    )
    logger = logging.getLogger(__name__)
    return logger


def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag


def preprocess_for_encoder(x, encoder_type='dinov2'):
    # [-1, 1] -> [0, 1]
    x = (x + 1) / 2.0
    
    if encoder_type == 'dinov2':
        # 224x224로 resize
        x = F.interpolate(x, size=224, mode='bicubic', align_corners=False)
        # ImageNet normalization
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(x.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(x.device)
        x = (x - mean) / std
    
    return x


def load_encoder(encoder_type='dinov2', device='cuda'):
    if encoder_type == 'dinov2':
        encoder = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
        encoder = encoder.to(device)
        encoder.eval()
        requires_grad(encoder, False)
    else:
        raise NotImplementedError(f"Encoder {encoder_type} not implemented")
    
    return encoder


@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        name = name.replace("module.", "")
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


@torch.no_grad()
def generate_samples_euler(model, parallel, savedir, step, device, num_steps=100, net_="normal"):
    """
    Simple Euler integration for sampling
    """
    model.eval()
    
    # DataParallel이면 .module 사용
    model_ = model.module if parallel else model
    
    # Start from noise
    x = torch.randn(64, 3, 32, 32, device=device)
    y = torch.zeros(64, dtype=torch.long, device=device)
    
    dt = 1.0 / num_steps
    
    for i in range(num_steps):
        t = i / num_steps
        t_batch = torch.full((64,), t, device=device)
        
        # Predict velocity
        vt = model_(x, t_batch, y=y)
        
        # Euler step: x_{t+dt} = x_t + vt * dt
        x = x + vt * dt
    
    # Clip and normalize
    samples = x.clip(-1, 1)
    samples = samples / 2 + 0.5  # [-1, 1] -> [0, 1]
    
    save_image(samples, f"{savedir}{net_}_generated_FM_images_step_{step}.png", nrow=8)
    
    model.train()
    return samples


#################################################################################
#                                  Training Loop                                #
#################################################################################

def main(args):
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpus = torch.cuda.device_count()
    
    # Create directories
    os.makedirs(args.output_dir, exist_ok=True)
    save_dir = os.path.join(args.output_dir, args.exp_name)
    os.makedirs(save_dir, exist_ok=True)
    
    # Save args
    args_dict = vars(args)
    json_dir = os.path.join(save_dir, "args.json")
    with open(json_dir, 'w') as f:
        json.dump(args_dict, f, indent=4)
    
    checkpoint_dir = f"{save_dir}/checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    logger = create_logger(save_dir)
    logger.info(f"Experiment directory created at {save_dir}")
    logger.info(f"Using {n_gpus} GPUs")
    logger.info(f"Mixed precision: {args.mixed_precision}")
    
    # Set seed
    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
    
    # Load encoder
    encoder = None
    if args.use_encoder:
        logger.info(f"Loading {args.encoder_type} encoder...")
        encoder = load_encoder(args.encoder_type, device)
        encoders = [encoder]
    else:
        encoders = []
    
    # Create model
    block_kwargs = {"fused_attn": args.fused_attn, "qk_norm": args.qk_norm}
    model = DiT_models['DiT-S/2'](
        input_size=32,
        num_classes=10,
        in_channels=3,
        learn_sigma=False,
    ).to(device)
    
    # DataParallel wrapper
    parallel = n_gpus > 1
    if parallel:
        model = torch.nn.DataParallel(model)
        logger.info(f"Using DataParallel with {n_gpus} GPUs")
    
    logger.info(f"DiT Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss function
    loss_fn = FlowMatchingWithProjectionLoss(
        encoders=encoders,
        accelerator=None,  # No accelerator needed
    )
    
    # Optimizer
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )
    
    # GradScaler for mixed precision
    use_amp = args.mixed_precision in ["fp16", "bf16"]
    dtype = torch.float16 if args.mixed_precision == "fp16" else torch.bfloat16
    scaler = GradScaler() if use_amp and args.mixed_precision == "fp16" else None
    
    if use_amp:
        logger.info(f"Using automatic mixed precision with {dtype}")
    
    # Setup data
    logger.info("Loading CIFAR10 dataset...")
    train_dataset = datasets.CIFAR10(
        root="./data/cifar10",
        train=True,
        download=True,
        transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
    )
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    logger.info(f"Dataset contains {len(train_dataset):,} images")
    
    # Create EMA
    ema = deepcopy(model.module if parallel else model)
    ema = ema.to(device)
    requires_grad(ema, False)
    ema.eval()
    update_ema(ema, model.module if parallel else model, decay=0)
    
    model.train()
    
    # Resume
    global_step = 0
    if args.resume_step > 0:
        ckpt_name = str(args.resume_step).zfill(7) + '.pt'
        ckpt = torch.load(
            f'{checkpoint_dir}/{ckpt_name}',
            map_location='cpu',
        )
        if parallel:
            model.module.load_state_dict(ckpt['net_model'])
        else:
            model.load_state_dict(ckpt['net_model'])
        ema.load_state_dict(ckpt['ema_model'])
        optimizer.load_state_dict(ckpt['optim'])
        if scaler is not None and 'scaler' in ckpt:
            scaler.load_state_dict(ckpt['scaler'])
        global_step = ckpt['step']
        logger.info(f"Resumed from step {global_step}")
    
    # WandB
    if args.report_to == "wandb":
        wandb.init(
            project="REPA",
            name=args.exp_name,
            config=args_dict
        )
    
    logger.info("Starting training!")
    
    # Training loop
    progress_bar = tqdm(range(0, args.max_train_steps), initial=global_step, desc="Steps")
    
    for epoch in range(args.epochs):
        model.train()
        for x1, y in train_dataloader:
            x1 = x1.to(device)
            y = y.to(device)
            
            # Extract encoder features
            zs = None
            if args.use_encoder and args.proj_coeff > 0:
                with torch.no_grad():
                    x1_preprocessed = preprocess_for_encoder(x1, args.encoder_type)
                    z = encoder.forward_features(x1_preprocessed)
                    # DINOv2: remove CLS token
                    if args.encoder_type == 'dinov2':
                        z = z['x_norm_patchtokens']
                    zs = [z]
            
            # Forward pass with mixed precision
            optimizer.zero_grad()
            
            if use_amp:
                # Mixed precision training
                with torch.amp.autocast('cuda', dtype=dtype):  # Fix deprecation warning
                    model_kwargs = dict(y=y)
                    flow_loss, proj_loss = loss_fn(model, x1, model_kwargs, zs=zs)
                    loss = flow_loss + args.proj_coeff * proj_loss

                # Backward with scaling (only for fp16)
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    # bfloat16 doesn't need scaling
                    loss.backward()
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    optimizer.step()
            else:
                # Standard FP32 training
                model_kwargs = dict(y=y)
                flow_loss, proj_loss = loss_fn(model, x1, model_kwargs, zs=zs)
                loss = flow_loss + args.proj_coeff * proj_loss
                
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
            
            # Update EMA
            update_ema(ema, model.module if parallel else model)
            
            # Logging
            global_step += 1
            progress_bar.update(1)
            
            logs = {
                "loss": loss.item(),
                "flow_loss": flow_loss.mean().item(),
                "grad_norm": grad_norm.item(),
            }
            if proj_loss != 0:
                logs["proj_loss"] = proj_loss.item()
            
            # Log scale if using fp16
            if scaler is not None:
                logs["scale"] = scaler.get_scale()
            
            progress_bar.set_postfix(**logs)
            
            if args.report_to == "wandb":
                wandb.log(logs, step=global_step)
            
            # Sampling
            if global_step % args.sampling_steps == 0 and global_step > 0:
                logger.info(f"Generating samples at step {global_step}")
                
                # Normal model
                samples_normal = generate_samples_euler(
                    model, parallel, save_dir + "/", global_step, device,
                    num_steps=100, net_="normal"
                )
                
                # EMA model
                samples_ema = generate_samples_euler(
                    ema, False, save_dir + "/", global_step, device,
                    num_steps=100, net_="ema"
                )
                
                if args.report_to == "wandb":
                    wandb.log({
                        "samples/normal": wandb.Image(
                            f"{save_dir}/normal_generated_FM_images_step_{global_step}.png"
                        ),
                        "samples/ema": wandb.Image(
                            f"{save_dir}/ema_generated_FM_images_step_{global_step}.png"
                        )
                    }, step=global_step)
            
            # Checkpointing
            if global_step % args.checkpointing_steps == 0 and global_step > 0:
                checkpoint = {
                    "net_model": (model.module.state_dict() if parallel 
                                 else model.state_dict()),
                    "ema_model": ema.state_dict(),
                    "optim": optimizer.state_dict(),
                    "step": global_step,
                }
                if scaler is not None:
                    checkpoint["scaler"] = scaler.state_dict()
                
                checkpoint_path = f"{checkpoint_dir}/{global_step:07d}.pt"
                torch.save(checkpoint, checkpoint_path)
                logger.info(f"Saved checkpoint to {checkpoint_path}")
            
            if global_step >= args.max_train_steps:
                break
        
        if global_step >= args.max_train_steps:
            break
    
    logger.info("Training completed!")
    if args.report_to == "wandb":
        wandb.finish()


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Training")

    # logging:
    parser.add_argument("--output-dir", type=str, default="exps")
    parser.add_argument("--exp-name", type=str, required=True)
    parser.add_argument("--logging-dir", type=str, default="logs")
    parser.add_argument("--report-to", type=str, default=None)
    parser.add_argument("--sampling-steps", type=int, default=10000)
    parser.add_argument("--resume-step", type=int, default=0)

    # model
    parser.add_argument("--model", type=str)
    parser.add_argument("--num-classes", type=int, default=10)
    parser.add_argument("--encoder-depth", type=int, default=8)
    parser.add_argument("--fused-attn", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--qk-norm", action=argparse.BooleanOptionalAction, default=False)

    # dataset
    parser.add_argument("--data-dir", type=str, default="../data/imagenet256")
    parser.add_argument("--resolution", type=int, choices=[32], default=32)
    parser.add_argument("--batch-size", type=int, default=256)

    # precision
    parser.add_argument("--allow-tf32", action="store_true")
    parser.add_argument("--mixed-precision", type=str, default="no", choices=["no", "fp16", "bf16"])

    # optimization
    parser.add_argument("--epochs", type=int, default=1400)
    parser.add_argument("--max-train-steps", type=int, default=400000)
    parser.add_argument("--checkpointing-steps", type=int, default=20000)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--adam-beta1", type=float, default=0.9)
    parser.add_argument("--adam-beta2", type=float, default=0.999)
    parser.add_argument("--adam-weight-decay", type=float, default=0.)
    parser.add_argument("--adam-epsilon", type=float, default=1e-08)
    parser.add_argument("--max-grad-norm", default=1.0, type=float)

    # seed
    parser.add_argument("--seed", type=int, default=0)

    # cpu
    parser.add_argument("--num-workers", type=int, default=4)

    # loss
    parser.add_argument("--path-type", type=str, default="linear", choices=["linear", "cosine"])
    parser.add_argument("--prediction", type=str, default="v", choices=["v"])
    parser.add_argument("--cfg-prob", type=float, default=0.1)
    parser.add_argument("--enc-type", type=str, default='dinov2-vit-b')
    parser.add_argument("--proj-coeff", type=float, default=0.5)
    parser.add_argument("--weighting", default="uniform", type=str)
    parser.add_argument("--legacy", action=argparse.BooleanOptionalAction, default=False)

    parser.add_argument("--use-encoder", action="store_true")
    parser.add_argument("--encoder-type", type=str, default="dinov2",
                       choices=["dinov2", "dino", "clip"])

    parser.add_argument("--num-sampling-steps", type=int, default=100)

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()
        
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)