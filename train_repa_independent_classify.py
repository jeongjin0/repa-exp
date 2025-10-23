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

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed

from models.dit import DiT_models
from models.dit_repa_independent import DiTZeroflowintegrated_independent_t
from loss import SILoss, IndependentFlowMatchingWithProjectionLoss
from utils import load_encoders

from dataset import CustomDataset
from diffusers.models import AutoencoderKL
# import wandb_utils
import wandb
import math
from torchvision.utils import make_grid
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torchvision.transforms import Normalize

from torchvision.utils import save_image
from torchdyn.core import NeuralODE

logger = get_logger(__name__)

CLIP_DEFAULT_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_DEFAULT_STD = (0.26862954, 0.26130258, 0.27577711)

def preprocess_raw_image(x, enc_type):
    resolution = x.shape[-1]
    if 'clip' in enc_type:
        x = x / 255.
        x = torch.nn.functional.interpolate(x, 224 * (resolution // 256), mode='bicubic')
        x = Normalize(CLIP_DEFAULT_MEAN, CLIP_DEFAULT_STD)(x)
    elif 'mocov3' in enc_type or 'mae' in enc_type:
        x = x / 255.
        x = Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)(x)
    elif 'dinov2' in enc_type:
        x = x / 255.
        x = Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)(x)
        x = torch.nn.functional.interpolate(x, 224 * (resolution // 256), mode='bicubic')
    elif 'dinov1' in enc_type:
        x = x / 255.
        x = Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)(x)
    elif 'jepa' in enc_type:
        x = x / 255.
        x = Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)(x)
        x = torch.nn.functional.interpolate(x, 224 * (resolution // 256), mode='bicubic')

    return x


def array2grid(x):
    nrow = round(math.sqrt(x.size(0)))
    x = make_grid(x.clamp(0, 1), nrow=nrow, value_range=(0, 1))
    x = x.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    return x


@torch.no_grad()
def sample_posterior(moments, latents_scale=1., latents_bias=0.):
    device = moments.device
    
    mean, std = torch.chunk(moments, 2, dim=1)
    z = mean + std * torch.randn_like(mean)
    z = (z * latents_scale + latents_bias) 
    return z 


@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        name = name.replace("module.", "")
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


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
def generate_samples_euler(model, parallel, savedir, step, device, num_steps=100, net_="normal"):
    """
    Simple Euler integration for sampling
    """
    model.eval()
    
    model_ = model.module if parallel else model
    
    # Start from noise
    x = torch.randn(64, 3, 32, 32, device=device)
    y = torch.zeros(64, dtype=torch.long, device=device)
    
    dt = 1.0 / num_steps
    
    for i in range(num_steps):
        t = i / num_steps
        t_batch = torch.full((64,), t, device=device)
        
        # Predict velocity
        vt = model_(x, t_batch, y=y, return_features=False)
        
        # Euler step: x_{t+dt} = x_t + vt * dt
        x = x + vt * dt
    
    # Clip and normalize
    samples = x.clip(-1, 1)
    samples = samples / 2 + 0.5  # [-1, 1] -> [0, 1]
    
    save_image(samples, f"{savedir}{net_}_generated_FM_images_step_{step}.png", nrow=8)
    
    model.train()
    return samples


def load_prototypes(prototype_path, device):
    if not os.path.exists(prototype_path):
        raise FileNotFoundError(
            f"Prototype file not found at {prototype_path}. "
            f"Please run compute_prototypes.py first to generate prototypes."
        )
    
    print(f"Loading prototypes from {prototype_path}...")
    checkpoint = torch.load(prototype_path, map_location=device)
    prototypes = checkpoint['prototypes'].to(device)
    
    return prototypes



#################################################################################
#                                  Training Loop                                #
#################################################################################

def main(args):    
    # set accelerator
    logging_dir = Path(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(
        project_dir=args.output_dir, logging_dir=logging_dir
        )

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)  # Make results folder (holds all experiment subfolders)
        save_dir = os.path.join(args.output_dir, args.exp_name)
        os.makedirs(save_dir, exist_ok=True)
        args_dict = vars(args)
        # Save to a JSON file
        json_dir = os.path.join(save_dir, "args.json")
        with open(json_dir, 'w') as f:
            json.dump(args_dict, f, indent=4)
        checkpoint_dir = f"{save_dir}/checkpoints"  # Stores saved model checkpoints
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(save_dir)
        logger.info(f"Experiment directory created at {save_dir}")
    device = accelerator.device
    if torch.backends.mps.is_available():
        accelerator.native_amp = False    
    if args.seed is not None:
        set_seed(args.seed + accelerator.process_index)
    

    # if args.enc_type != None:
    #     encoders, encoder_types, architectures = load_encoders(
    #         args.enc_type, device, args.resolution
    #         )
    # else:
    #     raise NotImplementedError()
    #z_dims = [encoder.embed_dim for encoder in encoders] if args.enc_type != 'None' else [0]
    block_kwargs = {"fused_attn": args.fused_attn, "qk_norm": args.qk_norm}

    if args.use_encoder:
        encoder = load_encoder(args.encoder_type, device)
        encoders = [encoder]
    else:
        encoders = []

    # model = SiT_models[args.model](
    #     input_size=latent_size,
    #     num_classes=args.num_classes,
    #     use_cfg = (args.cfg_prob > 0),
    #     z_dims = z_dims,
    #     encoder_depth=args.encoder_depth,
    #     **block_kwargs
    # )
    original_model = DiT_models['DiT-S/2'](
        input_size=32,
        num_classes=10,
        in_channels=3,
        learn_sigma=False,
    ).to(device)
    model = DiTZeroflowintegrated_independent_t(original_model, noise_dim=384, output_noise_dim=384).to(device)

    model = model.to(device)
    ema = deepcopy(model).to(device)  # Create an EMA of the model for use after training
    requires_grad(ema, False)
    
    latents_scale = torch.tensor(
        [0.18215, 0.18215, 0.18215, 0.18215]
        ).view(1, 4, 1, 1).to(device)
    latents_bias = torch.tensor(
        [0., 0., 0., 0.]
        ).view(1, 4, 1, 1).to(device)

    # create loss function
    loss_fn = SILoss(
        prediction=args.prediction,
        path_type=args.path_type, 
        encoders=encoders,
        accelerator=accelerator,
        latents_scale=latents_scale,
        latents_bias=latents_bias,
        weighting=args.weighting
    )

    loss_fn = IndependentFlowMatchingWithProjectionLoss(
        encoders=encoders,
        accelerator=accelerator,
    )

    if accelerator.is_main_process:
        logger.info(f"Model Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Setup optimizer (we used default Adam betas=(0.9, 0.999) and a constant learning rate of 1e-4 in our paper):
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
    
    # Setup data:
    # train_dataset = CustomDataset(args.data_dir)
    # local_batch_size = int(args.batch_size // accelerator.num_processes)
    # train_dataloader = DataLoader(
    #     train_dataset,
    #     batch_size=local_batch_size,
    #     shuffle=True,
    #     num_workers=args.num_workers,
    #     pin_memory=True,
    #     drop_last=True
    # )


    train_dataset = datasets.CIFAR10(root="./data/cifar10", train=True, download=True,
        transform=transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ]),)
    local_batch_size = int(args.batch_size // accelerator.num_processes)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=local_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )


    if accelerator.is_main_process:
        logger.info(f"Dataset contains {len(train_dataset):,} images ({args.data_dir})")
    
    # Prepare models for training:
    update_ema(ema, model, decay=0)  # Ensure EMA is initialized with synced weights
    model.train()  # important! This enables embedding dropout for classifier-free guidance
    ema.eval()  # EMA model should always be in eval mode
    
    # resume:
    global_step = 0
    if args.resume_step > 0:
        ckpt_name = str(args.resume_step).zfill(7) +'.pt'
        ckpt = torch.load(
            f'{os.path.join(args.output_dir, args.exp_name)}/checkpoints/{ckpt_name}',
            map_location='cpu',
            )
        model.load_state_dict(ckpt['model'])
        ema.load_state_dict(ckpt['ema'])
        optimizer.load_state_dict(ckpt['opt'])
        global_step = ckpt['steps']

    prototypes = load_prototypes(args.prototype_path, device)
    model.label_embedder.weight.data = prototypes


    model, optimizer, train_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader
    )

    if accelerator.is_main_process:
        tracker_config = vars(copy.deepcopy(args))
        accelerator.init_trackers(
            project_name="REPA", 
            config=tracker_config,
            init_kwargs={
                "wandb": {"name": f"{args.exp_name}"}
            },
        )
        
    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )


    for epoch in range(args.epochs):
        model.train()
        for x1, y in train_dataloader:
            x1 = x1.to(device)  # (B, 3, 32, 32)
            y = y.to(device)
            
            # Extract encoder features (if using)
            zs = None
            if args.use_encoder and args.proj_coeff > 0:
                with torch.no_grad():
                    x1_preprocessed = preprocess_for_encoder(x1, args.encoder_type)
                    z = encoder[0].forward_features(x1_preprocessed)
                    # DINOv2: remove CLS token
                    if args.encoder_type == 'dinov2':
                        z = z[:, 1:]  # (B, 256, 384) for ViT-S
                    zs = [z]
            
            with accelerator.accumulate(model):
                model_kwargs = dict(y=y)
                
                # Forward with features
                if args.use_encoder and args.proj_coeff > 0:
                    model.module.use_encoder_features = True if hasattr(model, 'module') \
                                                        else model.use_encoder_features
                
                flow_loss, proj_loss, loss_y = loss_fn(model, x1, y, device, model_kwargs, zs=zs)
                
                # Total loss
                loss = flow_loss + args.proj_coeff * proj_loss + args.y_coeff * loss_y
                
                # Optimization
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    grad_norm = accelerator.clip_grad_norm_(
                        model.parameters(), args.max_grad_norm
                    )
                optimizer.step()
                optimizer.zero_grad()
                
                if accelerator.sync_gradients:
                    update_ema(ema, model)
                
            with torch.no_grad():         
                all_labels = torch.arange(args.num_classes, device=device)
                all_embeddings = model.module.label_embedder(all_labels)

                distances = torch.cdist(vt_noise, all_embeddings)  # [B, num_classes]
                predict_logits = -distances  # 거리 negative해서 가까울수록 큰 값

                predicted_labels = torch.argmax(predict_logits, dim=1)
                acc = (predicted_labels == y).float().mean().item()
                
                acc_list.append(acc)
            
            # Logging
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
            
            logs = {
                "loss": accelerator.gather(loss).mean().item(),
                "flow_loss": accelerator.gather(flow_loss.mean()).mean().item(),
                "loss_y": accelerator.gather(loss_y.mean()).mean().item(),
            }
            if proj_loss != 0:
                logs["proj_loss"] = accelerator.gather(proj_loss).mean().item()
            if accelerator.sync_gradients:
                logs["grad_norm"] = accelerator.gather(grad_norm).mean().item()
            
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
            

            if global_step % args.sampling_steps == 0 and global_step > 0:
                if accelerator.is_main_process:
                    # Normal model sampling
                    samples_normal = generate_samples_euler(
                        model, 
                        FLAGS.parallel if 'FLAGS' in globals() else False,
                        save_dir + "/",
                        global_step,
                        device,
                        num_steps=100,
                        net_="normal"
                    )
                    
                    # EMA model sampling
                    samples_ema = generate_samples_euler(
                        ema,
                        FLAGS.parallel if 'FLAGS' in globals() else False,
                        save_dir + "/",
                        global_step,
                        device,
                        num_steps=100,
                        net_="ema"
                    )
                    
                    # Log to wandb
                    accelerator.log({
                        "samples/normal": wandb.Image(
                            f"{save_dir}/normal_generated_FM_images_step_{global_step}.png"
                        ),
                        "samples/ema": wandb.Image(
                            f"{save_dir}/ema_generated_FM_images_step_{global_step}.png"
                        )
                    }, step=global_step)
                    
                    logger.info(f"Generated samples at step {global_step}")
            
            # ===== Checkpointing =====
            if global_step % args.checkpointing_steps == 0 and global_step > 0:
                if accelerator.is_main_process:
                    checkpoint = {
                        "net_model": (model.module.state_dict() if hasattr(model, 'module')
                                     else model.state_dict()),
                        "ema_model": ema.state_dict(),
                        "optim": optimizer.state_dict(),
                        "step": global_step,
                    }
                    checkpoint_path = f"{checkpoint_dir}/{global_step:07d}.pt"
                    torch.save(checkpoint, checkpoint_path)
                    logger.info(f"Saved checkpoint to {checkpoint_path}")
            
            if global_step >= args.max_train_steps:
                break
        
        if global_step >= args.max_train_steps:
            break
    
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        logger.info("Done!")
    accelerator.end_training()


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Training")

    # logging:
    parser.add_argument("--output-dir", type=str, default="exps")
    parser.add_argument("--exp-name", type=str, required=True)
    parser.add_argument("--logging-dir", type=str, default="logs")
    parser.add_argument("--report-to", type=str, default="wandb")
    parser.add_argument("--sampling-steps", type=int, default=10000)
    parser.add_argument("--resume-step", type=int, default=0)

    # model
    parser.add_argument("--model", type=str)
    parser.add_argument("--num-classes", type=int, default=10)
    parser.add_argument("--encoder-depth", type=int, default=8)
    parser.add_argument("--fused-attn", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--qk-norm",  action=argparse.BooleanOptionalAction, default=False)

    # dataset
    parser.add_argument("--data-dir", type=str, default="../data/imagenet256")
    parser.add_argument("--resolution", type=int, choices=[32], default=32)
    parser.add_argument("--batch-size", type=int, default=256)

    # precision
    parser.add_argument("--allow-tf32", action="store_true")
    parser.add_argument("--mixed-precision", type=str, default="fp16", choices=["no", "fp16", "bf16"])

    # optimization
    parser.add_argument("--epochs", type=int, default=1400)
    parser.add_argument("--max-train-steps", type=int, default=400000)
    parser.add_argument("--checkpointing-steps", type=int, default=20000)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--adam-beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam-beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam-weight-decay", type=float, default=0., help="Weight decay to use.")
    parser.add_argument("--adam-epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max-grad-norm", default=1.0, type=float, help="Max gradient norm.")

    # seed
    parser.add_argument("--seed", type=int, default=0)

    # cpu
    parser.add_argument("--num-workers", type=int, default=4)

    # loss
    parser.add_argument("--path-type", type=str, default="linear", choices=["linear", "cosine"])
    parser.add_argument("--prediction", type=str, default="v", choices=["v"]) # currently we only support v-prediction
    parser.add_argument("--cfg-prob", type=float, default=0.1)
    parser.add_argument("--enc-type", type=str, default='dinov2-vit-b')
    parser.add_argument("--proj-coeff", type=float, default=0.5)
    parser.add_argument("--y-coeff", type=float, default=0.01)
    parser.add_argument("--weighting", default="uniform", type=str, help="Max gradient norm.")
    parser.add_argument("--legacy", action=argparse.BooleanOptionalAction, default=False)

    parser.add_argument("--use-encoder", action="store_true", 
                       help="use encoder for projection loss")
    parser.add_argument("--encoder-type", type=str, default="dinov2",
                       choices=["dinov2", "dino", "clip"])

    parser.add_argument("--num-sampling-steps", type=int, default=100,
                       help="number of steps for ODE integration")

    parser.add_argument("--prototype_path", type=str, default="./prototypes/dinov2_cifar10_prototypes.pt",)


    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()
        
    return args

if __name__ == "__main__":
    args = parse_args()
    
    main(args)
