import torch
import numpy as np
import torch.nn.functional as F

def mean_flat(x):
    """
    Take the mean over all non-batch dimensions.
    """
    return torch.mean(x, dim=list(range(1, len(x.size()))))

def sum_flat(x):
    """
    Take the mean over all non-batch dimensions.
    """
    return torch.sum(x, dim=list(range(1, len(x.size()))))

class SILoss:
    def __init__(
            self,
            prediction='v',
            path_type="linear",
            weighting="uniform",
            encoders=[], 
            accelerator=None, 
            latents_scale=None, 
            latents_bias=None,
            ):
        self.prediction = prediction
        self.weighting = weighting
        self.path_type = path_type
        self.encoders = encoders
        self.accelerator = accelerator
        self.latents_scale = latents_scale
        self.latents_bias = latents_bias

    def interpolant(self, t):
        if self.path_type == "linear":
            alpha_t = 1 - t
            sigma_t = t
            d_alpha_t = -1
            d_sigma_t =  1
        elif self.path_type == "cosine":
            alpha_t = torch.cos(t * np.pi / 2)
            sigma_t = torch.sin(t * np.pi / 2)
            d_alpha_t = -np.pi / 2 * torch.sin(t * np.pi / 2)
            d_sigma_t =  np.pi / 2 * torch.cos(t * np.pi / 2)
        else:
            raise NotImplementedError()

        return alpha_t, sigma_t, d_alpha_t, d_sigma_t

    def __call__(self, model, images, model_kwargs=None, zs=None):
        if model_kwargs == None:
            model_kwargs = {}
        # sample timesteps
        if self.weighting == "uniform":
            time_input = torch.rand((images.shape[0], 1, 1, 1))
        elif self.weighting == "lognormal":
            # sample timestep according to log-normal distribution of sigmas following EDM
            rnd_normal = torch.randn((images.shape[0], 1 ,1, 1))
            sigma = rnd_normal.exp()
            if self.path_type == "linear":
                time_input = sigma / (1 + sigma)
            elif self.path_type == "cosine":
                time_input = 2 / np.pi * torch.atan(sigma)
                
        time_input = time_input.to(device=images.device, dtype=images.dtype)
        
        noises = torch.randn_like(images)
        alpha_t, sigma_t, d_alpha_t, d_sigma_t = self.interpolant(time_input)
            
        model_input = alpha_t * images + sigma_t * noises
        if self.prediction == 'v':
            model_target = d_alpha_t * images + d_sigma_t * noises
        else:
            raise NotImplementedError() # TODO: add x or eps prediction
        model_output, zs_tilde  = model(model_input, time_input.flatten(), **model_kwargs)
        denoising_loss = mean_flat((model_output - model_target) ** 2)

        # projection loss
        proj_loss = 0.
        bsz = zs[0].shape[0]
        for i, (z, z_tilde) in enumerate(zip(zs, zs_tilde)):
            for j, (z_j, z_tilde_j) in enumerate(zip(z, z_tilde)):
                z_tilde_j = torch.nn.functional.normalize(z_tilde_j, dim=-1) 
                z_j = torch.nn.functional.normalize(z_j, dim=-1) 
                proj_loss += mean_flat(-(z_j * z_tilde_j).sum(dim=-1))
        proj_loss /= (len(zs) * bsz)

        return denoising_loss, proj_loss

class FlowMatchingWithProjectionLoss:
    def __init__(
        self,
        encoders=[],
        accelerator=None,
    ):
        self.encoders = encoders
        self.accelerator = accelerator
    
    def __call__(self, model, x1, model_kwargs=None, zs=None):
        """
        Flow Matching loss with projection loss
        
        Args:
            model: DiT model
            x1: clean images (B, 3, 32, 32)
            model_kwargs: dict with 'y' (class labels)
            zs: list of encoder features
        """
        if model_kwargs is None:
            model_kwargs = {}
        
        # Sample noise
        x0 = torch.randn_like(x1)
        
        # Sample timestep uniformly from [0, 1]
        #t = torch.rand((x1.shape[0],), device=x1.device, dtype=x1.dtype)  # (B,)
        t = torch.rand(x1.shape[0], device=x1.device, dtype=x1.dtype)  # (B,)

        t_expanded = t.view(-1, 1, 1, 1) # (B, 1, 1, 1) for broadcasting

        # Linear interpolation: xt = t*x1 + (1-t)*x0
        xt = t_expanded * x1 + (1 - t_expanded) * x0
        ut = x1 - x0
                
        # Model prediction
        return_features = (zs is not None and len(zs) > 0)
        
        y = torch.zeros(x1.size(0), dtype=torch.long, device=x1.device)
        
        if return_features:
            model_result = model(xt, t.float(), y=y, return_features=True)
            # Dict로 반환되었는지 확인
            if isinstance(model_result, dict):
                model_output = model_result['output']
                zs_tilde = model_result['features']
            else:
                model_output, zs_tilde = model_result
        else:
            model_output = model(xt, t, y=y, return_features=False)
            zs_tilde = None

        flow_loss = mean_flat((model_output - ut) ** 2)

        # Projection loss (REPA style)
        proj_loss = 0.
        if zs is not None and zs_tilde is not None and len(zs) > 0:
            bsz = x1.shape[0]
            for z, z_tilde in zip(zs, zs_tilde):
                for z_j, z_tilde_j in zip(z, z_tilde):
                    z_tilde_j_norm = F.normalize(z_tilde_j, dim=-1)
                    z_j_norm = F.normalize(z_j, dim=-1)
                    proj_loss += mean_flat(-(z_j_norm * z_tilde_j_norm).sum(dim=-1))

            proj_loss /= (len(zs) * bsz)
            proj_loss = proj_loss.mean()

        return flow_loss.mean(), proj_loss



class IndependentFlowMatchingWithProjectionLoss:
    def __init__(
        self,
        encoders=[],
        accelerator=None,
    ):
        self.encoders = encoders
        self.accelerator = accelerator
    
    def __call__(self, model, x1, y, device, model_kwargs=None, zs=None):
        """
        Flow Matching loss with projection loss
        
        Args:
            model: DiT model
            x1: clean images (B, 3, 32, 32)
            model_kwargs: dict with 'y' (class labels)
            zs: list of encoder features
        """
        if model_kwargs is None:
            model_kwargs = {}
        
        # Sample noise
        x0 = torch.randn_like(x1)
        
        # Sample timestep uniformly from [0, 1]
        t1 = torch.rand((x1.shape[0],), device=x1.device, dtype=x1.dtype)  # (B,)
        t2 = torch.rand((x1.shape[0],), device=x1.device, dtype=x1.dtype) 
        t1_expanded = t1.view(-1, 1, 1, 1) # (B, 1, 1, 1) for broadcasting
        t2_expanded = t2.view(-1, 1, 1, 1)  

        y_embedded = model.module.label_embedder(y)
        noise_y = torch.randn(x1.size(0), 384, device=device)
        y_embedded_t = [t2_expanded * y_embedded + (1 - t2_expanded) * noise_y]

        # Linear interpolation: xt = t*x1 + (1-t)*x0
        xt = t1_expanded * x1 + (1 - t1_expanded) * x0
        ut = x1 - x0
                
        # Model prediction
        return_features = (zs is not None and len(zs) > 0)
        
        y = torch.zeros(x1.size(0), dtype=torch.long, device=x1.device)
        
        if return_features:
            model_output, model_output_y ,zs_tilde = model(xt, t1,t2, y=y, noise_vector=y_embedded_t)
        else:
            raise NotImplementedError("IndependentFlowMatchingWithProjectionLoss requires return_features=True")

        
        
        flow_loss = mean_flat((model_output - ut) ** 2)
        flow_loss = flow_loss.mean()

        # Projection loss (REPA style)
        proj_loss = 0.
        if zs is not None and zs_tilde is not None and len(zs) > 0:
            for z, z_tilde in zip(zs, zs_tilde):
                z_tilde_norm = F.normalize(z_tilde, dim=-1)
                z_norm = F.normalize(z, dim=-1)
                proj_loss += mean_flat(-(z_norm * z_tilde_norm).sum(dim=-1))
            proj_loss = proj_loss.mean() / len(zs) 
        
        loss_y = torch.mean((model_output_y - (y_embedded - noise_y))**2)
        
        return flow_loss, proj_loss, loss_y


class IndependentFlowMatchingWithProjectionLossdino:
    def __init__(
        self,
        encoders=[],
        accelerator=None,
    ):
        self.encoders = encoders
        self.accelerator = accelerator
    
    # loss.py의 IndependentFlowMatchingWithProjectionLoss 수정
    def __call__(self, model, x1, y, target_embeddings, device, model_kwargs=None, zs=None):
        """
        Flow Matching loss with projection loss
        
        Args:
            model: DiT model
            x1: clean images (B, 3, 32, 32)
            y: class labels (B,)
            target_embeddings: 타겟 DINOv2 embeddings (B, 384)
            zs: list of encoder features (실시간 계산용, 선택적)
        """
        if model_kwargs is None:
            model_kwargs = {}
        
        x0 = torch.randn_like(x1)
        
        t1 = torch.rand((x1.shape[0],), device=x1.device, dtype=x1.dtype)
        t2 = torch.rand((x1.shape[0],), device=x1.device, dtype=x1.dtype)
        t1_expanded = t1.view(-1, 1, 1, 1)
        t2_expanded = t2.view(-1, 1)

        # 타겟: 이미지별 DINOv2 embedding
        noise_y = torch.randn_like(target_embeddings)
        y_embedded_t = t2_expanded * target_embeddings + (1 - t2_expanded) * noise_y

        xt = t1_expanded * x1 + (1 - t1_expanded) * x0
        ut = x1 - x0
        
        return_features = (zs is not None and len(zs) > 0)
        
        y_dummy = torch.zeros(x1.size(0), dtype=torch.long, device=x1.device)
        
        model_output, model_output_y, zs_tilde = model(
            xt, t1, t2, y=y_dummy, noise_vector=[y_embedded_t]
        )
        
        flow_loss = mean_flat((model_output - ut) ** 2).mean()

        # Projection loss
        proj_loss = 0.
        if zs is not None and zs_tilde is not None and len(zs) > 0:
            for z, z_tilde in zip(zs, zs_tilde):
                z_tilde_norm = F.normalize(z_tilde, dim=-1)
                z_norm = F.normalize(z, dim=-1)
                proj_loss += mean_flat(-(z_norm * z_tilde_norm).sum(dim=-1))
            proj_loss = proj_loss.mean() / len(zs)
        
        # Embedding prediction loss: 이미지별 DINOv2 embedding 예측
        loss_y = torch.mean((model_output_y.squeeze() - (target_embeddings - noise_y))**2)
        
        return flow_loss, proj_loss, loss_y