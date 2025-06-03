from typing import Optional

import torch
from torch import nn
from dataclasses import dataclass

from .transformer import QuickGELU
from .model import CLIPTextCfg, CLIPVisionCfg, _build_vision_tower
from typing import Callable, Optional
import numpy as np
import torch.nn.functional as F



@dataclass
class LIFTHeadCfg(CLIPTextCfg):
    mlp_ratio: int = 4
    layers: int = 1
    text_embed_dim: int = None



class LIFTHead(nn.Module):
    def __init__(
            self,
            width: int,
            layers: int,
            act_layer: Callable = nn.GELU,
            output_dim: int = 512,
    ):
        super().__init__()
        modules = [nn.Linear(width, output_dim)]
        for _ in range(1, layers):
            modules.append(act_layer())
            modules.append(nn.Linear(output_dim, output_dim))
        self.projector = nn.Sequential(*modules)

    def forward(self, x):
        return self.projector(x)



class LIFT(nn.Module):
    def __init__(
            self,
            embed_dim,
            head_cfg: None,
            text_cfg: CLIPTextCfg,
            vision_cfg: CLIPVisionCfg,
            quick_gelu: bool = False,
            cast_dtype: Optional[torch.dtype] = None,
            simplistic_cos: bool = False,
            init_logit_scale: float = np.log(1 / 0.07),
            init_logit_bias: Optional[float] = None,
    ):
        super().__init__()
        self.head_cfg = LIFTHeadCfg(**head_cfg) if isinstance(head_cfg, dict) else head_cfg
        self.vision_cfg = CLIPVisionCfg(**vision_cfg) if isinstance(vision_cfg, dict) else vision_cfg
        self.simplistic_cos = simplistic_cos

        self.visual = _build_vision_tower(
            embed_dim=embed_dim,
            vision_cfg=self.vision_cfg,
            quick_gelu=quick_gelu,
            cast_dtype=cast_dtype,
        )

        self.mlp_projector = LIFTHead(
            width=embed_dim if embed_dim else self.vision_cfg.width,
            layers=self.head_cfg.layers,
            act_layer=QuickGELU if quick_gelu else nn.GELU,
            output_dim=self.head_cfg.text_embed_dim
        )

        if not self.simplistic_cos:
            self.logit_scale = nn.Parameter(torch.ones([]) * init_logit_scale) # the trainable temperature
            if init_logit_bias is not None:
                self.logit_bias = nn.Parameter(torch.ones([]) * init_logit_bias)
            else:
                self.logit_bias = None


    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.visual.set_grad_checkpointing(enable)


    def forward(self, image, text):
        image_embs = self.visual(image)
                
        image_embs = F.normalize(self.mlp_projector(image_embs), dim=-1)
        text_embs = F.normalize(text.clone(), dim=-1)

        if self.simplistic_cos:
            return {"image_features": image_embs, "text_features": text_embs}
        else:
            return {"image_features": image_embs, "text_features": text_embs, "logit_scale": self.logit_scale.exp()}

    
    def encode_image(self, image, normalize=True):
        image_embs = self.visual(image)
        image_embs = self.mlp_projector(image_embs)
        if normalize:
            image_embs = F.normalize(image_embs, dim=-1)
        return image_embs