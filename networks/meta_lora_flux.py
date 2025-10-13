# temporary minimum implementation of LoRA
# FLUX doesn't have Conv2d, so we ignore it
# TODO commonize with the original implementation

print("--- RUNNING MODIFIED META LORA FLUX SCRIPT (v2) ---")

# LoRA network module
# reference:
# https://github.com/microsoft/LoRA/blob/main/loralib/layers.py
# https://github.com/cloneofsimo/lora/blob/master/lora_diffusion/lora.py

import math
import os
from contextlib import contextmanager
from typing import Dict, List, Optional, Tuple, Type, Union
from diffusers import AutoencoderKL
from transformers import CLIPTextModel
import numpy as np
import torch
from torch import Tensor
import re
from library.utils import setup_logging
from library.sdxl_original_unet import SdxlUNet2DConditionModel
from safetensors.torch import load_file
import weakref

setup_logging()
import logging

logger = logging.getLogger(__name__)


NUM_DOUBLE_BLOCKS = 19
NUM_SINGLE_BLOCKS = 38


class MetaLoRAModule(torch.nn.Module):
    """
    replaces forward method of the original Linear, instead of replacing the original Linear module.
    """

    def __init__(
        self,
        lora_name,
        org_module: torch.nn.Module,
        multiplier=1.0,
        lora_dim=4,
        alpha=1,
        dropout=None,
        rank_dropout=None,
        module_dropout=None,
        up_rank=4,
        **kwargs,
    ):
        """
        if alpha == 0 or None, alpha is rank (no scaling).
        """
        super().__init__()
        self.lora_name = lora_name

        if org_module.__class__.__name__ == "Conv2d":
            in_dim = org_module.in_channels
            out_dim = org_module.out_channels
        else:
            in_dim = org_module.in_features
            out_dim = org_module.out_features

        self.lora_dim = lora_dim
        self.up_rank = up_rank

        self.lora_down = torch.nn.Linear(in_dim, self.lora_dim, bias=False)
        self.lora_mid = torch.nn.Linear(self.lora_dim, self.up_rank, bias=False)
        self.lora_up = torch.nn.Linear(self.up_rank, out_dim, bias=False)

        torch.nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
        torch.nn.init.kaiming_uniform_(self.lora_mid.weight, a=math.sqrt(5))
        torch.nn.init.zeros_(self.lora_up.weight)

        if type(alpha) == torch.Tensor:
            alpha = alpha.detach().float().numpy()  # without casting, bf16 causes error
        alpha = self.lora_dim if alpha is None or alpha == 0 else alpha
        self.scale = alpha / self.lora_dim
        self.register_buffer("alpha", torch.tensor(alpha))  # 定数として扱える

        # same as microsoft's
        self.multiplier = multiplier
        self.org_module = org_module  # remove in applying
        self.dropout = dropout
        self.rank_dropout = rank_dropout
        self.module_dropout = module_dropout

    def apply_to(self):
        self.org_forward = self.org_module.forward
        self.org_module.forward = self.forward

        del self.org_module

    def forward(self, x):
        org_forwarded = self.org_forward(x)

        # module dropout
        if self.module_dropout is not None and self.training:
            if torch.rand(1) < self.module_dropout:
                return org_forwarded

        lx = self.lora_down(x)

        # normal dropout
        if self.dropout is not None and self.training:
            lx = torch.nn.functional.dropout(lx, p=self.dropout)

        # rank dropout
        if self.rank_dropout is not None and self.training:
            mask = torch.rand((lx.size(0), self.lora_dim), device=lx.device) > self.rank_dropout
            if len(lx.size()) == 3:
                mask = mask.unsqueeze(1)  # for Text Encoder
            elif len(lx.size()) == 4:
                mask = mask.unsqueeze(-1).unsqueeze(-1)  # for Conv2d
            lx = lx * mask
            scale = self.scale * (1.0 / (1.0 - self.rank_dropout))
        else:
            scale = self.scale

        lx = self.lora_mid(lx)
        lx = self.lora_up(lx)

        return org_forwarded + lx * self.multiplier * scale

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def dtype(self):
        return next(self.parameters()).dtype


def create_network(
    multiplier: float,
    network_dim: Optional[int],
    network_alpha: Optional[float],
    ae: AutoencoderKL,
    text_encoders: List[CLIPTextModel],
    flux,
    neuron_dropout: Optional[float] = None,
    **kwargs,
):
    if network_dim is None:
        network_dim = 4  # default
    if network_alpha is None:
        network_alpha = 1.0

    # extract dim/alpha for conv2d, and block dim
    conv_dim = kwargs.get("conv_dim", None)
    conv_alpha = kwargs.get("conv_alpha", None)
    if conv_dim is not None:
        conv_dim = int(conv_dim)
        if conv_alpha is None:
            conv_alpha = 1.0
        else:
            conv_alpha = float(conv_alpha)

    print("--- DEBUG: kwargs before pop ---")
    print(kwargs)

    up_rank = int(kwargs.pop("up_rank", 4))
    pretrained_meta_lora_path = kwargs.pop("pretrained_meta_lora_path", None)
    # pop dropout related args
    kwargs.pop("dropout", None)
    kwargs.pop("rank_dropout", None)
    kwargs.pop("module_dropout", None)

    train_t5xxl = kwargs.pop("train_t5xxl", False)
    if train_t5xxl is not None:
        train_t5xxl = True if str(train_t5xxl).lower() == "true" else False

    print(f"--- DEBUG: up_rank after pop: {up_rank}")
    print(f"--- DEBUG: pretrained_meta_lora_path after pop: {pretrained_meta_lora_path}")
    print("--- DEBUG: kwargs after pop ---")
    print(kwargs)

    # すごく引数が多いな ( ^ω^)･･･
    network = MetaLoRANetwork(
        text_encoders,
        flux,
        multiplier=multiplier,
        lora_dim=network_dim,
        alpha=network_alpha,
        dropout=neuron_dropout,
        up_rank=up_rank,
        pretrained_meta_lora_path=pretrained_meta_lora_path,
        train_t5xxl=train_t5xxl,
        **kwargs,
    )

    return network


# Create network from weights for inference, weights are not loaded here (because can be merged)
def create_network_from_weights(multiplier, file, ae, text_encoders, flux, weights_sd=None, for_inference=False, **kwargs):
    # if unet is an instance of SdxlUNet2DConditionModel or subclass, set is_sdxl to True
    if weights_sd is None:
        if os.path.splitext(file)[1] == ".safetensors":
            from safetensors.torch import load_file, safe_open

            weights_sd = load_file(file)
        else:
            weights_sd = torch.load(file, map_location="cpu")

    # get dim/alpha mapping, and train t5xxl
    modules_dim = {}
    modules_alpha = {}
    train_t5xxl = None
    for key, value in weights_sd.items():
        if "." not in key:
            continue

        lora_name = key.split(".")[0]
        if "alpha" in key:
            modules_alpha[lora_name] = value
        elif "lora_down" in key:
            dim = value.size()[0]
            modules_dim[lora_name] = dim
            # logger.info(lora_name, value.size(), dim)

        if train_t5xxl is None or train_t5xxl is False:
            train_t5xxl = "lora_te3" in lora_name

    if train_t5xxl is None:
        train_t5xxl = False

    # # split qkv
    # double_qkv_rank = None
    # single_qkv_rank = None
    # rank = None
    # for lora_name, dim in modules_dim.items():
    #     if "double" in lora_name and "qkv" in lora_name:
    #         double_qkv_rank = dim
    #     elif "single" in lora_name and "linear1" in lora_name:
    #         single_qkv_rank = dim
    #     elif rank is None:
    #         rank = dim
    #     if double_qkv_rank is not None and single_qkv_rank is not None and rank is not None:
    #         break
    # split_qkv = (double_qkv_rank is not None and double_qkv_rank != rank) or (
    #     single_qkv_rank is not None and single_qkv_rank != rank
    # )
    split_qkv = False  # split_qkv is not needed to care, because state_dict is qkv combined

    module_class = MetaLoRAModule if for_inference else MetaLoRAModule

    network = MetaLoRANetwork(
        text_encoders,
        flux,
        multiplier=multiplier,
        modules_dim=modules_dim,
        modules_alpha=modules_alpha,
        module_class=module_class,
        split_qkv=split_qkv,
        train_t5xxl=train_t5xxl,
    )
    return network, weights_sd


class MetaLoRANetwork(torch.nn.Module):
    # FLUX_TARGET_REPLACE_MODULE = ["DoubleStreamBlock", "SingleStreamBlock"]
    FLUX_TARGET_REPLACE_MODULE_DOUBLE = ["DoubleStreamBlock"]
    FLUX_TARGET_REPLACE_MODULE_SINGLE = ["SingleStreamBlock"]
    TEXT_ENCODER_TARGET_REPLACE_MODULE = ["CLIPAttention", "CLIPSdpaAttention", "CLIPMLP", "T5Attention", "T5DenseGatedActDense"]
    LORA_PREFIX_FLUX = "lora_unet"
    LORA_PREFIX_TEXT_ENCODER_CLIP = "lora_te1"
    LORA_PREFIX_TEXT_ENCODER_T5 = "lora_te3"

    def __init__(
        self,
        text_encoders: Union[List[CLIPTextModel], CLIPTextModel],
        unet,
        multiplier: float = 1.0,
        lora_dim: int = 4,
        alpha: float = 1,
        dropout: Optional[float] = None,
        rank_dropout: Optional[float] = None,
        module_dropout: Optional[float] = None,
        up_rank: int = 4,
        pretrained_meta_lora_path: str = None,
        train_t5xxl: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()
        self.multiplier = multiplier
        self.lora_dim = lora_dim
        self.alpha = alpha
        self.dropout = dropout
        self.rank_dropout = rank_dropout
        self.module_dropout = module_dropout
        self.up_rank = up_rank
        self.train_t5xxl = train_t5xxl
        
        # Original lora_flux arguments
        self.conv_lora_dim = kwargs.get("conv_dim", None)
        self.conv_alpha = kwargs.get("conv_alpha", None)
        self.train_blocks = kwargs.get("train_blocks", "all")
        self.split_qkv = kwargs.get("split_qkv", False)
        self.type_dims = kwargs.get("type_dims", None)
        self.in_dims = kwargs.get("in_dims", None)
        self.train_double_block_indices = kwargs.get("train_double_block_indices", None)
        self.train_single_block_indices = kwargs.get("train_single_block_indices", None)
        self.verbose = kwargs.get("verbose", False)


        logger.info(f"create MetaLoRA network. base dim (rank): {lora_dim}, up_rank: {up_rank}, alpha: {alpha}")

        # create module instances
        def create_modules(
            is_flux: bool,
            text_encoder_idx: Optional[int],
            root_module: torch.nn.Module,
            target_replace_modules: List[str],
            filter: Optional[str] = None,
            default_dim: Optional[int] = None,
        ) -> List[MetaLoRAModule]:
            prefix = (
                self.LORA_PREFIX_FLUX
                if is_flux
                else (self.LORA_PREFIX_TEXT_ENCODER_CLIP if text_encoder_idx == 0 else self.LORA_PREFIX_TEXT_ENCODER_T5)
            )

            loras = []
            skipped = []
            
            for name, module in root_module.named_modules():
                if target_replace_modules is None or module.__class__.__name__ in target_replace_modules:
                    if target_replace_modules is None:
                        module = root_module

                    for child_name, child_module in module.named_modules():
                        is_linear = child_module.__class__.__name__ == "Linear"
                        
                        if is_linear:
                            lora_name = prefix + "." + (name + "." if name else "") + child_name
                            lora_name = lora_name.replace(".", "_")
                            
                            if filter is not None and not filter in lora_name:
                                continue

                            dim = default_dim if default_dim is not None else self.lora_dim
                            
                            if dim is None or dim == 0:
                                skipped.append(lora_name)
                                continue

                            lora = MetaLoRAModule(
                                lora_name,
                                child_module,
                                self.multiplier,
                                dim,
                                self.alpha,
                                dropout=self.dropout,
                                rank_dropout=self.rank_dropout,
                                module_dropout=self.module_dropout,
                                up_rank=self.up_rank,
                            )
                            loras.append(lora)

                if target_replace_modules is None:
                    break
            return loras, skipped

        # create LoRA for text encoder
        self.text_encoder_loras: List[MetaLoRAModule] = []
        skipped_te = []
        for i, text_encoder in enumerate(text_encoders):
            index = i
            if not self.train_t5xxl and index > 0:
                break

            logger.info(f"create LoRA for Text Encoder {index+1}:")
            text_encoder_loras, skipped = create_modules(False, index, text_encoder, self.TEXT_ENCODER_TARGET_REPLACE_MODULE)
            logger.info(f"create LoRA for Text Encoder {index+1}: {len(text_encoder_loras)} modules.")
            self.text_encoder_loras.extend(text_encoder_loras)
            skipped_te += skipped

        # create LoRA for U-Net
        if self.train_blocks == "all":
            target_replace_modules = self.FLUX_TARGET_REPLACE_MODULE_DOUBLE + self.FLUX_TARGET_REPLACE_MODULE_SINGLE
        elif self.train_blocks == "single":
            target_replace_modules = self.FLUX_TARGET_REPLACE_MODULE_SINGLE
        elif self.train_blocks == "double":
            target_replace_modules = self.FLUX_TARGET_REPLACE_MODULE_DOUBLE
        else:
            target_replace_modules = []

        self.unet_loras: List[MetaLoRAModule]
        self.unet_loras, skipped_un = create_modules(True, None, unet, target_replace_modules)

        if self.in_dims:
            for filter, in_dim in zip(["_img_in", "_time_in", "_vector_in", "_guidance_in", "_txt_in"], self.in_dims):
                loras, _ = create_modules(True, None, unet, None, filter=filter, default_dim=in_dim)
                self.unet_loras.extend(loras)

        logger.info(f"create LoRA for FLUX {self.train_blocks} blocks: {len(self.unet_loras)} modules.")
        
        skipped = skipped_te + skipped_un
        if self.verbose and len(skipped) > 0:
            logger.warning(
                f"because dim (rank) is 0, {len(skipped)} LoRA modules are skipped"
            )
            for name in skipped:
                logger.info(f"\t{name}")

        # assertion
        names = set()
        for lora in self.text_encoder_loras + self.unet_loras:
            assert lora.lora_name not in names, f"duplicated lora name: {lora.lora_name}"
            names.add(lora.lora_name)

        # load pretrained weights
        if pretrained_meta_lora_path:
            self.load_pretrained_weights(pretrained_meta_lora_path)

        # Freeze lora_down
        for lora in self.text_encoder_loras + self.unet_loras:
            if hasattr(lora, 'lora_down'):
                lora.lora_down.requires_grad_(False)
            
    def load_pretrained_weights(self, path):
        if not os.path.exists(path):
            logger.warning(f"pretrained meta lora weights not found at {path}")
            return
        
        logger.info(f"loading pretrained meta lora weights from {path}")
        state_dict = load_file(path)

        for lora in self.text_encoder_loras + self.unet_loras:
            down_key = lora.lora_name + ".lora_down.weight"
            if down_key in state_dict:
                lora.lora_down.weight.data.copy_(state_dict[down_key])
                # print(f"Loaded pretrained weight for {lora.lora_name}")

    def set_multiplier(self, multiplier):
        self.multiplier = multiplier
        for lora in self.text_encoder_loras + self.unet_loras:
            lora.multiplier = self.multiplier

    def set_enabled(self, is_enabled):
        for lora in self.text_encoder_loras + self.unet_loras:
            lora.enabled = is_enabled

    def update_norms(self):
        for lora in self.text_encoder_loras + self.unet_loras:
            lora.update_norms()

    def update_grad_norms(self):
        for lora in self.text_encoder_loras + self.unet_loras:
            lora.update_grad_norms()

    def grad_norms(self) -> Tensor:
        grad_norms = []
        for lora in self.text_encoder_loras + self.unet_loras:
            if hasattr(lora, "grad_norms") and lora.grad_norms is not None:
                grad_norms.append(lora.grad_norms.mean(dim=0))
        return torch.stack(grad_norms) if len(grad_norms) > 0 else torch.tensor([])

    def weight_norms(self) -> Tensor:
        weight_norms = []
        for lora in self.text_encoder_loras + self.unet_loras:
            if hasattr(lora, "weight_norms") and lora.weight_norms is not None:
                weight_norms.append(lora.weight_norms.mean(dim=0))
        return torch.stack(weight_norms) if len(weight_norms) > 0 else torch.tensor([])

    def combined_weight_norms(self) -> Tensor:
        combined_weight_norms = []
        for lora in self.text_encoder_loras + self.unet_loras:
            if hasattr(lora, "combined_weight_norms") and lora.combined_weight_norms is not None:
                combined_weight_norms.append(lora.combined_weight_norms.mean(dim=0))
        return torch.stack(combined_weight_norms) if len(combined_weight_norms) > 0 else torch.tensor([])


    def load_weights(self, file):
        if os.path.splitext(file)[1] == ".safetensors":
            from safetensors.torch import load_file

            weights_sd = load_file(file)
        else:
            weights_sd = torch.load(file, map_location="cpu")

        info = self.load_state_dict(weights_sd, False)
        return info

    def load_state_dict(self, state_dict, strict=True):
        # override to convert original weight to split qkv
        if not self.split_qkv:
            return super().load_state_dict(state_dict, strict)

        # split qkv
        for key in list(state_dict.keys()):
            if "double" in key and "qkv" in key:
                split_dims = [3072] * 3
            elif "single" in key and "linear1" in key:
                split_dims = [3072] * 3 + [12288]
            else:
                continue

            weight = state_dict[key]
            lora_name = key.split(".")[0]
            if "lora_down" in key and "weight" in key:
                # dense weight (rank*3, in_dim)
                split_weight = torch.chunk(weight, len(split_dims), dim=0)
                for i, split_w in enumerate(split_weight):
                    state_dict[f"{lora_name}.lora_down.{i}.weight"] = split_w

                del state_dict[key]
                # print(f"split {key}: {weight.shape} to {[w.shape for w in split_weight]}")
            elif "lora_up" in key and "weight" in key:
                # sparse weight (out_dim=sum(split_dims), rank*3)
                rank = weight.size(1) // len(split_dims)
                i = 0
                for j in range(len(split_dims)):
                    state_dict[f"{lora_name}.lora_up.{j}.weight"] = weight[i : i + split_dims[j], j * rank : (j + 1) * rank]
                    i += split_dims[j]
                del state_dict[key]

                # # check is sparse
                # i = 0
                # is_zero = True
                # for j in range(len(split_dims)):
                #     for k in range(len(split_dims)):
                #         if j == k:
                #             continue
                #         is_zero = is_zero and torch.all(weight[i : i + split_dims[j], k * rank : (k + 1) * rank] == 0)
                #     i += split_dims[j]
                # if not is_zero:
                #     logger.warning(f"weight is not sparse: {key}")
                # else:
                #     logger.info(f"weight is sparse: {key}")

                # print(
                #     f"split {key}: {weight.shape} to {[state_dict[k].shape for k in [f'{lora_name}.lora_up.{j}.weight' for j in range(len(split_dims))]]}"
                # )

            # alpha is unchanged

        return super().load_state_dict(state_dict, strict)

    def state_dict(self, destination=None, prefix="", keep_vars=False):
        if not self.split_qkv:
            return super().state_dict(destination, prefix, keep_vars)

        # merge qkv
        state_dict = super().state_dict(destination, prefix, keep_vars)
        new_state_dict = {}
        for key in list(state_dict.keys()):
            if "double" in key and "qkv" in key:
                split_dims = [3072] * 3
            elif "single" in key and "linear1" in key:
                split_dims = [3072] * 3 + [12288]
            else:
                new_state_dict[key] = state_dict[key]
                continue

            if key not in state_dict:
                continue  # already merged

            lora_name = key.split(".")[0]

            # (rank, in_dim) * 3
            down_weights = [state_dict.pop(f"{lora_name}.lora_down.{i}.weight") for i in range(len(split_dims))]
            # (split dim, rank) * 3
            up_weights = [state_dict.pop(f"{lora_name}.lora_up.{i}.weight") for i in range(len(split_dims))]

            alpha = state_dict.pop(f"{lora_name}.alpha")

            # merge down weight
            down_weight = torch.cat(down_weights, dim=0)  # (rank, split_dim) * 3 -> (rank*3, sum of split_dim)

            # merge up weight (sum of split_dim, rank*3)
            rank = up_weights[0].size(1)
            up_weight = torch.zeros((sum(split_dims), down_weight.size(0)), device=down_weight.device, dtype=down_weight.dtype)
            i = 0
            for j in range(len(split_dims)):
                up_weight[i : i + split_dims[j], j * rank : (j + 1) * rank] = up_weights[j]
                i += split_dims[j]

            new_state_dict[f"{lora_name}.lora_down.weight"] = down_weight
            new_state_dict[f"{lora_name}.lora_up.weight"] = up_weight
            new_state_dict[f"{lora_name}.alpha"] = alpha

            # print(
            #     f"merged {lora_name}: {lora_name}, {[w.shape for w in down_weights]}, {[w.shape for w in up_weights]} to {down_weight.shape}, {up_weight.shape}"
            # )
            print(f"new key: {lora_name}.lora_down.weight, {lora_name}.lora_up.weight, {lora_name}.alpha")

        return new_state_dict

    def apply_to(self, text_encoders, flux, apply_text_encoder=True, apply_unet=True):
        if apply_text_encoder:
            logger.info(f"enable LoRA for text encoder: {len(self.text_encoder_loras)} modules")
        else:
            self.text_encoder_loras = []

        if apply_unet:
            logger.info(f"enable LoRA for U-Net: {len(self.unet_loras)} modules")
        else:
            self.unet_loras = []

        for lora in self.text_encoder_loras + self.unet_loras:
            lora.apply_to()
            self.add_module(lora.lora_name, lora)

    # マージできるかどうかを返す
    def is_mergeable(self):
        return True

    # TODO refactor to common function with apply_to
    def merge_to(self, text_encoders, flux, weights_sd, dtype=None, device=None):
        apply_text_encoder = apply_unet = False
        for key in weights_sd.keys():
            if key.startswith(MetaLoRANetwork.LORA_PREFIX_TEXT_ENCODER_CLIP) or key.startswith(MetaLoRANetwork.LORA_PREFIX_TEXT_ENCODER_T5):
                apply_text_encoder = True
            elif key.startswith(MetaLoRANetwork.LORA_PREFIX_FLUX):
                apply_unet = True

        if apply_text_encoder:
            logger.info("enable LoRA for text encoder")
        else:
            self.text_encoder_loras = []

        if apply_unet:
            logger.info("enable LoRA for U-Net")
        else:
            self.unet_loras = []

        for lora in self.text_encoder_loras + self.unet_loras:
            sd_for_lora = {}
            for key in weights_sd.keys():
                if key.startswith(lora.lora_name):
                    sd_for_lora[key[len(lora.lora_name) + 1 :]] = weights_sd[key]
            lora.merge_to(sd_for_lora, dtype, device)

        logger.info(f"weights are merged")

    def set_loraplus_lr_ratio(self, loraplus_lr_ratio, loraplus_unet_lr_ratio, loraplus_text_encoder_lr_ratio):
        self.loraplus_lr_ratio = loraplus_lr_ratio
        self.loraplus_unet_lr_ratio = loraplus_unet_lr_ratio
        self.loraplus_text_encoder_lr_ratio = loraplus_text_encoder_lr_ratio

        logger.info(f"LoRA+ UNet LR Ratio: {self.loraplus_unet_lr_ratio or self.loraplus_lr_ratio}")
        logger.info(f"LoRA+ Text Encoder LR Ratio: {self.loraplus_text_encoder_lr_ratio or self.loraplus_lr_ratio}")

    def prepare_optimizer_params(self, text_encoder_lr, unet_lr):
        print(f"--- [DEBUG] prepare_optimizer_params: num unet loras = {len(self.unet_loras)}")
        
        def get_params(loras, lr):
            if lr is None:
                return []

            params = []
            for lora in loras:
                if hasattr(lora, 'lora_mid') and hasattr(lora, 'lora_up'):
                    params.extend([
                        {"params": list(lora.lora_mid.parameters()), "lr": lr},
                        {"params": list(lora.lora_up.parameters()), "lr": lr},
                    ])
            return params

        text_encoder_params = get_params(self.text_encoder_loras, text_encoder_lr)
        unet_params = get_params(self.unet_loras, unet_lr)
        
        print(f"--- [DEBUG] prepare_optimizer_params: generated {len(unet_params)} unet param groups")
        return text_encoder_params + unet_params

    def enable_gradient_checkpointing(self):
        # not supported
        pass

    def prepare_grad_etc(self, text_encoder, unet):
        self.requires_grad_(True)

    def on_epoch_start(self, text_encoder, unet):
        self.train()

    def get_trainable_params(self):
        return self.parameters()

    def save_weights(self, file, dtype, metadata):
        if metadata is not None and len(metadata) == 0:
            metadata = None

        state_dict = self.state_dict()

        if dtype is not None:
            for key in list(state_dict.keys()):
                v = state_dict[key]
                v = v.detach().clone().to("cpu").to(dtype)
                state_dict[key] = v

        if os.path.splitext(file)[1] == ".safetensors":
            from safetensors.torch import save_file
            from library import train_util

            # Precalculate model hashes to save time on indexing
            if metadata is None:
                metadata = {}
            model_hash, legacy_hash = train_util.precalculate_safetensors_hashes(state_dict, metadata)
            metadata["sshs_model_hash"] = model_hash
            metadata["sshs_legacy_hash"] = legacy_hash

            save_file(state_dict, file, metadata)
        else:
            torch.save(state_dict, file)

    def backup_weights(self):
        # 重みのバックアップを行う
        loras: List[MetaLoRAModule] = self.text_encoder_loras + self.unet_loras
        for lora in loras:
            org_module = lora.org_module_ref[0]
            if not hasattr(org_module, "_lora_org_weight"):
                sd = org_module.state_dict()
                org_module._lora_org_weight = sd["weight"].detach().clone()
                org_module._lora_restored = True

    def restore_weights(self):
        # 重みのリストアを行う
        loras: List[MetaLoRAModule] = self.text_encoder_loras + self.unet_loras
        for lora in loras:
            org_module = lora.org_module_ref[0]
            if not org_module._lora_restored:
                sd = org_module.state_dict()
                sd["weight"] = org_module._lora_org_weight
                org_module.load_state_dict(sd)
                org_module._lora_restored = True

    def pre_calculation(self):
        # 事前計算を行う
        loras: List[MetaLoRAModule] = self.text_encoder_loras + self.unet_loras
        for lora in loras:
            org_module = lora.org_module_ref[0]
            sd = org_module.state_dict()

            org_weight = sd["weight"]
            lora_weight = lora.get_weight().to(org_weight.device, dtype=org_weight.dtype)
            sd["weight"] = org_weight + lora_weight
            assert sd["weight"].shape == org_weight.shape
            org_module.load_state_dict(sd)

            org_module._lora_restored = False
            lora.enabled = False

    def apply_max_norm_regularization(self, max_norm_value, device):
        downkeys = []
        upkeys = []
        alphakeys = []
        norms = []
        keys_scaled = 0

        state_dict = self.state_dict()
        for key in state_dict.keys():
            if "lora_down" in key and "weight" in key:
                downkeys.append(key)
                upkeys.append(key.replace("lora_down", "lora_up"))
                alphakeys.append(key.replace("lora_down.weight", "alpha"))

        for i in range(len(downkeys)):
            down = state_dict[downkeys[i]].to(device)
            up = state_dict[upkeys[i]].to(device)
            alpha = state_dict[alphakeys[i]].to(device)
            dim = down.shape[0]
            scale = alpha / dim

            if up.shape[2:] == (1, 1) and down.shape[2:] == (1, 1):
                updown = (up.squeeze(2).squeeze(2) @ down.squeeze(2).squeeze(2)).unsqueeze(2).unsqueeze(3)
            elif up.shape[2:] == (3, 3) or down.shape[2:] == (3, 3):
                updown = torch.nn.functional.conv2d(down.permute(1, 0, 2, 3), up).permute(1, 0, 2, 3)
            else:
                updown = up @ down

            updown *= scale

            norm = updown.norm().clamp(min=max_norm_value / 2)
            desired = torch.clamp(norm, max=max_norm_value)
            ratio = desired.cpu() / norm.cpu()
            sqrt_ratio = ratio**0.5
            if ratio != 1:
                keys_scaled += 1
                state_dict[upkeys[i]] *= sqrt_ratio
                state_dict[downkeys[i]] *= sqrt_ratio
            scalednorm = updown.norm() * ratio
            norms.append(scalednorm.item())

        return keys_scaled, sum(norms) / len(norms), max(norms)
