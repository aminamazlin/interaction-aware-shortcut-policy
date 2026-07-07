from __future__ import annotations

import copy
from typing import Dict, Tuple, Union

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T

from .crop_randomizer import CropRandomizer
from ..common.module_attr_mixin import ModuleAttrMixin


class SeedMultiImageObsEncoder(ModuleAttrMixin):
    """SeedPolicy's fused multi-image ResNet encoder for its UNet policy."""

    def __init__(
        self,
        shape_meta: dict,
        rgb_model: Union[nn.Module, Dict[str, nn.Module]],
        resize_shape: Union[Tuple[int, int], Dict[str, tuple], None] = None,
        crop_shape: Union[Tuple[int, int], Dict[str, tuple], None] = None,
        random_crop: bool = True,
        use_group_norm: bool = False,
        share_rgb_model: bool = False,
        imagenet_norm: bool = False,
        force_resize: int = 76,
        fusion_output_dim: int = 512,
    ) -> None:
        super().__init__()
        del resize_shape  # SeedPolicy forces a short-edge resize to 76.

        self.force_resize = int(force_resize)
        self.fusion_output_dim = int(fusion_output_dim)
        self.share_rgb_model = bool(share_rgb_model)

        rgb_keys = []
        low_dim_keys = []
        key_model_map = nn.ModuleDict()
        key_transform_map = nn.ModuleDict()
        key_shape_map = {}

        if self.share_rgb_model:
            if not isinstance(rgb_model, nn.Module):
                raise TypeError("Shared rgb_model must be an nn.Module.")
            if use_group_norm:
                self._replace_bn_with_gn(rgb_model)
            key_model_map["rgb"] = rgb_model

        obs_shape_meta = shape_meta["obs"]
        for key, attr in obs_shape_meta.items():
            shape = tuple(attr["shape"])
            obs_type = attr.get("type", "low_dim")
            key_shape_map[key] = shape

            if obs_type == "rgb":
                rgb_keys.append(key)
                if not self.share_rgb_model:
                    if isinstance(rgb_model, dict):
                        this_model = rgb_model[key]
                    elif isinstance(rgb_model, nn.Module):
                        this_model = copy.deepcopy(rgb_model)
                    else:
                        raise TypeError("rgb_model must be a module or module dict.")
                    if use_group_norm:
                        self._replace_bn_with_gn(this_model)
                    key_model_map[key] = this_model

                input_shape = (shape[0], self.force_resize, self.force_resize)
                randomizer: nn.Module = nn.Identity()
                if crop_shape is not None:
                    h, w = (
                        crop_shape[key]
                        if isinstance(crop_shape, dict)
                        else crop_shape
                    )
                    if random_crop:
                        randomizer = CropRandomizer(
                            input_shape=input_shape,
                            crop_height=h,
                            crop_width=w,
                            num_crops=1,
                            pos_enc=False,
                        )
                    else:
                        randomizer = T.CenterCrop(size=(h, w))

                normalizer: nn.Module = nn.Identity()
                if imagenet_norm:
                    normalizer = T.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225],
                    )
                key_transform_map[key] = nn.Sequential(
                    T.Resize(size=self.force_resize, antialias=True),
                    randomizer,
                    normalizer,
                )
            elif obs_type == "low_dim":
                low_dim_keys.append(key)
            else:
                raise RuntimeError(f"Unsupported obs type: {obs_type}")

        self.shape_meta = shape_meta
        self.key_model_map = key_model_map
        self.key_transform_map = key_transform_map
        self.rgb_keys = sorted(rgb_keys)
        self.low_dim_keys = sorted(low_dim_keys)
        self.key_shape_map = key_shape_map

        with torch.no_grad():
            dummy_obs = {
                key: torch.zeros((1,) + tuple(attr["shape"]))
                for key, attr in obs_shape_meta.items()
            }
            fusion_input_dim = self._raw_forward(dummy_obs).shape[-1]

        self.fusion_mlp = nn.Sequential(
            nn.Linear(fusion_input_dim, 512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, self.fusion_output_dim),
            nn.LayerNorm(self.fusion_output_dim),
        )

    @staticmethod
    def _replace_bn_with_gn(module: nn.Module) -> None:
        for name, child in module.named_children():
            if isinstance(child, nn.BatchNorm2d):
                num_groups = 32
                while child.num_features % num_groups != 0 and num_groups > 1:
                    num_groups //= 2
                setattr(
                    module,
                    name,
                    nn.GroupNorm(num_groups, child.num_features),
                )
            else:
                SeedMultiImageObsEncoder._replace_bn_with_gn(child)

    def _raw_forward(self, obs_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        batch_size = None
        features = []

        if self.share_rgb_model:
            images = []
            for key in self.rgb_keys:
                image = obs_dict[key]
                batch_size = image.shape[0] if batch_size is None else batch_size
                self._validate_obs(key, image, batch_size)
                images.append(self.key_transform_map[key](image))
            images = torch.cat(images, dim=0)
            feature = self.key_model_map["rgb"](images)
            feature = feature.reshape(-1, batch_size, *feature.shape[1:])
            features.append(torch.moveaxis(feature, 0, 1).reshape(batch_size, -1))
        else:
            for key in self.rgb_keys:
                image = obs_dict[key]
                batch_size = image.shape[0] if batch_size is None else batch_size
                self._validate_obs(key, image, batch_size)
                features.append(
                    self.key_model_map[key](self.key_transform_map[key](image))
                )

        for key in self.low_dim_keys:
            value = obs_dict[key]
            batch_size = value.shape[0] if batch_size is None else batch_size
            self._validate_obs(key, value, batch_size)
            features.append(value)

        return torch.cat(features, dim=-1)

    def _validate_obs(
        self,
        key: str,
        value: torch.Tensor,
        batch_size: int,
    ) -> None:
        if value.shape[0] != batch_size:
            raise ValueError(f"Observation {key!r} has a mismatched batch size.")
        if tuple(value.shape[1:]) != self.key_shape_map[key]:
            raise ValueError(
                f"Observation {key!r} has shape {tuple(value.shape[1:])}, "
                f"expected {self.key_shape_map[key]}."
            )

    def forward(self, obs_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        return self.fusion_mlp(self._raw_forward(obs_dict))

    @torch.no_grad()
    def output_shape(self) -> tuple[int]:
        return (self.fusion_output_dim,)


__all__ = ["SeedMultiImageObsEncoder"]
