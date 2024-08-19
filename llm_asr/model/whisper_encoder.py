from typing import Optional

import torch
import torch.nn as nn

from transformers import (
    CLIPVisionModel, CLIPImageProcessor, CLIPVisionConfig,\
    SiglipVisionModel, SiglipImageProcessor, SiglipVisionConfig
)
from llm_asr.s2wrapper import forward as multiscale_forward


class CLIPVisionTower(nn.Module):
    def __init__(
        self, 
        audio_tower_name: str="openai/clip-vit-large-patch14-336", 
        mm_audio_select_layer: int=-2, # v1.5 is -2
        mm_audio_select_feature: str="patch",
        delay_load: bool=False,
        requires_grad: bool=False,
        scales: Optional[float] = None
    ):
        super().__init__()

        self.is_loaded = False
        self.requires_grad = requires_grad
        self.scales = scales
        self.head_size_scales = 1

        self.audio_tower_name = audio_tower_name
        self.select_layer = mm_audio_select_layer
        self.select_feature = mm_audio_select_feature

        self.image_processor = None
        self.audio_tower = None
        self.is_siglip = False

        if scales is not None:
            self.head_size_scales = len(scales)
        
        if not delay_load:
            self.load_model()
        else:
            if "clip" in self.audio_tower_name:
                self.cfg_only = CLIPVisionConfig.from_pretrained(self.audio_tower_name)
            elif "siglip" in self.audio_tower_name:
                self.cfg_only = SiglipVisionConfig.from_pretrained(self.audio_tower_name)
                self.is_siglip = True
            else:
                raise ValueError(f'Unsupported audio_tower_name: {self.audio_tower_name}')

    def load_model(self):
        if "clip" in self.audio_tower_name:
            self.image_processor = CLIPImageProcessor.from_pretrained(self.audio_tower_name)
            self.audio_tower = CLIPVisionModel.from_pretrained(self.audio_tower_name)
        elif "siglip" in self.audio_tower_name:
            self.image_processor = SiglipImageProcessor.from_pretrained(self.audio_tower_name)
            self.audio_tower = SiglipVisionModel.from_pretrained(self.audio_tower_name)
            self.is_siglip = True
        else:
            raise ValueError(f'Unsupported audio_tower_name: {self.audio_tower_name}')
        self.audio_tower.requires_grad_(self.requires_grad)

        self.is_loaded = True

    def feature_select(self, image_forward_outs):
        audio_features = image_forward_outs.hidden_states[self.select_layer]
        if self.select_feature == 'patch':
            audio_features = audio_features[:, 1:]
        elif self.select_feature == 'cls_patch':
            audio_features = audio_features
        else:
            raise ValueError(f'Unexpected select feature: {self.select_feature}')
        return audio_features

    @torch.no_grad()
    def forward(self, images, is_forward_outs=False):
        if type(images) is list:
            audio_features = []
            for image in images:
                if self.scales is None:
                    image_feature = self._forward_feature(images.unsqueeze(0), is_forward_outs)
                else:
                    image_feature = multiscale_forward(
                        self._forward_feature, 
                        images.unsqueeze(0), 
                        scales=self.scales, 
                        num_prefix_token=0, 
                        max_split_size=self.image_processor.size["height"]
                    )
                audio_features.append(image_feature)
        else:
            if self.scales is None:
                audio_features = self._forward_feature(images, is_forward_outs)
            else:
                audio_features = multiscale_forward(
                    self._forward_feature, 
                    images, 
                    scales=self.scales, 
                    num_prefix_token=0, 
                    max_split_size=self.image_processor.size["height"]
                )

        return audio_features
    
    def _forward_feature(self, inputs, is_forward_outs=False):
        image_forward_outs = self.audio_tower(inputs.to(device=self.device, dtype=self.dtype), output_hidden_states=True)
        audio_features = self.feature_select(image_forward_outs)
        if is_forward_outs:
            # For Dense Connector
            return audio_features, image_forward_outs

        return audio_features

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return self.audio_tower.dtype

    @property
    def device(self):
        return self.audio_tower.device

    @property
    def config(self):
        if self.is_loaded:
            return self.audio_tower.config
        else:
            return self.cfg_only

    @property
    def hidden_size(self):
        #if self.scales is None:
        #    return self.config.hidden_size
        
        return self.config.hidden_size*self.head_size_scales
    
    @property
    def num_patches(self):
        return (self.config.image_size // self.config.patch_size) ** 2
