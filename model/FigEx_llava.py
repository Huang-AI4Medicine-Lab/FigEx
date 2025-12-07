import torch
import torch.nn as nn

from .llava.model.language_model.llava_llama import (LlavaLlamaForCausalLM, LlavaLlamaModel)

from mmdet.models.data_preprocessors.data_preprocessor import DetDataPreprocessor
from .grounding_dino.build_gdino_new import build_gdino

from torch.nn.utils.rnn import pad_sequence

from transformers import AutoTokenizer

from utils.create_test_annfile_mmdet import load_as_mmdet

from utils.utils import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN,
                         DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX)

import copy



# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained('weights/Lenna-7B', use_fast=False)
tokenizer.pad_token = tokenizer.unk_token


class FigExMetaModel:
    def __init__(
        self,
        config,
        **kwargs,
    ):
        super(FigExMetaModel, self).__init__(config)

        self.config = config
        self.vision_pretrained = kwargs.get("vision_pretrained", None)
        self.initialize_figex_modules(self.config)

    def initialize_figex_modules(self, config):
        self.visual_model = build_gdino()
        for param in self.visual_model.parameters():
            param.requires_grad = False
        in_dim = config.hidden_size
        out_dim = config.out_dim
        text_fc = [
            nn.Linear(in_dim, in_dim),
            nn.ReLU(inplace=True),
            nn.Linear(in_dim, out_dim),
            nn.Dropout(0.0),
        ]
        self.text_hidden_fcs = nn.ModuleList([nn.Sequential(*text_fc)])
        self.text_hidden_fcs.train()
        for param in self.text_hidden_fcs.parameters():
            param.requires_grad = True

class FigExModel(FigExMetaModel, LlavaLlamaModel):
    def __init__(
        self,
        config,
        **kwargs,
    ):
        super(FigExModel, self).__init__(config, **kwargs)

        self.config.use_cache = False
        self.config.vision_tower = self.config.mm_vision_tower       
        self.config.mm_vision_select_feature = "patch"
        self.config.image_aspect_ratio = "square"
        self.config.image_grid_pinpoints = None
        self.config.tune_mm_mlp_adapter = False
        self.config.freeze_mm_mlp_adapter = True
        self.config.pretrain_mm_mlp_adapter = None
        self.config.mm_use_im_patch_token = False


class FigExForCausalLM(LlavaLlamaForCausalLM):

    
    def __init__(self, config, **kwargs, ):
        super().__init__(config)
        self.det_token_idx = kwargs.pop("det_token_idx")
        self.model = FigExModel(config, **kwargs)
        
        '''
        config.mm_vision_tower = kwargs.get(
            "vision_tower", "openai/clip-vit-large-patch14"
        )
        '''
        
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()
        self.data_preprocessor = DetDataPreprocessor(
            mean=[123.675, 116.28, 103.53],
            std=[58.395, 57.12, 57.375],
            bgr_to_rgb=True,
            pad_mask=False,
        )
    
       
    def evaluate(self, images_clip, image_path, input_ids, resize_list, original_size_list,
                 max_new_tokens=32, tokenizer=None, caption=None):
        with torch.no_grad():
            
            '''
            print('images_clip', images_clip)
     
            print(input_ids)
            print(tokenizer.decode(input_ids[input_ids != IMAGE_TOKEN_INDEX].tolist()))
            '''
            
            outputs = self.generate(
                images=images_clip,
                input_ids=input_ids,
                max_new_tokens=max_new_tokens,
                num_beams=1,
                output_hidden_states=True,
                return_dict_in_generate=True,
            )
        
            output_hidden_states = outputs.hidden_states[-1]
            output_ids = outputs.sequences

        return output_ids

    def forward(self, **kwargs):
        if "past_key_values" in kwargs:
            return super().forward(**kwargs)
        return self.model_forward(**kwargs)

    def model_forward(
        self,
        images_clip: torch.FloatTensor,
        input_ids: torch.FloatTensor,
        attention_masks: torch.FloatTensor,
        labels: torch.FloatTensor,
        image_path,
        inference: bool = False,
        **kwargs,
    ):

        output = super().forward(
            images=images_clip,
            attention_mask=attention_masks,
            input_ids=input_ids,
            labels=labels,
            output_hidden_states=True,
        )
        
        return output.loss, output.logits
