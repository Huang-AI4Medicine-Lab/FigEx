import torch
import torch.nn as nn

from .llava.model.language_model.llava_llama import (LlavaLlamaForCausalLM, LlavaLlamaModel)

from mmdet.models.data_preprocessors.data_preprocessor import DetDataPreprocessor
#from .grounding_dino.build_gdino_new import build_gdino
from model.yolos.modeling_yolos_mix import YolosForObjectDetection
from transformers import YolosImageProcessor

from torch.nn.utils.rnn import pad_sequence

from transformers import AutoTokenizer

from utils.create_test_annfile_mmdet import load_as_mmdet

from utils.utils import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN,
                         DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX)

import re

from typing import Dict, List, Tuple

from mmdet.structures import DetDataSample

from PIL import Image

import string

import copy

import cv2

import os



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
        self.visual_model = YolosForObjectDetection.from_pretrained(kwargs.pop("yolos_path"), ignore_mismatched_sizes=True)                                    ################## new models

    def initialize_figex_modules(self, config):
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
        #for param in self.text_hidden_fcs.parameters():
            #param.requires_grad = True

class FigExModel(FigExMetaModel, LlavaLlamaModel):
    def __init__(
        self,
        config,
        **kwargs,
    ):
        super(FigExModel, self).__init__(config, **kwargs)

        self.config.use_cache = False
        #self.config.vision_tower = self.config.mm_vision_tower       
        #self.config.mm_vision_select_feature = "patch"
        #self.config.image_aspect_ratio = "square"
        #self.config.image_grid_pinpoints = None
        #self.config.tune_mm_mlp_adapter = False
        #self.config.freeze_mm_mlp_adapter = True
        #self.config.pretrain_mm_mlp_adapter = None
        #self.config.mm_use_im_patch_token = False


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
        '''
        self.data_preprocessor = DetDataPreprocessor(
            mean=[123.675, 116.28, 103.53],
            std=[58.395, 57.12, 57.375],
            bgr_to_rgb=True,
            pad_mask=False,
        )
        '''

    def language_evaluate(
        self,
        images_clip: torch.FloatTensor,       # (B, C, H, W)
        images_yolo: torch.FloatTensor,       # (B, C, H, W)
        input_ids: torch.LongTensor,          # (B, seq_len)
        attention_mask: torch.LongTensor,     # (B, seq_len)
        max_new_tokens: int,
        threshold: float,
        target_sizes: List[Tuple[int, int]],  # [(H, W), ...]
        tok,
    ) -> Tuple[List[Dict[str, torch.Tensor]], str]:
        """
        Generate text, decode only the newly-generated part, then run YOLOS.
        Prints out debug info: sequence shapes, token IDs, decoded text, and feature sizes.
        """
        # 1) Generate with hidden states
        gen_out = self.generate(
            images=images_clip,
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            pad_token_id=self.config.eos_token_id,
            num_beams=1,
            output_hidden_states=True,
            return_dict_in_generate=True,
        )

        # 2) Full sequence IDs and hidden states
        seq_ids     = gen_out.sequences               # (B, seq_len_total)
        seq_ids = seq_ids[:, 1:]
        last_hidden = gen_out.hidden_states[-1]       # (B, seq_len_total, D)
        if last_hidden.dim() == 2:
            last_hidden = last_hidden.unsqueeze(0)

        # Debug prints for sequence and hidden state sizes
        #print(f"[DEBUG] seq_ids.shape       = {seq_ids.shape}")
        #print(f"[DEBUG] first batch seq_ids = {seq_ids[0].tolist()}")
        #print(f"[DEBUG] last_hidden.shape   = {last_hidden.shape}")

        # 3) Compute prompt vs. generated split
        prompt_len = input_ids.size(1)
        gen_part   = seq_ids[0, prompt_len:]          # (seq_len_gen,)
        #print(f"[DEBUG] prompt_len          = {prompt_len}")
        #print(f"[DEBUG] gen_part.shape      = {gen_part.shape}")
        #print(f"[DEBUG] gen_part token IDs  = {gen_part.tolist()}")

        # 4) Decode only the generated tokens
        decoded_text = tok.decode(gen_part, skip_special_tokens=True).strip()

        return decoded_text
    
    def evaluate(
        self,
        images_clip: torch.FloatTensor,       # (B, C, H, W)
        images_yolo: torch.FloatTensor,       # (B, C, H, W)
        input_ids: torch.LongTensor,          # (B, seq_len)
        attention_mask: torch.LongTensor,     # (B, seq_len)
        max_new_tokens: int,
        threshold: float,
        target_sizes: List[Tuple[int, int]],  # [(H, W), ...]
        tok,
    ) -> Tuple[List[Dict[str, torch.Tensor]], str]:
        """
        Generate text, decode only the newly-generated part, then run YOLOS.
        Prints out debug info: sequence shapes, token IDs, decoded text, and feature sizes.
        """
        # 1) Generate with hidden states
        gen_out = self.generate(
            images=images_clip,
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            pad_token_id=self.config.eos_token_id,
            num_beams=1,
            output_hidden_states=True,
            return_dict_in_generate=True,
        )

        # 2) Full sequence IDs and hidden states
        seq_ids     = gen_out.sequences               # (B, seq_len_total)
        seq_ids = seq_ids[:, 1:]
        last_hidden = gen_out.hidden_states[-1]       # (B, seq_len_total, D)
        if last_hidden.dim() == 2:
            last_hidden = last_hidden.unsqueeze(0)

        # Debug prints for sequence and hidden state sizes
        #print(f"[DEBUG] seq_ids.shape       = {seq_ids.shape}")
        #print(f"[DEBUG] first batch seq_ids = {seq_ids[0].tolist()}")
        #print(f"[DEBUG] last_hidden.shape   = {last_hidden.shape}")

        # 3) Compute prompt vs. generated split
        prompt_len = input_ids.size(1)
        gen_part   = seq_ids[0, prompt_len:]          # (seq_len_gen,)
        #print(f"[DEBUG] prompt_len          = {prompt_len}")
        #print(f"[DEBUG] gen_part.shape      = {gen_part.shape}")
        #print(f"[DEBUG] gen_part token IDs  = {gen_part.tolist()}")

        # 4) Decode only the generated tokens
        decoded_text = tok.decode(gen_part, skip_special_tokens=True).strip()
        #print(f"[DEBUG] decoded_text       = \"{decoded_text}\"")

        # 5) Locate the first [DET] in the generated part
        det_id      = self.det_token_idx
        rel_pos     = (gen_part == det_id).nonzero(as_tuple=True)[0]
        #print(f"[DEBUG] det_id              = {det_id}")
        #print(f"[DEBUG] relative DET pos    = {rel_pos.tolist()}")
        if rel_pos.numel() == 0:
            return [], decoded_text

        # 6) Map to absolute position and extract det_feat
        det_pos  = prompt_len + rel_pos[0].item()
        det_feat = last_hidden[0, det_pos, :]         # (D,)
        #print(f"[DEBUG] absolute DET pos   = {det_pos}")
        #print(f"[DEBUG] det_feat.shape      = {det_feat.shape}")

        # 7) Run YOLOS with the DET feature
        preds = self.model.visual_model(
            pixel_values = images_yolo,                              # (B, C, H, W)
            det_feats    = det_feat.unsqueeze(0).unsqueeze(1),       # (1,1,D)
            text_feats   = last_hidden[0, prompt_len:, :].unsqueeze(0),  # (1, seq_len_gen, D)
        )

        return preds, decoded_text

    def forward(self, **kwargs):
        if "past_key_values" in kwargs:
            return super().forward(**kwargs)
        return self.model_forward(**kwargs)


    def model_forward(
        self,
        images_clip: torch.FloatTensor,
        images_yolo: torch.FloatTensor,
        input_ids: torch.LongTensor,
        attention_masks: torch.FloatTensor,
        labels: torch.LongTensor,
        image_path: str,
        tokenizer,
        gt_bboxes: torch.FloatTensor,
        class_labels: torch.FloatTensor,
        inference: bool = False,
        **kwargs,
    ) -> Tuple[torch.nn.Module, Dict[str, torch.Tensor]]:
        import torch

        # 1) Base forward through the LM, using the CLIP‐processed images
        output = super().forward(
            images=images_clip,
            attention_mask=attention_masks,
            input_ids=input_ids,
            labels=labels,
            output_hidden_states=True,
        )
        
        det_id   = self.det_token_idx    
        last_hidden_states = output.hidden_states[-1]   # (B, seq_len, D)
        logits = output.logits                         # (B, seq_len, V)

        if last_hidden_states.dim() == 2:
            # add batch dim → (1, seq_len, D)
            last_hidden_states = last_hidden_states.unsqueeze(0)
            
        # 1) Locate generation start (first non -100 label)
        mask = (labels[0] != -100).nonzero(as_tuple=True)[0]
        if mask.numel() == 0:
            return output, 0, 0
        gen_start = mask.min().item()

        # 2) Compute predicted token IDs for entire sequence
        pred_ids = logits.argmax(dim=-1)            # (B, seq_len)

        gen_ids  = pred_ids[0, gen_start+1:]          # (seq_len_gen,)

        # 3) Find the first “[DET]” in the generated IDs
        det_id      = tokenizer.convert_tokens_to_ids("[DET]")
        det_rel_pos = (gen_ids == det_id).nonzero(as_tuple=True)[0]
        # if no [DET] token was generated, bail out
        if det_rel_pos.numel() == 0:
            return output, 0, 0
        det_pos = gen_start + det_rel_pos[0].item()

        # 4) Extract the detection token feature and the text features
        det_feature  = last_hidden_states[0, det_pos, :]     # (D,)
        text_feature = last_hidden_states[0, gen_start:, :] # (seq_len_gen, D)
        

        # 2) YOLOS forward: use the separate YOLOS‐processed image batch
        inference = False
        if not inference:
            # Build the labels list expected by YOLOS: one dict per image
            yolos_labels = [{"class_labels": class_labels, "boxes": gt_bboxes}]
            yolos_out = self.model.visual_model(
                pixel_values=images_yolo,
                labels=yolos_labels,
                det_feats = det_feature.unsqueeze(0).unsqueeze(1),
                text_feats = text_feature.unsqueeze(0),
            )
            det_losses_dict = yolos_out.loss_dict
            det_losses = yolos_out.loss
            #print('det_losses_dict', det_losses_dict)
            #print('det_losses', det_losses)
        return output, det_losses, det_losses_dict
            
            
            
            
            
            
            
            

    
