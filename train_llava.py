import argparse
import os
import sys
import json
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
from transformers import AutoTokenizer, CLIPImageProcessor
from tqdm import tqdm
import torchvision.transforms as transforms
import torch.nn as nn

from model.FigEx_llava import FigExForCausalLM    
from peft import LoraConfig, get_peft_model
from utils.create_test_annfile_mmdet import load_as_mmdet 
from model.llava.mm_utils import tokenizer_image_token
from utils.utils import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN,
                         DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX)

# Disable SSL verification for CURL if needed
os.environ['CURL_CA_BUNDLE'] = ''

#########################################################################
# YOLOObjectDetectionDataset
#########################################################################
class YOLOObjectDetectionDataset(Dataset):
    def __init__(self, dataset_path, tokenizer, image_processor, max_boxes=1): 
        self.dataset_path    = dataset_path
        self.image_dir       = os.path.join(dataset_path, "images")
        self.annotation_dir  = os.path.join(dataset_path, "labels")
        self.caption_dir     = os.path.join(dataset_path, "captions")
        self.subcap_dir      = os.path.join(dataset_path, "subcaptions")
        self.tokenizer       = tokenizer
        self.image_processor = image_processor
        self.max_boxes       = max_boxes

        # Get sorted image filenames ending with .jpg or .png
        self.image_filenames = sorted(
            [f for f in os.listdir(self.image_dir) if f.endswith((".jpg", ".png"))]
        )

    def __len__(self):
        return len(self.image_filenames)

    def num_to_label(self, num):
        return chr(ord('A') + num)

    def __getitem__(self, idx):
        image_name = self.image_filenames[idx]
        image_path = os.path.join(self.image_dir, image_name)
        annotation_path = os.path.join(
            self.annotation_dir,
            image_name.replace(".jpg", ".txt").replace(".png", ".txt")
        )
        caption_path = os.path.join(
            self.caption_dir,
            image_name.replace(".jpg", ".txt").replace(".png", ".txt")
        )
        subcap_path = os.path.join(
            self.subcap_dir,
            image_name.replace(".jpg", ".txt").replace(".png", ".txt")
        )

        if not (
            os.path.exists(image_path)
            and os.path.exists(annotation_path)
            and os.path.exists(caption_path)
            and os.path.exists(subcap_path)
        ):
            return None

        image = cv2.imread(image_path)
        if image is None:
            return None
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        orig_h, orig_w = image.shape[:2]

        bboxes, _ = self._load_yolo_annotation(annotation_path, orig_w, orig_h)
        if len(bboxes) == 0:
            return None

        image_clip = self.image_processor.preprocess(
            image, return_tensors="pt"
        )["pixel_values"][0]

        with open(caption_path, "r", encoding="utf-8") as f:
            caption_text = f.read().strip()
        if not caption_text:
            return None

        image_marker = f"{DEFAULT_IM_START_TOKEN} {DEFAULT_IMAGE_TOKEN} {DEFAULT_IM_END_TOKEN}"
        input_prompt = (
            image_marker
            + f"Input Caption: {caption_text}\nOutput:"
        )

        try:
            with open(subcap_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
        except Exception:
            return None

        # build subcaption text without [DET]
        subcaption_items = []
        for line in lines:
            line = line.strip()
            if not line or " " not in line:
                continue
            class_id_str, subcap = line.split(" ", 1)
            try:
                class_id = int(class_id_str)
                label_char = chr(ord('A') + class_id)
                subcaption_items.append(f"{label_char}: {subcap}")
            except Exception:
                continue

        if not subcaption_items:
            return None

        subcaption_text = " ".join(subcaption_items)

        with open(annotation_path, "r", encoding="utf-8") as f:
            raw_label_lines = [line.strip() for line in f if line.strip()]

        # place EOS at the very end
        answer_text = (
            self.tokenizer.bos_token
            + subcaption_text
            + "\nBounding Boxes:\n"
            + "\n".join(raw_label_lines)
            + self.tokenizer.eos_token
        )

        full_text = input_prompt + " " + answer_text


        full_tokenized = self.tokenizer(
            full_text,
            return_tensors="pt",
            max_length=1024,
            truncation=True,
        )
        full_input_ids      = full_tokenized.input_ids
        full_attention_mask = full_tokenized.attention_mask

        prompt_tokenized = self.tokenizer(
            input_prompt,
            return_tensors="pt",
            max_length=1024,
            truncation=True,
        )
        prompt_length = prompt_tokenized.input_ids.shape[1]

        labels_ids = full_input_ids.clone()
        labels_ids[0, :prompt_length] = -100

        target_boxes = torch.tensor(bboxes, dtype=torch.float16)
        num_boxes    = target_boxes.shape[0]
        padded_boxes = torch.zeros((self.max_boxes, 4), dtype=torch.float16)
        padded_boxes[:num_boxes] = target_boxes[: self.max_boxes]

        return (
            image_clip,
            full_input_ids,
            full_attention_mask,
            labels_ids,
            padded_boxes,
            num_boxes,
            (orig_h, orig_w),
            image_path,
        )
    

    def _load_yolo_annotation(self, annotation_path, img_width, img_height):
        bboxes = []
        labels = []
        if os.path.exists(annotation_path):
            with open(annotation_path, "r") as f:
                for line in f.readlines():
                    parts = line.strip().split()
                    if len(parts) == 5 and parts[0].isdigit():
                        class_id = int(parts[0])
                        if 0 <= class_id < 26:
                            x_center, y_center, width, height = map(float, parts[1:])
                            x_min = (x_center - width / 2) * img_width
                            y_min = (y_center - height / 2) * img_height
                            x_max = (x_center + width / 2) * img_width
                            y_max = (y_center + height / 2) * img_height
                            bboxes.append([x_min, y_min, x_max, y_max])
                            labels.append(class_id)
        return bboxes, labels

#########################################################################
# Collate function
#########################################################################
def collate_fn(batch):
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    try:
        images, input_ids, attention_masks, labels_tokenized, bboxes, num_boxes, original_sizes, image_path = zip(*batch)
    except Exception as e:
        print(f"[collate_fn] Error unpacking batch: {e}")
        return None
    return (
        torch.stack(images, dim=0),
        torch.stack(input_ids, dim=0).squeeze(1),
        torch.stack(attention_masks, dim=0).squeeze(1),
        torch.stack(labels_tokenized, dim=0).squeeze(1),
        torch.stack(bboxes, dim=0),
        torch.tensor(num_boxes, dtype=torch.float16),
        list(original_sizes),
        image_path
    )

#########################################################################
# Main training/inference functions and utilities
#########################################################################
def parse_args(args):
    parser = argparse.ArgumentParser(description="FigEx Object Detection Training with LoRA")
    parser.add_argument("--ckpt-path", default="ckpt/Lenna-7B", type=str)
    parser.add_argument("--dataset-path", default="dummy_dataset", type=str)
    parser.add_argument("--val-dataset-path", default="/ix/yufeihuang/Jifeng/MM/dataset_generation/datasets/medicat/val", type=str)
    parser.add_argument("--batch-size", default=4, type=int)
    parser.add_argument("--epochs", default=10, type=int)
    parser.add_argument("--lr", default=0.0003, type=float)
    parser.add_argument("--weight-decay", default=0.01, type=float)
    parser.add_argument("--num-workers", default=4, type=int)
    parser.add_argument("--lora-r", default=8, type=int)
    parser.add_argument("--lora-alpha", default=16, type=int)
    parser.add_argument("--lora-dropout", default=0.05, type=float)
    parser.add_argument("--lora-target-modules", default="q_proj,v_proj", type=str)
    return parser.parse_args(args)

def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Parameters: {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")
    print(f"Trainable Ratio: {trainable_params / total_params:.4f}")

def reinitialize_missing_parameters(model, scaling_factor=10.0, bias_offset=0.05):
    for name, param in model.named_parameters():
        if param is not None and torch.is_floating_point(param):
            if torch.isnan(param).any():
                print(f"[WARNING] Parameter '{name}' contains NaNs. Reinitializing.")
                if param.dim() < 2:
                    nn.init.constant_(param, bias_offset)
                else:
                    nn.init.kaiming_normal_(param, mode='fan_in', nonlinearity='linear')
                    param.data.mul_(scaling_factor)
                    param.data.add_(bias_offset)
    return model

def main(args):
    args = parse_args(args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    tokenizer = AutoTokenizer.from_pretrained(args.ckpt_path, use_fast=False)
    tokenizer.pad_token = tokenizer.unk_token
    det_token_idx = tokenizer("[DET]", add_special_tokens=False).input_ids[0]

    model = FigExForCausalLM.from_pretrained(
        args.ckpt_path,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        det_token_idx=det_token_idx,
    )
    model.config.pad_token_id = tokenizer.pad_token_id

    # Load pretrained weights
    trained_weight = torch.load(os.path.join(args.ckpt_path, 'attn_weight.pt'))
    for name, param in model.named_parameters():
        if 'gamma_' in name or 'text_choose_attn' in name:
            save_name = name.replace('model.visual_model.', 'base_model.model.model.visual_model.')
            param.data = trained_weight[save_name]

    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id

    model.get_model().initialize_vision_modules(model.get_model().config)
    vision_tower = model.get_model().get_vision_tower().to(dtype=torch.float16)
    vision_tower.to(device)
    clip_image_processor = CLIPImageProcessor.from_pretrained(model.config.vision_tower)

    model = reinitialize_missing_parameters(model)

    model.base_model.visual_model.neg_weight_positive = nn.Parameter(torch.tensor([1.2], dtype=torch.float16))
    model.base_model.visual_model.neg_weight_negative = nn.Parameter(torch.tensor([-0.2], dtype=torch.float16))

    def find_linear_layers(model, lora_target_modules):
        cls = nn.Linear
        exclude_list = ["visual_model", "vision_tower", "mm_projector", "text_hidden_fcs", "lm_head"]
        return sorted([
            name for name, module in model.named_modules()
            if isinstance(module, cls)
            and all(x not in name for x in exclude_list)
            and any(x for x in lora_target_modules)
        ])
    
    lora_target_modules = find_linear_layers(model, args.lora_target_modules.split(","))
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=lora_target_modules,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.to(device)
    count_parameters(model)
    
    train_dataset = YOLOObjectDetectionDataset(args.dataset_path, tokenizer, clip_image_processor)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=collate_fn)

    val_dataset = YOLOObjectDetectionDataset(args.val_dataset_path, tokenizer, clip_image_processor)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_fn)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Initialize loss logs
    train_losses = []
    val_losses = []

    # Training loop with epoch-wise saving and logging
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}", leave=True)
        
        for batch in progress_bar:
            if batch is None:
                continue

            try:
                optimizer.zero_grad()

                (images_clip,
                 input_ids,
                 attention_masks,
                 labels,
                 padded_boxes,
                 num_boxes,
                 original_size,
                 image_path) = batch

                attention_masks = (input_ids != tokenizer.pad_token_id).float().to(device)
                images_clip     = images_clip.to(device)
                input_ids       = input_ids.to(device)
                labels          = labels.to(device)

                loss, logits = model(
                    images_clip=images_clip,
                    input_ids=input_ids,
                    attention_masks=attention_masks,
                    labels=labels,
                    image_path=image_path[0]
                )
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                progress_bar.set_postfix(loss=loss.item())

            except RuntimeError as e:
                if 'out of memory' in str(e):
                    print(f"[WARNING] GPU OOM on epoch {epoch}, skipping batch. {e}")
                    torch.cuda.empty_cache()
                    continue
                else:
                    raise

        avg_train_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch} Training Loss: {avg_train_loss:.6f}")

        # Validation
        model.eval()
        val_loss_total = 0.0
        val_batches = 0
        
        
        with torch.no_grad():
            for batch in progress_bar:
                if batch is None:
                    continue

                try:
                    # 解包 batch
                    (images_clip,
                     input_ids,
                     attention_masks,
                     labels,
                     padded_boxes,
                     num_boxes,
                     original_size,
                     image_path) = batch

                    attention_masks = (input_ids != tokenizer.pad_token_id).float().to(device)
                    images_clip     = images_clip.to(device)
                    input_ids       = input_ids.to(device)
                    labels          = labels.to(device)

                    loss, _ = model(
                        images_clip=images_clip,
                        input_ids=input_ids,
                        attention_masks=attention_masks,
                        labels=labels,
                        image_path=image_path[0]
                    )

                    val_loss_total += loss.item()
                    val_batches += 1
                    progress_bar.set_postfix(val_loss=loss.item())

                except RuntimeError as e:
                    if 'out of memory' in str(e):
                        print(f"[WARNING] GPU OOM on validation epoch {epoch+1}, skipping batch. {e}")
                        torch.cuda.empty_cache()
                        continue
                    else:
                        raise

        avg_val_loss = val_loss_total / val_batches if val_batches > 0 else float('inf')
        print(f"Epoch {epoch+1} Validation Loss: {avg_val_loss:.6f}")
  
        # Save weights and losses each epoch
        save_epoch_dir = f'./fine-tuned-model/llava_medicat/epoch_{epoch+1}'
        os.makedirs(save_epoch_dir, exist_ok=True)
        # Save base model and LoRA adapter
        model.model.save_pretrained(save_epoch_dir)
        model.save_pretrained(os.path.join(save_epoch_dir, "lora_adapter"))
        tokenizer.save_pretrained(save_epoch_dir)

        # Record and save losses
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        losses_log = {"train_loss": train_losses, "val_loss": val_losses}
        with open(os.path.join(save_epoch_dir, "losses.json"), "w") as lf:
            json.dump(losses_log, lf, indent=2)

    print("Training completed.")

if __name__ == "__main__":
    main(sys.argv[1:])
