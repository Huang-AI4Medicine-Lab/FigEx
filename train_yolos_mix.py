#!/usr/bin/env python
# coding: utf-8

import argparse
import os
import sys
import json
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import cv2
from transformers import AutoTokenizer, CLIPImageProcessor, YolosImageProcessor
from tqdm import tqdm
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from peft import PeftModel

from torchvision.transforms.functional import to_pil_image

# Performance tuning
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

from model.FigEx_yolos_mix import FigExForCausalLM
from peft import LoraConfig, get_peft_model
from model.llava.mm_utils import tokenizer_image_token
from utils.utils import (
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IMAGE_TOKEN,
    IMAGE_TOKEN_INDEX,
)

import os
from PIL import Image, ImageDraw

import builtins
_old_print = builtins.print
def print(*args, **kwargs):
    if args and isinstance(args[0], str) and "[nltk_data]" in args[0]:
        return
    _old_print(*args, **kwargs)
builtins.print = print

os.environ["CURL_CA_BUNDLE"] = ""  # disable SSL verify if needed


class YOLOObjectDetectionDataset(Dataset):
    """
    Dataset that loads images, captions, sub-captions, and YOLO-format boxes,
    """
    def __init__(
        self,
        dataset_path: str,
        tokenizer,
        clip_processor: CLIPImageProcessor,
        yolo_processor: YolosImageProcessor,
        max_boxes: int = 26,
    ):
        self.image_dir       = os.path.join(dataset_path, "images")
        self.annotation_dir  = os.path.join(dataset_path, "labels")
        self.caption_dir     = os.path.join(dataset_path, "captions")
        self.subcap_dir      = os.path.join(dataset_path, "subcaptions")
        self.tokenizer       = tokenizer
        self.clip_processor  = clip_processor
        self.yolo_processor  = yolo_processor
        self.max_boxes       = max_boxes

        self._save_dir = "processed_with_boxes"
        os.makedirs(self._save_dir, exist_ok=True)
        # ---------------------------------

        self.image_filenames = sorted(
            f for f in os.listdir(self.image_dir)
            if f.lower().endswith((".jpg", ".png"))
        )

    def __len__(self):
        return len(self.image_filenames)

    def num_to_label(self, num: int) -> str:
        return chr(ord("A") + num)

    def __getitem__(self, idx):
        fn       = self.image_filenames[idx]
        img_path = os.path.join(self.image_dir, fn)
        ann_path = os.path.join(self.annotation_dir, fn.rsplit(".",1)[0] + ".txt")
        cap_path = os.path.join(self.caption_dir,   fn.rsplit(".",1)[0] + ".txt")
        sub_path = os.path.join(self.subcap_dir,    fn.rsplit(".",1)[0] + ".txt")
        if not all(os.path.exists(p) for p in (img_path, ann_path, cap_path, sub_path)):
            return None

        pil_img = Image.open(img_path).convert("RGB")
        w, h    = pil_img.size

        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            return None
        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # 3) CLIP preprocessing
        clip_inputs = self.clip_processor(images=img, return_tensors="pt")
        image_clip   = clip_inputs["pixel_values"][0]

        annos = []
        with open(ann_path, "r") as f:
            for line in f:
                ce, xc, yc, bw, bh = map(float, line.split())
                #print(f"1", xc, yc, bw, bh)
                x0 = (xc - bw/2) * w
                y0 = (yc - bh/2) * h
                box_w = bw * w
                box_h = bh * h
                x1 = x0 + box_w
                y1 = y0 + box_h
                #print(f"2", x0, y0, box_w, box_h)
                annos.append({
                    "bbox":        [x0, y0, box_w, box_h],
                    "category_id": int(ce),
                    "area":        bw * w * bh * h,
                    "iscrowd":     0
                })

        yolo_inputs = self.yolo_processor(
            images=[pil_img],
            annotations=[{"image_id": idx, "annotations": annos}],
            return_tensors="pt"
        )
        image_yolo  = yolo_inputs["pixel_values"][0]   # Tensor[C,H,W]
        labels_yolo = yolo_inputs["labels"][0]         # Dict: 'class_labels', 'boxes'
        class_ids   = labels_yolo["class_labels"]
        proc_boxes  = labels_yolo["boxes"]
        nb          = proc_boxes.size(0)

        caption = open(cap_path, "r", encoding="utf-8").read().strip()
        if not caption:
            return None
        marker = f"{DEFAULT_IM_START_TOKEN} {DEFAULT_IMAGE_TOKEN} {DEFAULT_IM_END_TOKEN}"
        prompt = f"{marker} Input Caption: {caption}\nOutput:"
        lines  = [l.strip() for l in open(sub_path, "r", encoding="utf-8") if l.strip()]
        det_id = self.tokenizer.convert_tokens_to_ids("[DET]")
        det_tok= self.tokenizer.convert_ids_to_tokens(det_id)
        subs   = []
        for l in lines:
            parts = l.split(maxsplit=1)
            try:
                cid = int(parts[0])
            except:
                continue
            txt = parts[1] if len(parts)>1 else ""
            ch  = self.num_to_label(cid)
            subs.append(f"{ch}: {txt}")
        if not subs:
            return None

        answer = self.tokenizer.bos_token + " " + det_tok + " " + " ".join(subs) + " " + self.tokenizer.eos_token
        full    = f"{prompt} {answer}"
        tok_full= self.tokenizer(full, return_tensors="pt", max_length=1024, truncation=True)
        input_ids      = tok_full.input_ids
        attention_mask = tok_full.attention_mask

        tok_pr = self.tokenizer(prompt, return_tensors="pt", max_length=1024, truncation=True)
        plen   = tok_pr.input_ids.size(1)
        labels = input_ids.clone()
        labels[0, :plen] = -100

        return (
            image_clip,    
            image_yolo,    
            input_ids,
            attention_mask,
            labels,
            proc_boxes,    
            class_ids,       
            nb,
            img_path
        )



def collate_fn(batch):
    batch = [b for b in batch if b]
    if not batch:
        return None
    clips, yolos, ids, masks, labs, bbs, class_ids_list, nums, paths = zip(*batch)
    return (
        torch.stack(clips,0),
        torch.stack(yolos,0),
        torch.stack(ids,0).squeeze(1),
        torch.stack(masks,0).squeeze(1),
        torch.stack(labs,0).squeeze(1),
        torch.stack(bbs,0),
        torch.stack(class_ids_list,0),
        torch.tensor(nums, dtype=torch.int32),
        list(paths),
    )


def parse_args(args):
    p = argparse.ArgumentParser(description="FigEx Object Detection Training with LoRA")
    p.add_argument(
        "--cuda-device",
        type=str,
        required=True,
        help="Which CUDA device to use, e.g. 'cuda:0'"
    )
    p.add_argument(
        "--output-path",
        type=str,
        required=True,
        help="Directory where to save checkpoints and logs"
    )
    p.add_argument("--ckpt-path",        type=str, required=True)
    p.add_argument("--yolos-path",        type=str, required=True)
    p.add_argument("--dataset-path",     type=str, required=True)
    p.add_argument("--val-dataset-path", type=str, required=True)
    p.add_argument("--batch-size",       type=int, default=4)
    p.add_argument("--epochs",           type=int, default=10)
    p.add_argument("--lr",               type=float, default=3e-4)
    p.add_argument("--weight-decay",     type=float, default=0.01)
    p.add_argument("--num-workers",      type=int, default=4)
    p.add_argument("--lora-r",           type=int, default=8)
    p.add_argument("--lora-alpha",       type=int, default=16)
    p.add_argument("--lora-dropout",     type=float, default=0.05)
    p.add_argument(
        "--lora-target-modules", type=str,
        default="q_proj,v_proj",
        help="Comma-separated list of modules to apply LoRA",
    )
    return p.parse_args(args)


def count_parameters(model):
    tot = sum(p.numel() for p in model.parameters())
    trn = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Params: {tot:,}")
    print(f"Trainable Params: {trn:,}")
    print(f"Ratio: {trn/tot:.4f}")


def main(argv):
    args = parse_args(argv)

    device = torch.device(args.cuda_device)
    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(args.ckpt_path, use_fast=True)
    tokenizer.pad_token = tokenizer.unk_token
    det_idx = tokenizer("[DET]", add_special_tokens=False).input_ids[0]

    base_model = FigExForCausalLM.from_pretrained(
        args.ckpt_path,
        torch_dtype=torch.float32,
        det_token_idx=det_idx,
        yolos_path= args.yolos_path,
    )

    lora_adapter_path = os.path.join(args.ckpt_path, "lora_adapter")
    model = PeftModel.from_pretrained(base_model, lora_adapter_path)
    

    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.eos_token_id = tokenizer.eos_token_id

    model.get_model().initialize_FigEx_modules(model.get_model().config)
    
    for n, p in model.named_parameters():
        if (
            "lora_" in n
            or any(k in n for k in [
                "lm_head", 
                "model.embed_tokens"
                "model.text_hidden_fcs",
                "base_model.model.model.visual_model.class_labels_classifier",
                "base_model.model.model.visual_model.bbox_predictor",
                "mm_projector",   #"vision_tower.vision_model.encoder.layers.23.layer_norm2", "post_layernorm", "fusion_layers.0",
            ]) 
        ):
            p.requires_grad = True
            print(n)
        else:
            p.requires_grad = False
            
    model.to(device)
    
    '''
    with open("model_weights.txt", "w") as f:
        for name, param in model.named_parameters():
            f.write(f"{name}\t{tuple(param.shape)}\n")
    '''
            
    count_parameters(model)

    # Initialize separate processors
    clip_processor = CLIPImageProcessor.from_pretrained(model.config.vision_tower)
    yolo_processor = YolosImageProcessor.from_pretrained(args.yolos_path)

    # Build datasets and loaders
    train_ds = YOLOObjectDetectionDataset(
        args.dataset_path, tokenizer, clip_processor, yolo_processor
    )
    val_ds   = YOLOObjectDetectionDataset(
        args.val_dataset_path, tokenizer, clip_processor, yolo_processor
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn
    )

    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    scaler = GradScaler()

    for epoch in range(1, args.epochs + 1):
        
        #k = 0
        
        
        
        
        # --- Training ---
        model.train()
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", leave=False)
        
        for batch in train_pbar:
            #k = k + 1
            #if k > 1:
                #break            
            if batch is None:
                continue
            clip_imgs, yolo_imgs, input_ids, attn_mask, labels, bboxes, class_ids, nums, paths = batch
            clip_imgs = clip_imgs.to(device)
            yolo_imgs = yolo_imgs.to(device)
            input_ids = input_ids.to(device)
            attn_mask = attn_mask.to(device)
            labels    = labels.to(device)
            gt_bboxes = bboxes[0, : nums[0].item(), :].to(device)
            gt_labels = class_ids[0, : nums[0].item()].to(device)

            optimizer.zero_grad()
            with autocast():
                text_out, det_losses, _ = model(
                    images_clip=clip_imgs,
                    images_yolo=yolo_imgs,
                    input_ids=input_ids,
                    attention_masks=attn_mask,
                    labels=labels,
                    image_path=paths[0],
                    tokenizer=tokenizer,
                    gt_bboxes=gt_bboxes,
                    class_labels=gt_labels,
                )
                # only optimize text loss if det_losses == 0
                if det_losses == 0:
                    loss = text_out.loss
                    det_losses = torch.tensor(0.0, device=text_out.loss.device)
                else:
                    loss = 0.5 * text_out.loss + 0.5 * det_losses
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_pbar.set_postfix({
                'text_loss': f'{text_out.loss.item():.4f}',
                'det_loss' : f'{det_losses.item():.4f}',
            })

        # --- Validation ---
        
        #k = 0
        
        model.eval()
        val_pbar = tqdm(val_loader, desc=f"Validating Epoch {epoch}", leave=False)
        all_val_text, all_val_ce, all_val_bbox, all_val_iou, all_val_vis = [], [], [], [], []
        with torch.no_grad():
            for batch in val_pbar:
                #k = k + 1
                #if k > 1:
                    #break
                if batch is None:
                    continue
                clip_imgs, yolo_imgs, input_ids, attn_mask, labels, bboxes, class_ids, nums, paths = batch
                clip_imgs = clip_imgs.to(device)
                yolo_imgs = yolo_imgs.to(device)
                input_ids = input_ids.to(device)
                attn_mask = attn_mask.to(device)
                labels    = labels.to(device)
                gt_bboxes = bboxes[0, : nums[0].item(), :].to(device)
                gt_labels = class_ids[0, : nums[0].item()].to(device)

                with autocast():
                    # unpack det_losses_dict
                    text_out, det_losses, det_losses_dict = model(
                        images_clip=clip_imgs,
                        images_yolo=yolo_imgs,
                        input_ids=input_ids,
                        attention_masks=attn_mask,
                        labels=labels,
                        image_path=paths[0],
                        tokenizer=tokenizer,
                        gt_bboxes=gt_bboxes,
                        class_labels=gt_labels,
                    )
                    # if no DET token, zero out losses
                    if det_losses == 0:
                        ce_val  = 0.0
                        bbox_val = 0.0
                        iou_val  = 0.0
                        vis_val  = 0.0
                    else:
                        ce_val  = det_losses_dict['loss_ce'].item()
                        bbox_val = det_losses_dict['loss_bbox'].item()
                        iou_val  = det_losses_dict['loss_giou'].item()
                        vis_val  = det_losses.item()

                all_val_text.append(text_out.loss.item())
                all_val_ce.append(ce_val)
                all_val_bbox.append(bbox_val)
                all_val_iou.append(iou_val)
                all_val_vis.append(vis_val)

        avg_val_text = sum(all_val_text) / len(all_val_text) if all_val_text else 0.0
        avg_val_ce  = sum(all_val_ce) / len(all_val_ce) if all_val_ce else 0.0
        avg_val_bbox = sum(all_val_bbox) / len(all_val_bbox) if all_val_bbox else 0.0
        avg_val_iou  = sum(all_val_iou) / len(all_val_iou) if all_val_iou else 0.0
        avg_val_vis  = sum(all_val_vis) / len(all_val_vis) if all_val_vis else 0.0

        print(
            f"Epoch {epoch} Val ▶ "
            f"text={avg_val_text:.4f} vision={avg_val_vis:.4f} "
            f"ce={avg_val_ce:.4f} bbox={avg_val_bbox:.4f} iou={avg_val_iou:.4f}"
        )

        # Save checkpoint
        save_dir = os.path.join(args.output_path, f"epoch_{epoch}")
        os.makedirs(save_dir, exist_ok=True)
        model.model.save_pretrained(save_dir)
        model.save_pretrained(os.path.join(save_dir, "lora_adapter"))
        tokenizer.save_pretrained(save_dir)
        with open(os.path.join(save_dir, "losses.json"), "w") as f:
            json.dump({
                "train_loss":  loss.item(),
                "val_text_loss":   avg_val_text,
                "val_loss_ce":    avg_val_ce,
                "val_loss_bbox":   avg_val_bbox,
                "val_loss_iou":    avg_val_iou,
                "val_vision_loss": avg_val_vis,
            }, f, indent=2)

    print("Training complete.")


if __name__ == "__main__":
    main(sys.argv[1:])
