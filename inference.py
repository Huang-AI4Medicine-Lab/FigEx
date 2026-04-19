#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Resep-YOLOS batch inference  ·  require --yolos-path
"""

import os
import cv2
import shutil
import argparse
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm
from transformers import AutoTokenizer, CLIPImageProcessor, YolosImageProcessor
from peft import PeftModel
from model.FigEx_yolos_mix import FigExForCausalLM
from utils.utils import (
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IMAGE_TOKEN,
    IMAGE_TOKEN_INDEX,
)
from torch.cuda.amp import autocast

import re
from typing import Dict, List, Tuple

# ---------- Dataset ---------- #
class YOLOObjectDetectionDataset(Dataset):
    """Inference dataset: prompt only (“Input Caption … Output:”)."""
    def __init__(self, root, tokenizer, clip_proc, yolo_proc):
        self.img_dir  = os.path.join(root, "images")
        self.cap_dir  = os.path.join(root, "captions")
        self.tok, self.clip_proc, self.yolo_proc = tokenizer, clip_proc, yolo_proc
        self.files = sorted(
            f for f in os.listdir(self.img_dir)
            if f.lower().endswith((".jpg", ".png"))
        )
        self.marker = (
            f"{DEFAULT_IM_START_TOKEN} "
            f"{DEFAULT_IMAGE_TOKEN} "
            f"{DEFAULT_IM_END_TOKEN}"
        )

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fn = self.files[idx]
        img_p = os.path.join(self.img_dir, fn)
        cap_p = os.path.join(self.cap_dir, os.path.splitext(fn)[0] + ".txt")

        caption = open(cap_p, encoding="utf-8").read().strip()
        prompt  = f"{self.marker} Input Caption: {caption}\nOutput:"
        tok_out = self.tok(prompt, return_tensors="pt")
        ids, mask = tok_out.input_ids[0], tok_out.attention_mask[0]

        pil = Image.open(img_p).convert("RGB")
        H, W = pil.height, pil.width

        rgb  = cv2.cvtColor(cv2.imread(img_p), cv2.COLOR_BGR2RGB)
        clip = self.clip_proc(images=rgb, return_tensors="pt")["pixel_values"][0]
        yolo = self.yolo_proc(images=[pil], return_tensors="pt")["pixel_values"][0]

        return dict(
            image_clip=clip,
            image_yolo=yolo,
            input_ids=ids,
            attention_mask=mask,
            path=img_p,
            orig_size=(H, W),
        )

def collate_fn(batch):
    clip = torch.stack([x["image_clip"] for x in batch])
    yolo = torch.stack([x["image_yolo"] for x in batch])
    ids  = torch.nn.utils.rnn.pad_sequence(
        [x["input_ids"] for x in batch],
        batch_first=True, padding_value=0
    )
    msk  = torch.nn.utils.rnn.pad_sequence(
        [x["attention_mask"] for x in batch],
        batch_first=True, padding_value=0
    )
    paths = [x["path"] for x in batch]
    sizes = [x["orig_size"] for x in batch]
    return clip, yolo, ids, msk, paths, sizes

# ---------- CLI ---------- #
def get_args():
    p = argparse.ArgumentParser("Resep-YOLOS inference")
    p.add_argument("--cuda-device", required=True)
    p.add_argument("--ckpt-path",  required=True, help="checkpoint dir")
    p.add_argument("--yolos-path", required=True, help="YOLOS backbone path")
    p.add_argument("--input-path", required=True)
    p.add_argument("--output-path",required=True)
    p.add_argument("--batch-size",  type=int, default=4)
    p.add_argument("--threshold",   type=float, default=0.02)
    p.add_argument("--max-new-tokens", type=int, default=450)
    return p.parse_args()

def main():
    args = get_args()
    dev = torch.device(args.cuda_device)

    # --- tokenizer & model ---
    tok = AutoTokenizer.from_pretrained(args.ckpt_path, use_fast=True)
    tok.pad_token = tok.unk_token
    det_idx = tok("[DET]", add_special_tokens=False).input_ids[0]

    base = FigExForCausalLM.from_pretrained(
        args.ckpt_path,
        det_token_idx=det_idx,
        yolos_path=args.yolos_path,
    )
    model = PeftModel.from_pretrained(
        base, os.path.join(args.ckpt_path, "lora_adapter")
    ).to(dev).eval()

    # --- processors ---
    clip_proc = CLIPImageProcessor.from_pretrained(base.config.vision_tower)
    yolo_proc = YolosImageProcessor.from_pretrained(args.yolos_path)
    if hasattr(model, "image_processor"):
        model.image_processor = yolo_proc

    # --- data ---
    ds = YOLOObjectDetectionDataset(args.input_path, tok, clip_proc, yolo_proc)
    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        collate_fn=collate_fn,
        shuffle=False,
        pin_memory=True,
    )

    # --- output dirs ---
    out_det = os.path.join(args.output_path, "det")
    out_cap = os.path.join(args.output_path, "cap")
    for d in (out_det, out_cap):
        os.makedirs(d, exist_ok=True)
        
    model.eval()

    # --- inference ---
    with torch.no_grad():
        for i, (clips, yolos, ids, masks, paths, sizes) in enumerate(tqdm(dl), 1):

            try:                            
                clips, yolos = clips.to(dev), yolos.to(dev)
                ids, masks   = ids.to(dev), masks.to(dev)

                with autocast():
                    dets, decoded_text = model.evaluate(
                        images_clip    = clips,
                        images_yolo    = yolos,
                        input_ids      = ids,
                        attention_mask = masks,
                        max_new_tokens = args.max_new_tokens,
                        threshold      = args.threshold,
                        target_sizes   = sizes,
                        tok            = tok,
                    )
            except RuntimeError as e:
                if "CUDA out of memory" in str(e):
                    #torch.cuda.empty_cache()
                    print(f"[Batch {i}] OOM, skipped.")
                    continue
                raise

            if len(dets) == 0:
                for p in paths:
                    stem = os.path.splitext(os.path.basename(p))[0]
                    open(os.path.join(out_det, stem + ".txt"), "w").close()
                    open(os.path.join(out_cap, stem + ".txt"), "w").close()
                del dets, decoded_text
                continue

            detections = yolo_proc.post_process_object_detection(
                outputs      = dets,
                threshold    = args.threshold,
                target_sizes = sizes,
            )

            for det, p, (H, W) in zip(detections, paths, sizes):
                stem = os.path.splitext(os.path.basename(p))[0]

                best_per_class = {}
                for score, cls, box in zip(det["scores"], det["labels"], det["boxes"]):
                    if score < args.threshold:
                        continue
                    cid = cls.item(); val = score.item(); coords = box.tolist()
                    if cid not in best_per_class or val > best_per_class[cid][0]:
                        best_per_class[cid] = (val, coords)

                with open(os.path.join(out_det, stem + ".txt"), "w") as f:
                    for cid, (_, (x0, y0, x1, y1)) in best_per_class.items():
                        xc = ((x0 + x1) / 2) / W
                        yc = ((y0 + y1) / 2) / H
                        bw = (x1 - x0) / W
                        bh = (y1 - y0) / H
                        f.write(f"{cid} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n")

                m = re.search(r'[A-Z]:', decoded_text)
                trimmed = decoded_text[m.start():] if m else decoded_text
                parts = re.split(r'\s*([A-Z]):\s*', trimmed)
                subcaps = [f"{parts[j]}: {parts[j+1].strip()}"
                           for j in range(1, len(parts), 2)]

                with open(os.path.join(out_cap, stem + ".txt"), "w", encoding="utf-8") as f:
                    f.writelines(line + "\n" for line in subcaps)

            del dets, decoded_text, detections  

if __name__ == "__main__":
    main()
