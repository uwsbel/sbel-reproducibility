#!/usr/bin/env python3
import argparse
import os
import sys

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

import groundingdino.datasets.transforms as T
from groundingdino.models import build_model
from groundingdino.util import box_ops
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap
from groundingdino.util.vl_utils import create_positive_map_from_span


def plot_boxes_to_image(image_pil, tgt):
    H, W = tgt["size"]
    boxes = tgt["boxes"]
    labels = tgt["labels"]
    assert len(boxes) == len(labels), "boxes and labels must have same length"

    draw = ImageDraw.Draw(image_pil)
    mask = Image.new("L", image_pil.size, 0)
    mask_draw = ImageDraw.Draw(mask)

    for box, label in zip(boxes, labels):
        # box is [cx, cy, w, h] normalized
        box = box * torch.Tensor([W, H, W, H])
        # xywh -> xyxy
        x_c, y_c, bw, bh = box.tolist()
        x0 = int(x_c - bw / 2)
        y0 = int(y_c - bh / 2)
        x1 = int(x_c + bw / 2)
        y1 = int(y_c + bh / 2)

        color = tuple(np.random.randint(0, 255, size=3).tolist())
        draw.rectangle([x0, y0, x1, y1], outline=color, width=3)

        font = ImageFont.load_default()
        if hasattr(font, "getbbox"):
            tb = draw.textbbox((x0, y0), label, font=font)
        else:
            w, h = draw.textsize(label, font=font)
            tb = (x0, y0, x0 + w, y0 + h)
        draw.rectangle(tb, fill=color)
        draw.text((x0, y0), label, fill="white", font=font)

        mask_draw.rectangle([x0, y0, x1, y1], fill=255, width=3)

    return image_pil, mask


def load_image(image_path):
    image_pil = Image.open(image_path).convert("RGB")
    transform = T.Compose([
        T.RandomResize([800], max_size=1333),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406],
                    [0.229, 0.224, 0.225]),
    ])
    image_tensor, _ = transform(image_pil, None)
    return image_pil, image_tensor


def load_model(model_config_path: str,
               model_checkpoint_path: str,
               cpu_only: bool = False):
    args = SLConfig.fromfile(model_config_path)
    args.device = "cuda" if not cpu_only else "cpu"
    model = build_model(args)
    checkpoint = torch.load(model_checkpoint_path,
                            map_location="cpu")
    model.load_state_dict(
        clean_state_dict(checkpoint["model"]),
        strict=False
    )
    model.eval()
    return model


def get_grounding_output(model,
                         image: torch.Tensor,
                         caption: str,
                         box_threshold: float,
                         text_threshold: float = None,
                         cpu_only: bool = False,
                         token_spans=None):
    assert (text_threshold is not None) or (token_spans is not None), \
        "Either text_threshold or token_spans must be set"
    caption = caption.lower().strip()
    if not caption.endswith("."):
        caption += "."

    device = "cpu" if cpu_only else "cuda"
    model = model.to(device)
    image = image.to(device)

    with torch.no_grad():
        outputs = model(image[None], captions=[caption])

    logits = outputs["pred_logits"].sigmoid()[0]  # (nq, 256)
    boxes  = outputs["pred_boxes"][0]             # (nq, 4)

    # filter by max token logit > box_threshold
    logits_f = logits.cpu().clone()
    boxes_f  = boxes.cpu().clone()
    keep = logits_f.max(dim=1)[0] > box_threshold
    logits_f = logits_f[keep]
    boxes_f  = boxes_f[keep]

    # decode phrases per box
    tokenlizer = model.tokenizer
    tokenized = tokenlizer(caption)
    phrases = []
    for logit_row in logits_f:
        mask = logit_row > (text_threshold if text_threshold is not None else 0.0)
        phrase = get_phrases_from_posmap(mask, tokenized, tokenlizer)
        phrases.append(phrase)

    return boxes_f, phrases


if __name__ == "__main__":
    p = argparse.ArgumentParser("GroundingDINO Inference")
    p.add_argument("-c", "--config_file",    required=True)
    p.add_argument("-p", "--checkpoint_path",required=True)
    p.add_argument("-i", "--image_path",     required=True)
    p.add_argument("-t", "--text_prompt",    default="all objects")
    p.add_argument("-o", "--output_dir",     default="outputs")
    p.add_argument("--box_threshold", type=float, default=0.3)
    p.add_argument("--text_threshold",type=float, default=0.25)
    p.add_argument("--cpu_only", action="store_true")
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # 1) load image & model
    image_pil, image = load_image(args.image_path)
    model = load_model(args.config_file,
                       args.checkpoint_path,
                       cpu_only=args.cpu_only)

    # 2) run grounding
    boxes, labels = get_grounding_output(
        model,
        image,
        args.text_prompt,
        box_threshold = args.box_threshold,
        text_threshold = args.text_threshold if args.text_threshold else None,
        cpu_only = args.cpu_only
    )

    # 3) draw + save
    W, H = image_pil.size
    pred = {
        "size": [H, W],         # H, W
        "boxes": boxes,         # Tensor[N,4]
        "labels": labels        # List[str] length N
    }
    out_img, _ = plot_boxes_to_image(image_pil.copy(), pred)
    out_img.save(os.path.join(args.output_dir, "pred.jpg"))
    print(f"Saved {os.path.join(args.output_dir,'pred.jpg')}")




