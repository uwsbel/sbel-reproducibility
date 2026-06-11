#!/usr/bin/env python3
import sys
import os
import torch
import math
import torch.nn.functional as F
from PIL import Image
from blip2_image_text_matching_custom_itc import Blip2ITM
from omegaconf import OmegaConf
from lavis.processors.base_processor import BaseProcessor
from lavis.common.registry import registry

def load_preprocess(preprocess_cfg):
    def _build_proc_from_cfg(cfg):
        return (
            registry.get_processor_class(cfg.name).from_config(cfg)
            if cfg is not None
            else BaseProcessor()
        )

    vis_processors = dict()
    txt_processors = dict()

    vis_proc_cfg = preprocess_cfg.get("vis_processor")
    txt_proc_cfg = preprocess_cfg.get("text_processor")

    vis_processors["eval"] = _build_proc_from_cfg(vis_proc_cfg.get("eval") if vis_proc_cfg else None)
    txt_processors["eval"] = _build_proc_from_cfg(txt_proc_cfg.get("eval") if txt_proc_cfg else None)

    return vis_processors, txt_processors

class BLIP2ITMWrapper:
    def __init__(self, name: str = "blip2_image_text_matching", model_type: str = "pretrain", device: torch.device = None):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        self.model = Blip2ITM.from_pretrained(model_type=model_type)
        self.model.eval().to(self.device)

        cfg_path = Blip2ITM.default_config_path(model_type)
        preprocess_cfg = OmegaConf.load(cfg_path).preprocess
        self.vis_processors, self.text_processors = load_preprocess(preprocess_cfg)

    def extract_feats(self, image_path: str, dummy_text: str = "room") -> torch.Tensor:
        pil_img = Image.open(image_path).convert("RGB")
        img_tensor = self.vis_processors["eval"](pil_img).unsqueeze(0).to(self.device)
        txt_input = self.text_processors["eval"](dummy_text)

        with torch.no_grad():
            out = self.model({"image": img_tensor, "text_input": txt_input}, match_head="itc")
        # returns shape (1, 32, 256) -> squeeze -> (32, 256)
        return out["image_feats"].squeeze(0) # (32, 256)

def is_image_file(fn: str) -> bool:
    return fn.lower().endswith((".jpg", ".jpeg", ".png"))

def per_query_gated_fusion(Qb: torch.Tensor, Qn: torch.Tensor) -> torch.Tensor:
    # Qb, Qn: (K, D)
    K, D = Qb.shape

    # 1) compute per-slot dot / sqrt(D)
    scores = (Qb * Qn).sum(dim=1) / math.sqrt(D)     # (K,)

    # 2) gate via sigmoid
    gates = torch.sigmoid(scores).unsqueeze(1)      # (K,1)

    # 3) novelty weight
    novelty = (1 - gates) * Qn                     # (K, D)

    # 4) residual fuse + optional layer‐norm
    fused = Qb + novelty
    return fused

def compute_cosine_similarity(feat1: torch.Tensor, feat2: torch.Tensor) -> float:
    # flatten (32,256) -> (1, 8192) -> unsqueeze (1,1,8192) and (1,8192,1)
    feat1 = F.normalize(feat1.view(1, -1), dim=-1).unsqueeze(1)  # (1, 1, 8192)
    feat2 = F.normalize(feat2.view(1, -1), dim=-1).unsqueeze(2)  # (1, 8192, 1)   
    sim = torch.bmm(feat1, feat2).mean().item()
    return sim

def main():
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} /path/to/images_root /path/to/test_images_dir")
        sys.exit(1)

    images_root, test_dir = sys.argv[1], sys.argv[2]
    if not os.path.isdir(images_root) or not os.path.isdir(test_dir):
        print("Both arguments must be existing directories.")
        sys.exit(1)

    itm = BLIP2ITMWrapper()

    room_feature_db = {}  # (house_id, room_id) -> fused_feats

    # 1. Traverse all house dirs starting with '8'
    for house_name in sorted(os.listdir(images_root)):
        if not house_name.startswith("8"):
            continue
        house_path = os.path.join(images_root, house_name)
        if not os.path.isdir(house_path):
            continue

        # 2. Traverse all rooms inside this house
        for room_name in sorted(os.listdir(house_path)):
            room_path = os.path.join(house_path, room_name)
            if not os.path.isdir(room_path):
                continue

            fused_feats = None
            for fn in sorted(os.listdir(room_path)):
                img_path = os.path.join(room_path, fn)
                if not is_image_file(img_path):
                    continue

                feats = itm.extract_feats(img_path)  # (32, 256)
                if fused_feats is None:
                    fused_feats = feats.clone()
                    D = fused_feats.shape[1]
                    num_heads = 8 if D % 8 == 0 else 1

                else:
        
                    fused_feats = per_query_gated_fusion(fused_feats, feats)  # paired gated fusion (best performer)
                    # fused_feats = torch.max(fused_feats, feats)             # max-pool (slightly better than EMA)
                    # fused_feats = (fused_feats + feats) * 0.5               # EMA (worst performer)
            
            if fused_feats is not None:
                room_feature_db[(house_name, room_name)] = fused_feats
                

    if not room_feature_db:
        print("No valid room features found.")
        sys.exit(1)

    # 3. Compare each test image to all room features
    for fn in sorted(os.listdir(test_dir)):
        test_path = os.path.join(test_dir, fn)
        if not is_image_file(test_path):
            continue
        test_feats = itm.extract_feats(test_path)

        print(f"Scores for {fn}:")
        for (house_name, room_name), feats in room_feature_db.items():
            score = compute_cosine_similarity(feats, test_feats)
            print(f"  {house_name}/{room_name:20s}  cosine similarity = {score:.4f}")
        print()



if __name__ == "__main__":
    main()
