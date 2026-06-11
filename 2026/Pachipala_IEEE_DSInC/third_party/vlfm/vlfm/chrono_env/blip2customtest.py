#!/usr/bin/env python3
import sys
import torch
from PIL import Image
# from lavis.processors import load_preprocess
# from lavis.models import load_model_and_preprocess
from blip2itm_custom import Blip2ITM
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

    vis_processors["train"] = _build_proc_from_cfg(vis_proc_cfg.get("train") if vis_proc_cfg else None)
    vis_processors["eval"] = _build_proc_from_cfg(vis_proc_cfg.get("eval") if vis_proc_cfg else None)

    txt_processors["train"] = _build_proc_from_cfg(txt_proc_cfg.get("train") if txt_proc_cfg else None)
    txt_processors["eval"] = _build_proc_from_cfg(txt_proc_cfg.get("eval") if txt_proc_cfg else None)

    return vis_processors, txt_processors

class BLIP2ITMWrapper:
    """BLIP-2 Image-Text Matching via LAVIS."""
    def __init__(self,
                 name: str = "blip2_image_text_matching",
                 model_type: str = "pretrain",
                 device: torch.device = None):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        # Load pretrained model weights
        self.model = Blip2ITM.from_pretrained(model_type=model_type)
        self.model.eval().to(self.device)

        # Load processor config
        cfg_path = Blip2ITM.default_config_path(model_type)
        preprocess_cfg = OmegaConf.load(cfg_path).preprocess

        # Load vision and text processors
        self.vis_processors, self.text_processors = load_preprocess(preprocess_cfg)

    def cosine(self, image_path: str, txt: str) -> float:
        # 1) load & preprocess image
        pil_img    = Image.open(image_path).convert("RGB")
        img_tensor = self.vis_processors["eval"](pil_img).unsqueeze(0).to(self.device)

        # 2) preprocess text (no .to(), no .unsqueeze here)
        txt_input = self.text_processors["eval"](txt)

        # 3) run the ITM head
        with torch.no_grad():
            out = self.model(
                {"image": img_tensor, "text_input": txt_input},
                match_head="itc"
            )

        score = out["score"].item()
        image_feats = out["image_feats"]
        
        return score, image_feats

def main():
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} /path/to/image.png \"Your prompt here\"")
        sys.exit(1)

    image_path = sys.argv[1]
    prompt     = sys.argv[2]

    itm = BLIP2ITMWrapper()
    score, image_feats = itm.cosine(image_path, prompt)
    answer = "yes" if score > 0.3 else "no"

    print(f"Prompt: \"{prompt}\"")
    print(f"ITM score: {score:.3f} → {answer}")

if __name__ == "__main__":
    main()
