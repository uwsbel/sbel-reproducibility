# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.

from typing import Any, Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from vlfm.chrono_env.blip2_image_text_matching_custom_itc import Blip2ITM
from .server_wrapper import ServerMixin, host_model, send_request, str_to_image
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

class BLIP2ITM:
    """BLIP 2 Image-Text Matching model."""

    def __init__(self, name: str = "blip2_image_text_matching_custom_itc", model_type: str = "pretrain", device: Optional[Any] = None,) -> None:
        if device is None:
            device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
        self.device = device

        self.model = Blip2ITM.from_pretrained(model_type=model_type)
        self.model.eval().to(self.device)

        cfg_path = Blip2ITM.default_config_path(model_type)
        preprocess_cfg = OmegaConf.load(cfg_path).preprocess
        self.vis_processors, self.text_processors = load_preprocess(preprocess_cfg)

    def infer(self, image: np.ndarray, txt: str) -> float:
        """
        Run inference on the model to compute cosine similarity between an image and a text prompt
        and extract image features projected to language space.

        Args:
            image (numpy.ndarray): The input image as a numpy array.
            txt (str): The text to compare the image to.

        Returns:
            float: The cosine similarity between the image and the prompt.
            tensor: The image features extracted by the model projected to language space.
        """
        pil_img = Image.fromarray(image)
        img = self.vis_processors["eval"](pil_img).unsqueeze(0).to(self.device)
        txt = self.text_processors["eval"](txt)
        with torch.inference_mode():
            out = self.model({"image": img, "text_input": txt}, match_head="itc")

        return out

class BLIP2ITMClient:
    def __init__(self, port: int = 12182, device=None):
        self.url = f"http://localhost:{port}/blip2itm"
        self.device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))

    def infer(self, image: np.ndarray, txt: str) -> float:
        # print(f"Robot {robot_id}: BLIP2ITMClient.infer: {txt}")
        out = send_request(self.url, image=image, txt=txt)
        feats = torch.tensor(out["image_feats"], device=self.device)
        return out["score"], feats


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=12182)
    args = parser.parse_args()

    print("Loading model...")

    class BLIP2ITMServer(ServerMixin, BLIP2ITM):
        def process_payload(self, payload):
            img = str_to_image(payload["image"])
            # out = self.model({"image": img, "text_input": payload["txt"]}, match_head="itc")
            out = self.infer(img, payload["txt"])
            return {
                "score":          out["score"].item(),
                "image_feats":    out["image_feats"].squeeze(0).cpu().tolist(),
            }

    blip = BLIP2ITMServer()
    print("Model loaded!")
    print(f"Hosting on port {args.port}...")
    host_model(blip, name="blip2itm", port=args.port)
