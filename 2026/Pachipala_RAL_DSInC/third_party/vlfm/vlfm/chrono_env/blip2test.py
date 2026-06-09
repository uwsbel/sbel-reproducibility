#!/usr/bin/env python3
import sys
import torch
from PIL import Image
from lavis.models import load_model_and_preprocess

class BLIP2ITM:
    """BLIP-2 Image-Text Matching via LAVIS."""
    def __init__(self,
                 name: str = "blip2_image_text_matching",
                 model_type: str = "pretrain",
                 device: torch.device = None):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # load the vision+ITM head
        self.model, self.vis_processors, self.text_processors = load_model_and_preprocess(
            name=name,
            model_type=model_type,
            is_eval=True,
            device=device,
        )
        self.device = device

    def cosine(self, image_path: str, txt: str) -> float:
        # 1) load & preprocess image
        pil_img    = Image.open(image_path).convert("RGB")
        img_tensor = self.vis_processors["eval"](pil_img).unsqueeze(0).to(self.device)

        # 2) preprocess text (no .to(), no .unsqueeze here)
        txt_input = self.text_processors["eval"](txt)

        # 3) run the ITM head
        with torch.no_grad():
            score = self.model(
                {"image": img_tensor, "text_input": txt_input},
                match_head="itc"
            ).item()

        return score, img_tensor

def main():
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} /path/to/image.png \"Your prompt here\"")
        sys.exit(1)

    image_path = sys.argv[1]
    prompt     = sys.argv[2]

    itm = BLIP2ITM()
    score, img_tensor = itm.cosine(image_path, prompt)
    answer = "yes" if score > 0.3 else "no"

    print(f"Prompt: \"{prompt}\"")
    print(f"ITM score: {score:.3f} → {answer}")
    print(img_tensor, "\n", img_tensor.shape)

if __name__ == "__main__":
    main()
