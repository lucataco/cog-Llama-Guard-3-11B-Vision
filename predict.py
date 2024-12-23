# Prediction interface for Cog ⚙️
# https://cog.run/python

from cog import BasePredictor, Input, Path
import os
import time
import torch
import subprocess
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor

MODEL_CACHE = "checkpoints"
MODEL_URL = "https://weights.replicate.delivery/default/meta-llama/Llama-Guard-3-11B-Vision/model.tar"

def download_weights(url, dest):
    start = time.time()
    print("downloading url: ", url)
    print("downloading to: ", dest)
    subprocess.check_call(["pget", "-xf", url, dest], close_fds=False)
    print("downloading took: ", time.time() - start)

class Predictor(BasePredictor):
    def setup(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.bfloat16
        # download weights
        if not os.path.exists(MODEL_CACHE):
            download_weights(MODEL_URL, MODEL_CACHE)

        self.processor = AutoProcessor.from_pretrained(MODEL_CACHE)
        self.model = AutoModelForVision2Seq.from_pretrained(
            MODEL_CACHE,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )

    @torch.inference_mode()
    def predict(
        self,
        image: Path = Input(description="Image to moderate"),
        prompt: str = Input(description="User message to moderate", default="Which one should I buy?"),
    ) -> str:
        img = Image.open(image).convert("RGB")
        conversation = [{ "role": "user", "content": [{ "type": "text", "text": prompt },{ "type": "image" }] }]
        
        input_prompt = self.processor.apply_chat_template(
            conversation, return_tensors="pt"
        )
        inputs = self.processor(text=input_prompt, images=img, return_tensors="pt").to(self.model.device)
        prompt_len = len(inputs['input_ids'][0])
        output = self.model.generate(
            **inputs,
            max_new_tokens=20,
            pad_token_id=0,
        )
        generated_tokens = output[:, prompt_len:]
        decoded = self.processor.decode(generated_tokens[0])
        output = decoded.replace("<|eot_id|>", "").strip()
        return output
