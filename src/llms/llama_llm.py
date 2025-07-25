from PIL import Image

from transformers import AutoProcessor
from llms.llm_abc import VisionLLM
from transformers import MllamaForConditionalGeneration, AutoProcessor
import torch

class LlamaVLLM(VisionLLM):
    def __init__(self, model: str = "AdaptLLM/remote-sensing-Llama-3.2-11B-Vision-Instruct"):
        self.model = MllamaForConditionalGeneration.from_pretrained(
            model,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        self.processor = AutoProcessor.from_pretrained(model)
    
    def call_vision_llm(self, image: Image.Image, prompt: str) -> str:
        messages = [
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": prompt}
            ]}
        ]
        input_text = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self.processor(
            image,
            input_text,
            add_special_tokens=False,
            return_tensors="pt"
        ).to(self.model.device)

        output = self.model.generate(**inputs, max_new_tokens=512)
        return self.processor.decode(output[0])