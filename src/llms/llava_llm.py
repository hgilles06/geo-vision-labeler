from PIL import Image
from llms.llm_abc import VisionLLM
import torch
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
from llms.utils import clean_output_text

class LlavaVLLM(VisionLLM):
    def __init__(self, model: str = "AdaptLLM/remote-sensing-LLaVA-NeXT-Llama3-8B"):
        self.model = LlavaNextForConditionalGeneration.from_pretrained(model, torch_dtype=torch.float16, device_map="auto")
        self.processor = LlavaNextProcessor.from_pretrained(model)
    
    def call_vision_llm(self, image: Image.Image, prompt: str) -> str:
        formatted_prompt = (
            f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
            f"You are a helpful language and vision assistant. You are able to understand the visual content that the user provides, and assist the user with a variety of tasks using natural language."
            f"<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
            f"<image>\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        )
        inputs = self.processor(images=image, text=formatted_prompt, return_tensors="pt").to(self.model.device)
        answer_start = int(inputs["input_ids"].shape[-1])
        output = self.model.generate(**inputs, max_new_tokens=512)

        # Decode predictions
        pred = self.processor.decode(output[0][answer_start:], skip_special_tokens=True)
        return clean_output_text(pred)