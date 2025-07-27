from PIL import Image
from llms.llm_abc import VisionLLM
import torch
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
from llms.utils import convert_pil_image2base64, clean_output_text

class LlavaVLLM(VisionLLM):
    def __init__(self, model: str = "AdaptLLM/remote-sensing-LLaVA-NeXT-Llama3-8B"):
        self.model = LlavaNextForConditionalGeneration.from_pretrained(model, torch_dtype=torch.float16, device_map="auto")
        self.processor = LlavaNextProcessor.from_pretrained(model)
    
    def call_vision_llm(self, image: Image.Image, prompt: str) -> str:
        image = image.convert("RGB")
        # Validate image
        if not isinstance(image, Image.Image):
            raise ValueError("Input must be a valid PIL Image")

        # Convert image to base64
        image_b64 = convert_pil_image2base64(image)
        # Structured message format with base64 image
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": f"data:image/png;base64,{image_b64}"},
                    {"type": "text", "text": prompt}
                ]
            }
        ]
        # Apply chat template
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        # Process text and image together (pass PIL image for pixel_values)
        inputs = self.processor(
            text=text,
            images=[image],  # Pass PIL image to ensure image_sizes
            return_tensors="pt",
            padding=True
        ).to(self.model.device)

        if "pixel_values" not in inputs:
            raise ValueError("No pixel values generated for the image")
        if "image_sizes" not in inputs:
            raise ValueError("No image sizes generated")

        # Generate output
        output = self.model.generate(**inputs, max_new_tokens=512)
        # Decode the output, skipping the input tokens
        answer_start = inputs["input_ids"].shape[-1]
        pred = self.processor.decode(output[0][answer_start:], skip_special_tokens=True)
        return clean_output_text(pred)