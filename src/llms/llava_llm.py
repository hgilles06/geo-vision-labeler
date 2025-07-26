from PIL import Image
from llms.llm_abc import VisionLLM
import torch
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration

# class LlavaVLLM(VisionLLM):
#     def __init__(self, model: str = "AdaptLLM/remote-sensing-LLaVA-NeXT-Llama3-8B"):
#         self.model = LlavaNextForConditionalGeneration.from_pretrained(model, torch_dtype=torch.float16, device_map="auto")
#         self.processor = LlavaNextProcessor.from_pretrained(model)
    
    # def call_vision_llm(self, image: Image.Image, prompt: str) -> str:
    #     image = image.convert("RGB")
    #     #image_token = "<|reserved_special_token_4|>" ---(creating error :ValueError: Image features and image tokens do not match: tokens: 1175, features 1176 )
    #     complete_prompt = (
    #         f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
    #         f"You are a helpful language and vision assistant. You are able to understand the visual content that the user provides, and assist the user with a variety of tasks using natural language."
    #         f"<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
    #         f"{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    #     )
    #     inputs = self.processor(images=image, text=complete_prompt, return_tensors="pt").to(self.model.device)
    #     answer_start = int(inputs["input_ids"].shape[-1])
    #     output = self.model.generate(**inputs, max_new_tokens=512)

    #     # Decode predictions
    #     pred = self.processor.decode(output[0][answer_start:], skip_special_tokens=True)
    #     return pred
    



from PIL import Image
from llms.llm_abc import VisionLLM
import torch
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
import base64
from io import BytesIO

def convert_pil_image2base64(image: Image.Image) -> str:
    """Convert PIL Image to base64 string."""
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

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

        # Debug inputs
        print(f"Input IDs shape: {inputs['input_ids'].shape}")
        print(f"Pixel values shape: {inputs['pixel_values'].shape}")
        print(f"Image sizes: {inputs.get('image_sizes', 'None')}")
        print(f"Input text: {text}")
        print(f"Input keys: {inputs.keys()}")
        if "pixel_values" not in inputs:
            raise ValueError("No pixel values generated for the image")
        if "image_sizes" not in inputs:
            raise ValueError("No image sizes generated")
        tokenized_text = self.processor.tokenizer.encode(text)
        image_token_id = self.processor.tokenizer.convert_tokens_to_ids("<|reserved_special_token_4|>")
        image_token_count = tokenized_text.count(image_token_id)
        print(f"Image token ID: {image_token_id}, Count: {image_token_count}")

        # Generate output
        output = self.model.generate(**inputs, max_new_tokens=512)
        # Decode the output, skipping the input tokens
        answer_start = inputs["input_ids"].shape[-1]
        pred = self.processor.decode(output[0][answer_start:], skip_special_tokens=True)
        return pred