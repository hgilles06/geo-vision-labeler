from PIL import Image
from transformers import AutoProcessor
from llms.llm_abc import VisionLLM
from transformers import MllamaForConditionalGeneration
import torch
from llms.utils import clean_output_text

class LlamaVLLM(VisionLLM):
    def __init__(self, model: str = "AdaptLLM/remote-sensing-Llama-3.2-11B-Vision-Instruct", device: str = "cuda:1"):
        torch.cuda.empty_cache()  # Clear GPU memory
        self.device = torch.device(device)
        self.model = MllamaForConditionalGeneration.from_pretrained(
            model,
            torch_dtype=torch.bfloat16,
            device_map={"": self.device},
            load_in_4bit=True,  # 4-bit quantization
            low_cpu_mem_usage=True
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
        ).to(self.device)
        
        # Generate output
        output = self.model.generate(**inputs, max_new_tokens=512)
        
        # Trim input tokens and decode, skipping special tokens
        answer_start = inputs["input_ids"].shape[-1]
        decoded_output = self.processor.decode(
            output[0][answer_start:], 
            skip_special_tokens=True, 
            clean_up_tokenization_spaces=True
        )
        return clean_output_text(decoded_output)
