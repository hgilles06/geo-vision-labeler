from PIL import Image
from llms.llm_abc import VisionLLM
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
from llms.utils import convert_pil_image2base64


class QwenVLLM(VisionLLM):
    def __init__(self, model: str = "AdaptLLM/remote-sensing-Qwen2.5-VL-3B-Instruct", device: str = "cuda:1"):
        torch.cuda.empty_cache()  # Clear GPU memory
        self.device = torch.device(device)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model,
            torch_dtype="auto",
            device_map={"": self.device},
            load_in_4bit=True,  # 4-bit quantization
            low_cpu_mem_usage=True
        )
        self.processor = AutoProcessor.from_pretrained(model, use_fast=True)

    def call_vision_llm(self, image: Image.Image, prompt: str) -> str:
        image_b64 = convert_pil_image2base64(image)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": f"data:image/png;base64,{image_b64}"},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.device)
        generated_ids = self.model.generate(**inputs, max_new_tokens=128)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return output_text