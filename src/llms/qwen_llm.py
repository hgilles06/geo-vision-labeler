from PIL import Image
from llms.llm_abc import VisionLLM
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info

from llms.utils import convert_pil_image2base64

class QwenVLLM(VisionLLM):
    def __init__(self, model: str = "AdaptLLM/remote-sensing-Qwen2.5-VL-3B-Instruct"):
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model, torch_dtype="auto", device_map="auto"
        )
        self.processor = AutoProcessor.from_pretrained(model)
    
    def call_vision_llm(self, image: Image.Image, prompt: str) -> str:
        # Convert image to base64
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

        # Preparation for inference
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
        )
        inputs = inputs.to("cuda")

        # Inference: Generation of the output
        generated_ids = model.generate(**inputs, max_new_tokens=128)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return output_text