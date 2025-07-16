from llms.llm_abc import VisionLLM
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq
import re


class HfVLLM(VisionLLM):
    def __init__(self, model: str = "microsoft/kosmos-2-patch14-224", device: str = "cuda", apply_chat_template: bool = True, max_new_tokens: int = 128):
        self.processor = AutoProcessor.from_pretrained(model)
        self.model = AutoModelForVision2Seq.from_pretrained(model).to(device)
        self.device = device
        self.apply_chat_template = apply_chat_template
        self.max_new_tokens = max_new_tokens

    def call_vision_llm(self, image: Image.Image, prompt: str) -> str:
        image = image.convert("RGB")

        if self.apply_chat_template:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": prompt}
                    ],
                }
            ]

            input_text = self.processor.apply_chat_template(
                messages, add_generation_prompt=True
            )

            inputs = self.processor(
                image, input_text, add_special_tokens=False, return_tensors="pt"
            ).to(self.device)
            generated_ids = self.model.generate(**inputs, max_new_tokens=128)
        else:
            inputs = self.processor(text=[prompt], images=[image], return_tensors="pt")
            inputs = {key: value.to(self.device) for key, value in inputs.items()}

            try:
                generated_ids = self.model.generate(
                    pixel_values=inputs["pixel_values"],
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    image_embeds=None,
                    image_embeds_position_mask=inputs.get(
                        "image_embeds_position_mask", None
                    ),
                    use_cache=True,
                    max_new_tokens=self.max_new_tokens,
                )
            except ValueError as e:
                raise ValueError(
                    f"Error generating text: {e}. "
                    f"Retry with the --apply_vision_template flag."
                )

        generated_text = self.processor.batch_decode(
            generated_ids, skip_special_tokens=True
        )[0]
        # Cleaning the generated text from any tags
        generated_text = re.sub(r"<.*?>.*?<.*?>", "", generated_text)
        generated_text = generated_text.replace("user\n\n", "").replace(
            "assistant\n\n", ""
        )
        # Remove the prompt from the description
        description = re.sub(
            rf"^.*?{re.escape(prompt.lower())}", "", generated_text.lower()
        ).strip()
        return description