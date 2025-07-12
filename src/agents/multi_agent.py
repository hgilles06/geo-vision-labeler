from openai import AzureOpenAI, OpenAI
from PIL import Image
import base64
from io import BytesIO
from agents.agent_abc import VLMAgent
from agents.prompts import AGGREGATION_PROMPT_TEMPLATE, PROMPTS
from concurrent.futures import ThreadPoolExecutor, as_completed
from types import ModuleType

class MultiAgent(VLMAgent):
    def __init__(self, client: OpenAI | AzureOpenAI | ModuleType, classes: list[str], prompts: list[str] | None = None, model: str = "gpt-4o-mini"):
        self.client = client
        self.classes = classes
        self.prompts = prompts or PROMPTS
        self.model = model
        
    def run(self, image: Image.Image) -> str:
        with ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(self._generate_description, image, prompt.format(classes=', '.join(self.classes) if self.classes else ''))
                for prompt in self.prompts
            ]
            descriptions = [f.result() for f in as_completed(futures)]
        messages = self._build_messages_text(self._build_aggregation_prompt(descriptions))
        return self._generate_response(messages)
    
    def _generate_description(self, image: Image.Image, prompt: str) -> str:
        messages = self._build_messages_vision(image, prompt)
        return self._generate_response(messages)

    def _generate_response(self, messages: list[dict]) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
        )
        return response.choices[0].message.content
    
    def _build_messages_vision(self, image: Image.Image, prompt: str) -> list[dict]:
        return [
                {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url":  {"url": f"data:image/png;base64,{_convert_pil_image2base64(image)}"},
                },
                ],
            }
        ]
    
    def _build_messages_text(self, prompt: str) -> list[dict]:
        return [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                ],
            }
        ]
    
    
    def _build_aggregation_prompt(self, descriptions: list[str]) -> str:
        description_lines = [f"Description {i+1}: {desc}" for i, desc in enumerate(descriptions)]
        formatted = "\n".join(description_lines)
        return AGGREGATION_PROMPT_TEMPLATE.format(descriptions=formatted)


def _convert_pil_image2base64(image: Image.Image) -> str:
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str