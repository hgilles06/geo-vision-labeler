from openai import AzureOpenAI, OpenAI
from PIL import Image
import base64
from io import BytesIO
from agent_abc import VLMAgent
from concurrent.futures import ThreadPoolExecutor, as_completed
from types import ModuleType


AGGREGATION_PROMPT_TEMPLATE = """
Generate a single consolidated description from the following individual descriptions:
{descriptions}
"""

PROMPTS = [
    "You are a remote sensing object detection expert. Based on the satellite image, analyze list the visible object types, counts, and spatial arrangement. Based on this, what category (e.g., airport, port, farmland, urban area) might this image belong to?. Classes : {classes}",
    "You are a remote sensing scene analyst. Describe the overall scene in this satellite image. Focus on land use (e.g., residential, industrial, agricultural), spatial layout (grid-like, irregular), density, vegetation coverage. Avoid listing objects and focus on high-level interpretation and spatial relationships. Classes : {classes}"
    "You are a geospatial intelligence analyst. Based on the aerial image and assuming it was taken from a satellite or drone over a semi-urban area in, infer the possible use or function of this area. Consider geographic elements (roads, water bodies), seasonal effects (e.g., crop growth), human activity (construction, traffic), and infrastructure roles (e.g., logistics, residential, recreational). Provide your analysis as a functional and contextual description in 2-3 sentences. Classes : {classes}"
]

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
            model="gpt-4o-mini",
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
                    "image_url":  {"url": f"data:image/jpeg;base64,{_convert_pil_image2base64(image)}"},
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
    image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str