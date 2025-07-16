from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage
from PIL import Image
import base64
from io import BytesIO
from llms.llm_abc import VisionLLM, TextLLM

class OpenAIVLLM(VisionLLM):
    def __init__(self, model: str = "gpt-4o-mini"):
        self.llm = init_chat_model(model, model_provider="openai")
    
    def call_vision_llm(self, image: Image.Image, prompt: str) -> str:
        # Convert image to base64
        image_b64 = self._convert_pil_image2base64(image)
        
        # Create message with image
        message = HumanMessage(
            content=[
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{image_b64}"}
                }
            ]
        )
        
        # Generate response
        response = self.llm.invoke([message])
        return response.content
    
    def _convert_pil_image2base64(self, image: Image.Image) -> str:
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return img_str

class OpenAITextLLM(TextLLM):
    def __init__(self, model: str = "gpt-4o-mini"):
        self.llm = init_chat_model(model, model_provider="openai")
    
    def call_text_llm(self, prompt: str) -> str:
        message = HumanMessage(content=prompt)
        response = self.llm.invoke([message])
        return response.content
