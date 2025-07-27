
from PIL import Image
import base64
from io import BytesIO

def convert_pil_image2base64(image: Image.Image) -> str:
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str

def clean_output_text(output_text: str | list[str]) -> str:
    if isinstance(output_text, list) and len(output_text) > 0:
        return output_text[0]
    return output_text