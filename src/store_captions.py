from typing import Literal
from dotenv import find_dotenv, load_dotenv
import logging

from torchgeo import datasets

from agents.langgraph_agent import LangGraphMultiAgent
from llms.hf_llm import HfVLLM
from llms.openai_llm import OpenAIVLLM
import os
import torchvision.transforms as transforms
import json
import argparse



logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("labeling.log"), logging.StreamHandler()],
)

dotenv_path = find_dotenv()
if dotenv_path:
    load_dotenv(dotenv_path)
else:
    logging.warning(
        "No .env file found. Make sure to set environment variables manually."
    )


def load_dataset(name: Literal["ucmerced", "resisc45"], split:Literal["train", "val", "test"]= "test", root: str  = "../vision_data") -> datasets.UCMerced | datasets.RESISC45:
    if name == "ucmerced":
        return datasets.UCMerced(root = os.path.join(root, name), download=True, split=split)
    elif name ==  "resisc45":
        return datasets.RESISC45(root=os.path.join(root, name), download=True, split=split)
    else:
        raise ValueError(f"Name must be one of ucmerced or resisc45")

def resize_torch_to_pil(torch_tensor):
    """
    Resize a torch tensor [3, 256, 256] to 224x224 and convert to PIL image
    """
    # Method 1: Using torchvision transforms
    resize_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToPILImage()
    ])
    
    # Ensure tensor values are in [0, 1] range for ToPILImage
    if torch_tensor.max() > 1.0:
        torch_tensor = torch_tensor / 255.0
    
    pil_image = resize_transform(torch_tensor)
    return pil_image

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Label images using a vision LLM and a text classifier."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["ucmerced", "resisc45"],
        default="ucmerced",
        help="Dataset to use for caption generation",
    )
    parser.add_argument(
        "--dataset_split",
        type=str,
        choices=["train", "val", "test"],
        default="test",
        help="Dataset split to use for caption generation",
    )
    args = parser.parse_args()
    
    ds = load_dataset(name = args.dataset, split = args.dataset_split)
    vision_llms  = [OpenAIVLLM(), OpenAIVLLM(), OpenAIVLLM()]
    agent = LangGraphMultiAgent(classes=ds.classes, vision_llms = vision_llms)
    idx2class_map = {v:k for k, v  in ds.class_to_idx.items()}
    captions_data = []
    for item, image_meta in zip(ds, ds.imgs):
        pil_image = resize_torch_to_pil(item["image"])
        state = agent.run(pil_image)
        captions_data.append({
            "image_id": image_meta[0].split(os.path.sep)[-1],
            "image_path": image_meta[0],
            "caption_agent_1": state["descriptions"][0],
            "caption_agent_2": state["descriptions"][1],
            "caption_agent_3": state["descriptions"][2],
            "final_caption": state["final_description"],
            "class_label": idx2class_map[image_meta[1]]
        })
        break

    # Save caption file
    output_file = f"{args.dataset}_{args.dataset_split}_caption.json"
    with open(output_file, 'w') as f:
        json.dump(captions_data, f, indent=2)