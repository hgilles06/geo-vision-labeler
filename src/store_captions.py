from typing import Literal
from dotenv import find_dotenv, load_dotenv
import logging
import json
import os
from pathlib import Path
import torch
from torchgeo import datasets
from agents.langgraph_agent import LangGraphMultiAgent
import torchvision.transforms as transforms
import argparse
from llms.llama_llm import LlamaVLLM
from llms.qwen_llm import QwenVLLM
from llms.llava_llm import LlavaVLLM
from llms.mistral_llm import MistralTextLLM

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


def load_dataset(name: Literal["ucmerced", "resisc45"], split:Literal["train", "val", "test"] = "test", root: str  = "../vision_data") -> datasets.UCMerced | datasets.RESISC45:
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


def append_to_json(data, file_path):
    """
    Append a single dictionary to a JSON file, creating the file if it doesn't exist.
    """
    file_path = Path(file_path)
    if file_path.exists():
        with open(file_path, 'r+') as f:
            try:
                existing_data = json.load(f)
                if not isinstance(existing_data, list):
                    existing_data = [existing_data]
            except json.JSONDecodeError:
                existing_data = []
            existing_data.append(data)
            f.seek(0)
            json.dump(existing_data, f, indent=2)
    else:
        with open(file_path, 'w') as f:
            json.dump([data], f, indent=2)


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

    parser.add_argument(
        "--output_dir",
        type=str,
        default="../result",
        help="Directory to save the output JSON file",
    )
    args = parser.parse_args()

    ds = load_dataset(name=args.dataset, split=args.dataset_split)
    os.makedirs(args.output_dir, exist_ok=True)
    output_file = os.path.join(args.output_dir, f"{args.dataset}_{args.dataset_split}_caption.json")
    idx2class_map = {v: k for k, v in ds.class_to_idx.items()}
    
    # Initialize vision LLMs and agent once
    logging.info("Initializing vision LLMs and agent...")
    try:
        vision_llms = [LlamaVLLM(), LlamaVLLM(), QwenVLLM()]
        agent = LangGraphMultiAgent(classes=ds.classes, vision_llms=vision_llm, text_llm = MistralTextLLM())
    except Exception as e:
        logging.error(f"Failed to initialize LLMs or agent: {str(e)}")
        exit(1)
    
    # Process images one by one
    total_images = len(ds)
    for idx, (item, image_meta) in enumerate(zip(ds, ds.imgs)):
        image_id = image_meta[0].split(os.path.sep)[-1]
        logging.info(f"Processing image {idx + 1}/{total_images}: {image_id}")
        try:
            pil_image = resize_torch_to_pil(item["image"])
            state = agent.run(pil_image)
            
            # Prepare caption data
            caption_data = {
                "image_id": image_id,
                "image_path": image_meta[0],
                "caption_agent_1": state["descriptions"][0],
                "caption_agent_2": state["descriptions"][1],
                "caption_agent_3": state["descriptions"][2],
                "final_caption": state["final_description"],
                "class_label": idx2class_map[image_meta[1]]
            }
            
            # Append to JSON file immediately
            append_to_json(caption_data, output_file)
            logging.info(f"Saved captions for image {image_id} to {output_file}")
            
            # Clean up memory after each image
            torch.cuda.empty_cache()
            
        except Exception as e:
            logging.error(f"Error processing image {image_id}: {str(e)}")
            continue  # Skip to next image
    
    # Clean up
    del agent
    del vision_llms
    torch.cuda.empty_cache()
    logging.info(f"Completed processing. Captions saved to {output_file}")