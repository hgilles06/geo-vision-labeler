# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import argparse
import csv
import glob
import logging
import os

import numpy as np
import torch
import tqdm
from PIL import Image
from dotenv import find_dotenv, load_dotenv
from huggingface_hub import login
from transformers import (
    AutoModelForCausalLM,
    AutoModelForVision2Seq,
    AutoProcessor,
    AutoTokenizer,
    pipeline,
)
import openai
from openai import AzureOpenAI

from vision import describe_image
from classifier import classify_with_openai, classify_with_huggingface

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

login(token=os.getenv("HF_TOKEN"))


def split_image(image_path, split_height_by, split_width_by):
    image = Image.open(image_path)
    image = image.convert("RGB")
    width, height = image.size
    chunk_width = width // split_width_by
    chunk_height = height // split_height_by
    chunks = []
    coords_dict_list = []
    for i in range(split_height_by):
        for j in range(split_width_by):
            left = j * chunk_width
            lower = i * chunk_height
            right = left + chunk_width
            upper = lower + chunk_height
            coords = (left, lower, right, upper)
            chunk = image.crop(coords)
            # Return the chunk and its metadata (row, col, pixel coordinates)
            chunks.append(chunk)
            coords_dict_list.append(
                {
                    "row": i,
                    "col": j,
                    "left": left,
                    "upper": upper,
                    "right": right,
                    "lower": lower,
                }
            )
    return chunks, coords_dict_list


def init_openai_client(variant):
    if variant.lower() == "azure":
        client = AzureOpenAI(
            api_version=os.environ.get(
                "AZURE_OPENAI_API_VERSION", "2023-03-15-preview"
            ),
            azure_endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT", ""),
            api_key=os.environ.get("AZURE_OPENAI_API_KEY", ""),
        )
    else:
        openai.api_key = os.environ.get("OPENAI_API_KEY", "")
        client = openai
    return client


def main():
    parser = argparse.ArgumentParser(
        description="Label images using a vision LLM and a text classifier."
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Directory containing input images.",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="data/labels.csv",
        help="CSV file to write the results.",
    )
    parser.add_argument(
        "--classes_file",
        type=str,
        default="data/classes.txt",
        help="File containing the classes for classification.",
    )
    parser.add_argument(
        "--vision_model",
        type=str,
        default="microsoft/kosmos-2-patch14-224",
        help="Vision model to use for image description.",
    )
    parser.add_argument(
        "--classifier",
        type=str,
        default="meta-llama/Llama-3.1-8B-Instruct",
        help="Name of the classifier model.",
    )
    parser.add_argument(
        "--classifier_type",
        type=str,
        choices=["openai", "huggingface"],
        default="huggingface",
        help="Classifier type: OpenAI or Hugging Face.",
    )
    parser.add_argument(
        "--openai_variant",
        type=str,
        choices=["azure", "openai"],
        default="azure",
        help="For OpenAI classifier: choose Azure or OpenAI API.",
    )
    parser.add_argument(
        "--split_height_by",
        type=int,
        default=1,
        help="Number of vertical splits per image.",
    )
    parser.add_argument(
        "--split_width_by",
        type=int,
        default=1,
        help="Number of horizontal splits per image.",
    )
    parser.add_argument(
        "--context",
        type=str,
        default="This is a satellite image. ",
        help="Meta prompt to guide the vision model.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Detailed long description of the image: ",
        help="Prompt to guide the vision model.",
    )
    parser.add_argument(
        "--include_filename",
        action="store_true",
        help="Include the filename in the prompt for the vision model.",
    )
    parser.add_argument(
        "--test_time_augmentation",
        type=list,
        default="",
        help="Test time augmentation strategies for rotation with x, y, and/or both axes [x, y, xy].",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0" if torch.cuda.is_available() else "cpu",
        help="Device to run the models on.",
    )
    args = parser.parse_args()

    if not os.path.exists(args.classes_file):
        raise FileNotFoundError(f"Classes file {args.classes_file} not found.")
    with open(args.classes_file, "r") as f:
        classes = [line.strip() for line in f.readlines() if line.strip()]
    if not classes:
        raise ValueError("No classes found in the classes file.")
    logging.info(f"Loaded {len(classes)} classes from {args.classes_file}.")

    processor = AutoProcessor.from_pretrained(args.vision_model)
    model = AutoModelForVision2Seq.from_pretrained(args.vision_model).to(args.device)

    filepaths = glob.glob(os.path.join(args.input_dir, "*"))
    if not filepaths:
        logging.info("No images found in the specified input directory.")
        return

    np.random.seed(42)
    filepaths = np.random.permutation(filepaths)

    if args.classifier_type == "openai":
        client = init_openai_client(args.openai_variant)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.classifier)
        llm_model = AutoModelForCausalLM.from_pretrained(args.classifier).to(
            args.device
        )
        gen_pipeline = pipeline(
            "text-generation",
            model=llm_model,
            tokenizer=tokenizer,
            device=0 if "cuda" in args.device else -1,
        )

    results = []
    for filepath in tqdm.tqdm(filepaths):
        chunks, coords_dict_list = split_image(
            filepath, args.split_height_by, args.split_width_by
        )
        for i, chunk in enumerate(chunks):
            desc = describe_image(
                image_chunk=chunk,
                image_path=filepath,
                context=args.context,
                prompt=args.prompt,
                include_filename=args.include_filename,
                test_time_augmentation=args.test_time_augmentation,
                processor=processor,
                model=model,
                device=args.device,
            )
            filename = os.path.basename(filepath)
            # Prepare metadata columns
            coords_dict = coords_dict_list[i]
            row, col, left, upper, right, lower = (
                coords_dict["row"],
                coords_dict["col"],
                coords_dict["left"],
                coords_dict["upper"],
                coords_dict["right"],
                coords_dict["lower"],
            )
            meta_data = [filename, row, col, left, upper, right, lower, desc]

            if args.classifier_type == "openai":
                label = classify_with_openai(
                    desc, f"{filename}_r{row}_c{col}", classes, client, "gpt-4"
                )
            else:
                label = classify_with_huggingface(
                    desc,
                    f"{filename}_r{row}_c{col}",
                    classes,
                    gen_pipeline,
                    tokenizer,
                    10,
                )
            meta_data.append(label)
            results.append(meta_data)

    with open(args.output_file, mode="w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(
            [
                "filename",
                "row",
                "col",
                "left",
                "upper",
                "right",
                "lower",
                "description",
                "classification",
            ]
        )
        writer.writerows(results)

    logging.info("Done.")


if __name__ == "__main__":
    main()
