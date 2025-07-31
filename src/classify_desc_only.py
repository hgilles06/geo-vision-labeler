# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
import sys
import argparse
import logging

import torch
from tqdm import tqdm
import json
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from huggingface_hub import login
from openai import OpenAI, AzureOpenAI
from PIL import Image
from clip import CLIPClassifier
import torch
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from dotenv import load_dotenv, find_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables from .env file if it exists
dotenv_path = find_dotenv()
if dotenv_path:
    load_dotenv(dotenv_path)
else:
    logging.warning(
        "No .env file found. Make sure to set environment variables manually."
    )

login(token=os.environ.get("HF_TOKEN", ""))

# Constants
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"

def init_openai_classifier_client(variant, use_azure_entra_id):
    if variant.lower() == "azure":
        if use_azure_entra_id:
            token_provider = get_bearer_token_provider(
                DefaultAzureCredential(),
                "https://cognitiveservices.azure.com/.default"
            )
            client = AzureOpenAI(
                azure_endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT", ""),
                azure_ad_token_provider=token_provider,
                api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2023-03-15-preview"),
            )
        else:
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

def classify_with_openai(desc, include_filename, filename, classes, client, model_id):
    if not desc:
        return classes[0] if classes else "unknown"
    
    # Construct the prompt
    text = desc if not include_filename else f"Filename: {filename}\nDescription: {desc}"
    prompt = f"""Given the following description, classify it into one of these categories: {', '.join(classes)}.
    
    {text}
    
    The category is:"""
    
    if model_id.startswith("gpt-"):
        # Standard OpenAI model
        response = client.chat.completions.create(
            model=model_id,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=50
        )
        prediction = response.choices[0].message.content.strip()
    else:
        # Azure OpenAI deployment
        response = client.chat.completions.create(
            model=model_id,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=50
        )
        prediction = response.choices[0].message.content.strip()
        
    # Extract just the class name from the response
    for cls in classes:
        if cls.lower() in prediction.lower():
            return cls
            
    # If no exact match, return the first word as a fallback
    return prediction.split()[0]

def classify_with_huggingface(desc, include_filename, filename, classes, gen_pipeline, tokenizer, max_length):
    if not desc:
        return classes[0] if classes else "unknown"
    
    # Construct the prompt
    text = desc if not include_filename else f"Filename: {filename}\nDescription: {desc}"
    classes_str = ", ".join(classes)
    prompt = f"""Instruction: Classify the following description into one of these categories: {classes_str}.
    
    Description: {text}
    
    Classification:"""
    
    try:
        inputs = tokenizer(prompt, return_tensors="pt").to(gen_pipeline.device)
        outputs = gen_pipeline(
            prompt,
            max_new_tokens=max_length,
            do_sample=False,
            temperature=0.1,
            top_p=0.9,
            num_return_sequences=1
        )
        
        prediction = outputs[0]["generated_text"][len(prompt):].strip()
        
        # Extract just the class name from the response
        for cls in classes:
            if cls.lower() in prediction.lower():
                return cls
                
        # If no exact match, return the first word as a fallback
        return prediction.split()[0]
    
    except Exception as e:
        logging.error(f"Error with Hugging Face classification: {str(e)}")
        return classes[0] if classes else "unknown"
        
def main():
    parser = argparse.ArgumentParser(description="Classify descriptions into predefined classes")
    
    # Required arguments
    parser.add_argument("--descriptions_path", type=str, required=True, 
                        help="Path to the file containing descriptions to classify")
    parser.add_argument("--classes_file", type=str, required=True,
                        help="Path to the file containing list of classes (one per line)")
    parser.add_argument("--output_file", type=str, required=True,
                        help="Path to save the classification results")
    
    # Optional arguments
    parser.add_argument("--classifier", type=str, default="microsoft/Phi-3-mini-4k-instruct",
                        help="Name of the classifier model.")
    parser.add_argument("--classifier_type", type=str, choices=["openai", "huggingface"],
                        default="huggingface", help="Classifier type: openai and huggingface.")
    parser.add_argument("--openai_variant", type=str, choices=["azure", "openai"],
                        default="azure", help="For OpenAI classifier: choose Azure or OpenAI API.")
    parser.add_argument("--use_azure_entra_id", action="store_true",
                        help="Use Azure Entra ID for authentication with Azure OpenAI.")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu",
                        help="Device to run the Hugging Face models on.")
    parser.add_argument("--include_filename", action="store_true",
                        help="Include filename in the classification prompt")
    parser.add_argument("--context", type=str, default=None,
                        help="Optional context to add to CLIP classification")
    
    args = parser.parse_args()
    
    # Load classes
    try:
        with open(args.classes_file, "r") as f:
            classes_flat = [line.strip() for line in f.readlines()]
        logging.info(f"Loaded {len(classes_flat)} classes from {args.classes_file}")
    except Exception as e:
        logging.error(f"Failed to load classes file: {str(e)}")
        return
    
    # Load descriptions
    try:
        with open(args.descriptions_path, "r") as f:
            descriptions = json.load(f)
        logging.info(f"Loaded {len(descriptions)} descriptions from {args.descriptions_path}")
    except Exception as e:
        logging.error(f"Failed to load descriptions file: {str(e)}")
        return
    
    # Setup classifier
    clip_classifier = CLIPClassifier(
        model_name=CLIP_MODEL_NAME,
        device=args.device,
    )

    # Setup classifier pipelines
    client = None
    gen_pipeline = None
    tokenizer = None
    
    if args.classifier_type == "openai":
        client = init_openai_classifier_client(args.openai_variant, args.use_azure_entra_id)
    elif args.classifier_type == "huggingface":
        tokenizer = AutoTokenizer.from_pretrained(args.classifier)
        llm_model = AutoModelForCausalLM.from_pretrained(args.classifier).to(args.device)
        gen_pipeline = pipeline(
            "text-generation",
            model=llm_model,
            tokenizer=tokenizer,
            device=args.device,
        )
    
    # Process descriptions
    for i, description_dict in tqdm(enumerate(descriptions), desc="Classifying descriptions", total=len(descriptions)):
        filename, desc = os.path.basename(description_dict["image_path"]), description_dict["final_caption"].strip()
        coords = {'row': 0, 'col': 0}  # Default coords since we're only dealing with descriptions
        
        if args.classifier_type == "openai":
            label = classify_with_openai(
                desc,
                args.include_filename,
                filename,
                classes_flat,
                client,
                args.classifier,
            )
        elif args.classifier_type == "huggingface":
            label = classify_with_huggingface(
                desc,
                args.include_filename,
                filename,
                classes_flat,
                gen_pipeline,
                tokenizer,
                28,  # Max token length for output
            )
        else:  
            raise ValueError(f"Unsupported classifier type: {args.classifier_type}. Supported types are 'openai' and 'huggingface'.")
        
        # Apply fallback if label is not in the current hierarchy
        label = fallback_label(
            label, classes_flat, None, filename, coords, clip_classifier, args
        )
        print(f"Classified {filename} as {label}")

        descriptions[i]["classification"] = label

    # Save results
    with open(args.output_file, "w") as f:
        json.dump(descriptions, f, indent=4)
    
    logging.info(f"Classification completed. Results saved to {args.output_file}")

def fallback_label(label, classes, chunk, filename, coords, clip_classifier, args):
    """
    Fallback mechanism to handle cases where the label is not found in the current hierarchy.
    """
    if label not in classes:
        # Log a warning and try to find the closest match
        logging.warning(
            f"Label '{label}' not found in classes for file {filename}_r{coords['row']}_c{coords['col']}. "
            "Trying to find the closest match."
        )
        matched_classes = sorted(
            [
                (class_name, label.find(class_name))
                for class_name in classes
                if class_name in label
            ]
            + [
                (class_name, class_name.find(label))
                for class_name in classes
                if label in class_name
            ],
            key=lambda x: x[1],
        )
        if matched_classes:
            # Use the closest match
            label = matched_classes[0][0]
        else:
            # If no match found, use the first class as a fallback
            label = classes[0] if classes else "unknown"
            logging.warning(
                f"No close match found for label '{label}' in file {filename}_r{coords['row']}_c{coords['col']}. "
                "Using first class as fallback."
            )
    return label

if __name__ == "__main__":
    main()