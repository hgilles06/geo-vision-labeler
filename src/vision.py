# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
import re

from PIL import Image


def apply_test_time_augmentation(image, test_time_augmentation):
    """
    Applies test time augmentation to the image if specified.
    Test time augmentation strategies includes flipping on x, y, and/or both axes [x, y, xy].
    """
    images = [image]
    for strategy in test_time_augmentation:
        if strategy == "x":
            images.append(image.transpose(Image.FLIP_LEFT_RIGHT))
        elif strategy == "y":
            images.append(image.transpose(Image.FLIP_TOP_BOTTOM))
        elif strategy == "xy":
            images.append(image.transpose(Image.ROTATE_180))
        else:
            raise ValueError(f"Unknown test time augmentation strategy: {strategy}")
    return images


def describe_image(
    image_chunk,
    image_path,
    context,
    prompt,
    include_filename,
    test_time_augmentation,
    processor,
    model,
    device,
):
    """
    Opens an image, processes it through the vision LLM to generate a detailed description,
    and returns a processed text label.
    """
    if test_time_augmentation:
        images = apply_test_time_augmentation(image_chunk, test_time_augmentation)
    else:
        images = [image_chunk]

    descriptions = []
    for image in images:
        image = image_chunk.convert("RGB")
        filename = os.path.basename(image_path).split(".")[0]
        if include_filename:
            input_text = (
                f"{context}. The file name with geo-information is {filename}. "
                + prompt
            )
        else:
            input_text = f"{context}. " + prompt

        inputs = processor(text=[input_text], images=[image], return_tensors="pt")
        inputs = {key: value.to(device) for key, value in inputs.items()}

        generated_ids = model.generate(
            pixel_values=inputs["pixel_values"],
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            image_embeds=None,
            image_embeds_position_mask=inputs.get("image_embeds_position_mask", None),
            use_cache=True,
            max_new_tokens=128,
        )
        generated_text = processor.batch_decode(
            generated_ids, skip_special_tokens=True
        )[0]
        # Cleaning the generated text from any tags
        generated_text = re.sub(r"<.*?>.*?<.*?>", "", generated_text)
        # Remove the prompt from the description if present
        description = generated_text.replace(prompt, "").strip().lower()
        descriptions.append(description)
    # Remove duplicates and join the descriptions
    descriptions = list(set(descriptions))
    return ";".join(descriptions)
