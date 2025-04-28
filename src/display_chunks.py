# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import argparse
import os
import math

import pandas as pd
from PIL import Image, ImageDraw, ImageFont


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Display image chunks with metadata and predicted labels from a CSV labels file."
    )
    parser.add_argument(
        "--labels_file", type=str, required=True, help="Path to the CSV labels file."
    )
    parser.add_argument(
        "--images_dir",
        type=str,
        required=True,
        help="Directory containing the original images.",
    )
    parser.add_argument(
        "--num_files", type=int, default=None, help="Number of chunks to display."
    )
    parser.add_argument("--random", action="store_true", help="Select chunks randomly.")
    parser.add_argument(
        "--filename",
        type=str,
        default=None,
        help="Display only chunks for the specified filename.",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="data/outputs/collage.png",
        help="Output file for the collage image.",
    )
    return parser.parse_args()


def load_chunk(row, images_dir):
    filename = row["filename"]
    image_path = os.path.join(images_dir, filename)
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file {image_path} not found.")
    image = Image.open(image_path).convert("RGB")
    left = int(row["left"])
    lower = int(row["lower"])
    right = int(row["right"])
    upper = int(row["upper"])
    chunk = image.crop((left, lower, right, upper))
    return chunk


def annotate_chunk(chunk, row):
    chunk = chunk.resize((256, 256), Image.NEAREST)
    draw = ImageDraw.Draw(chunk)
    text = (
        f"{row['filename']}\n"
        f"({row['row']}, {row['col']})\n"
        f"Label: {row['classification']}"
    )
    font = ImageFont.load_default()
    # draw proportionally to the chunk size
    text_x = (chunk.width - chunk.size[0]) // 2
    text_y = (chunk.height - chunk.size[1]) // 2

    draw.text((text_x, text_y), text, fill="yellow", font=font)
    return chunk


def create_collage(chunks, output_file):
    n = len(chunks)
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)
    w, h = chunks[0].size
    line_thickness = 5  # Thickness of the red lines
    collage_width = cols * w + (cols - 1) * line_thickness
    collage_height = rows * h + (rows - 1) * line_thickness
    collage = Image.new("RGB", (collage_width, collage_height), color="black")

    for idx, chunk in enumerate(chunks):
        col = idx % cols
        row = idx // cols
        x = col * (w + line_thickness)
        y = row * (h + line_thickness)
        collage.paste(chunk, (x, y))

    # Draw red lines
    draw = ImageDraw.Draw(collage)
    for col in range(1, cols):
        x = col * w + (col - 1) * line_thickness
        draw.rectangle([x, 0, x + line_thickness - 1, collage_height], fill="red")
    for row in range(1, rows):
        y = row * h + (row - 1) * line_thickness
        draw.rectangle([0, y, collage_width, y + line_thickness - 1], fill="red")

    collage.save(output_file)
    print(f"Collage saved to {output_file}")


def main():
    args = parse_arguments()
    df = pd.read_csv(args.labels_file)
    if args.filename:
        df = df[df["filename"] == args.filename]
    if df.empty:
        print("No entries found for the given criteria.")
        return

    n_selected = num_files if args.num_files is not None else len(df)
    if args.random:
        selected = df.sample(n=min(n_selected, len(df)))
    else:
        selected = df.head(n_selected)
    selected = selected.reset_index(drop=True)

    chunks = []
    for _, row in selected.iterrows():
        chunk = load_chunk(row, args.images_dir)
        chunk = annotate_chunk(chunk, row)
        chunks.append((chunk, row))
    # reorder chunks to match the original order
    chunks = sorted(chunks, key=lambda x: (x[1]["row"], x[1]["col"]))
    chunks = [chunk for chunk, _ in chunks]
    # Create a collage of the chunks
    create_collage(chunks, args.output_file)


if __name__ == "__main__":
    main()
