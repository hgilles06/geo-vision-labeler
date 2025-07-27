import json
import os
import sys

def convert_json(input_json_path, output_dir):
    # Ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Read the input JSON file
    with open(input_json_path, 'r') as f:
        input_data = json.load(f)
    
    # Prepare output data
    output_data = {}
    index = 0
    
    # Process each entry in the input JSON
    for entry in input_data:
        # First instruction format
        output_data[str(index)] = {
            "img_id": entry["image_id"],
            "instruction": "Given the visual evidence in the image and the following textual context, what is the correct scene category?",
            "input": entry["final_caption"],
            "output": entry["class_label"],
            "image_path": entry["image_path"]
        }
        index += 1
        
        # Second instruction format
        output_data[str(index)] = {
            "img_id": entry["image_id"],
            "instruction": "Based on the provided image and its textual context, determine the most accurate classification label.",
            "input": entry["final_caption"],
            "output": entry["class_label"],
            "image_path": entry["image_path"]
        }
        index += 1
    
    # Generate output filename
    base_name = os.path.splitext(os.path.basename(input_json_path))[0]
    output_json_path = os.path.join(output_dir, f"{base_name}_FT.json")
    
    # Save output JSON
    with open(output_json_path, 'w') as f:
        json.dump(output_data, f, indent=4)
    
    return output_json_path

# Example usage
if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python json_converter.py <input_json_path> <output_directory>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_dir = sys.argv[2]
    output_file = convert_json(input_file, output_dir)
    print(f"Output JSON saved as: {output_file}")