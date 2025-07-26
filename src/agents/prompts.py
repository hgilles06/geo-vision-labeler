# AGGREGATION_PROMPT_TEMPLATE = """
# Generate a single consolidated description from the following individual descriptions:
# {descriptions}
# """

# PROMPTS = [
#     "You are a remote sensing object detection expert. Based on the satellite image, analyze list the visible object types, counts, and spatial arrangement. Based on this, what category (e.g., airport, port, farmland, urban area) might this image belong to?. Classes : {classes}",
#     "You are a remote sensing scene analyst. Describe the overall scene in this satellite image. Focus on land use (e.g., residential, industrial, agricultural), spatial layout (grid-like, irregular), density. Avoid listing objects and focus on high-level interpretation and spatial relationships. Classes : {classes}",
#     "You are a geospatial intelligence analyst. Based on the aerial image and assuming it was taken from a satellite or drone, infer the possible use or function of this area based on given classes. Provide your analysis as a contextual description in 2-3 sentences. Classes : {classes}"
# ]

AGGREGATION_PROMPT_TEMPLATE = """
You are a Senior Intelligence Editor tasked with producing a final, definitive caption for scene classification. Synthesize the following three analyst reports into a single, coherent description.

{descriptions}

Follow these rules precisely to generate the final caption:
1.  Start with the high-level scene assessment from the Scene Analyst to set the context.
2.  Next, integrate the key findings from the Object Detection Expert's inventory, presenting them as supporting evidence. If there are conflicts in object counts, the Object Detection Expert's numbers are authoritative.
3.  Conclude with the final, inferred function from the Geospatial Intelligence Analyst, ensuring it is presented as the definitive scene classification.
4.  The final output should be a single, flowing paragraph that logically connects the scene description, object evidence, and final classification. Do not use headings or bullet points.
"""

PROMPTS = [ "You are a remote sensing scene analyst. Based on the satellite image, describe the overall scene focusing on high-level land use, spatial layout, and density, and based on this interpretation, suggest the most likely scene category. Classes: {classes}",
             "You are a remote sensing object detection expert. Based on the satellite image and the class list '{classes}', systematically identify and list the visible object types and their counts, and based on this inventory, determine the primary category of the location.",
             "You are a geospatial intelligence analyst. Based on the satellite image, infer the primary function or activity of this area, providing your analysis in 2-3 sentences and explicitly citing the visual evidence from the image that supports your conclusion."
]