AGGREGATION_PROMPT_TEMPLATE = """
Generate a single consolidated description from the following individual descriptions:
{descriptions}
"""

PROMPTS = [
    "You are a remote sensing object detection expert. Based on the satellite image, analyze list the visible object types, counts, and spatial arrangement. Based on this, what category (e.g., airport, port, farmland, urban area) might this image belong to?. Classes : {classes}",
    "You are a remote sensing scene analyst. Describe the overall scene in this satellite image. Focus on land use (e.g., residential, industrial, agricultural), spatial layout (grid-like, irregular), density, vegetation coverage. Avoid listing objects and focus on high-level interpretation and spatial relationships. Classes : {classes}",
    "You are a geospatial intelligence analyst. Based on the aerial image and assuming it was taken from a satellite or drone over a semi-urban area in, infer the possible use or function of this area. Consider geographic elements (roads, water bodies), seasonal effects (e.g., crop growth), human activity (construction, traffic), and infrastructure roles (e.g., logistics, residential, recreational). Provide your analysis as a functional and contextual description in 2-3 sentences. Classes : {classes}"
]