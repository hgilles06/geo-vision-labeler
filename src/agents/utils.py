from agents.prompts import AGGREGATION_PROMPT_TEMPLATE

from typing import Annotated, List, Any
from typing_extensions import TypedDict
from PIL import Image

from llms.llm_abc import VisionLLM, TextLLM

def merge_dicts(left: dict, right: dict) -> dict:
    """Custom reducer to merge dictionaries"""
    return {**left, **right}

class MultiAgentState(TypedDict):
    image: Image.Image
    classes: List[str]
    prompts: List[str]
    descriptions: Annotated[dict[int, str], merge_dicts]
    final_description: str


def create_vision_node(llm: VisionLLM, prompt_index: int):
    def vision_node(state: MultiAgentState) -> dict[str, Any]:
        image = state["image"]
        classes = state["classes"]
        prompts = state["prompts"]
        
        # Format prompt with classes
        formatted_prompt = prompts[prompt_index].format(
            classes=', '.join(classes) if classes else ''
        )
        
        # Call LLM
        description = llm.call_vision_llm(image, formatted_prompt)
        # Return only the new description - it will be merged with existing descriptions
        return {"descriptions": {prompt_index: description}}
    
    return vision_node


def create_aggregator_node(llm: TextLLM):
    def aggregator_node(state: MultiAgentState) -> dict[str, Any]:
        # Build aggregation prompt
        description_lines = [f"Description {idx + 1}: {desc}" for idx, desc in state["descriptions"].items()]
        formatted_descriptions = "\n".join(description_lines)
        aggregation_prompt = AGGREGATION_PROMPT_TEMPLATE.format(descriptions=formatted_descriptions)
        
        # Call LLM for aggregation
        final_description = llm.call_text_llm(aggregation_prompt)
        
        return {"final_description": final_description}
    
    return aggregator_node