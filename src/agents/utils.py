from agents.prompts import AGGREGATION_PROMPT_TEMPLATE
from llms.vision_llm import VisionLLM
from typing import Annotated, List, Any
from typing_extensions import TypedDict
from PIL import Image
import operator

class MultiAgentState(TypedDict):
    image: Image.Image
    classes: List[str]
    prompts: List[str]
    descriptions: Annotated[List[str], operator.add]
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
        
        # Return description as a list - will be appended to descriptions
        return {"descriptions": [description]}
    
    return vision_node


def create_aggregator_node(llm: VisionLLM):
    def aggregator_node(state: MultiAgentState) -> dict[str, Any]:
        descriptions = state["descriptions"]
        
        # Build aggregation prompt
        description_lines = [f"Description {i+1}: {desc}" for i, desc in enumerate(descriptions)]
        formatted_descriptions = "\n".join(description_lines)
        aggregation_prompt = AGGREGATION_PROMPT_TEMPLATE.format(descriptions=formatted_descriptions)
        
        # Call LLM for aggregation
        final_description = llm.call_text_llm(aggregation_prompt)
        
        return {"final_description": final_description}
    
    return aggregator_node