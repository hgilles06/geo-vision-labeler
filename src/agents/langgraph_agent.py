
from typing import Annotated, List
from langgraph.graph.state import CompiledStateGraph
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from PIL import Image
from agents.agent_abc import VLMAgent
from agents.prompts import PROMPTS
from agents.utils import create_aggregator_node, create_vision_node, MultiAgentState
from llms.vision_llm import VisionLLM


class LangGraphMultiAgent(VLMAgent):
    def __init__(self, classes: List[str], prompts: List[str] | None = None, model: str = "gpt-4o-mini"):
        self.classes = classes
        self.prompts = prompts or PROMPTS
        self.llm = VisionLLM(model=model)
        
        # Build the workflow graph
        self.graph = self._build_graph()
    
    def _build_graph(self) -> CompiledStateGraph:
        """Build the LangGraph workflow with dynamic number of agents"""
        graph_builder = StateGraph(MultiAgentState)
        
        # Create nodes for each prompt
        for i in range(len(self.prompts)):
            node_name = f"vision_agent_{i+1}"
            vision_node = create_vision_node(self.llm, i)
            graph_builder.add_node(node_name, vision_node)
        
        # Create aggregator node
        aggregator_node = create_aggregator_node(self.llm)
        graph_builder.add_node("aggregator", aggregator_node)
        
        # Add edges for parallel execution
        for i in range(len(self.prompts)):
            node_name = f"vision_agent_{i+1}"
            graph_builder.add_edge(START, node_name)
            graph_builder.add_edge(node_name, "aggregator")
        
        graph_builder.add_edge("aggregator", END)
        
        return graph_builder.compile()
    
    def run(self, image: Image.Image) -> str:
        initial_state = {
            "image": image,
            "classes": self.classes,
            "prompts": self.prompts,
            "descriptions": [],
            "final_description": ""
        }
        
        final_state = self.graph.invoke(initial_state)
        return final_state["final_description"]