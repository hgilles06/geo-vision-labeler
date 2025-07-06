from src.agents.agent_abc import VLMAgent
from openai import AzureOpenAI, Client

class MultiAgent(VLMAgent):
    def __init__(self, client: Client | AzureOpenAI, prompts: list[str], agg_prompt: str):
        self.client = client
        self.prompts = prompts or ["Provide a brief description for the image"] * 3
        self.agg_prompt = agg_prompt

    def run(self, images: list[str]) -> str:
        return ""