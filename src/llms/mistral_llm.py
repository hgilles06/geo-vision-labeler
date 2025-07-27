from transformers import AutoModelForCausalLM, AutoTokenizer

from llms.llm_abc import TextLLM
from llms.utils import clean_output_text

class MistralTextLLM(TextLLM):
    def __init__(self, model: str = "mistralai/Mixtral-8x7B-Instruct-v0.1"):
        self.model = AutoModelForCausalLM.from_pretrained(model, load_in_4bit=True, device_map="auto")
        self.tokenizer = AutoTokenizer.from_pretrained(model)

    def call_text_llm(self, prompt: str) -> str:
        inputs = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], return_tensors="pt").to(self.model.device)   
        outputs = model.generate(inputs, max_new_tokens=512)
        return clean_output_text(tokenizer.decode(outputs[0], skip_special_tokens=True))
