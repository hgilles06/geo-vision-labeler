from transformers import AutoModelForCausalLM, AutoTokenizer
from llms.llm_abc import TextLLM
from llms.utils import clean_output_text
import torch

class MistralTextLLM(TextLLM):
    def __init__(self, model: str = "mistralai/Mixtral-8x7B-Instruct-v0.1"):
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                model,
                load_in_8bit=True,  # Use 8-bit quantization for fp32 offloading
                llm_int8_enable_fp32_cpu_offload=True,  # Keep specific modules in 32-bit
                device_map="auto",  # Automatic device placement
                offload_folder="offload"  # Folder for disk offloading
            )
            self.tokenizer = AutoTokenizer.from_pretrained(model)
        except Exception as e:
            raise RuntimeError(f"Failed to load model or tokenizer: {e}")

    def call_text_llm(self, prompt: str, max_new_tokens: int = 512) -> str:
        try:
            # Format the prompt using the chat template
            inputs = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}], return_tensors="pt"
            )
            # Move inputs to the primary device (GPU if available)
            if torch.cuda.is_available():
                inputs = inputs.to("cuda:0")
            
            # Generate output
            outputs = self.model.generate(inputs, max_new_tokens=max_new_tokens)
            
            # Decode and clean the output
            decoded_output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return clean_output_text(decoded_output)
        except Exception as e:
            raise RuntimeError(f"Error during text generation: {e}")