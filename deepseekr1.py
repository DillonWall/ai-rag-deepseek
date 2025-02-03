from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
import argparse

# Parse args
parser = argparse.ArgumentParser("deepseek r1 7b prompter")
parser.add_argument("prompt", help="The prompt to send to DeepSeek-R1", type=str)
args = parser.parse_args()

# Config
# MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
MAX_NEW_TOKENS = 3000
# 4-bit quantization (requires bitsandbytes)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)


# Setup
device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(device)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto"
)
prompt = args.prompt + "\n<think>\n"

# Eval
inputs = tokenizer(prompt, return_tensors="pt").to(device)
model = model.to(device)
outputs = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))

