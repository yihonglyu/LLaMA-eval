import argparse
import time
from transformers import AutoTokenizer
from optimum.onnxruntime import ORTModelForCausalLM

parser = argparse.ArgumentParser()

parser.add_argument('--prompt-length', type=int)
parser.add_argument('--new-token-length', type=int)

args = parser.parse_args()

model_path = "/home/yilyu/LLaMA/llama_7b_onnx-dr"

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = ORTModelForCausalLM.from_pretrained(model_path)

prompt = "happy " * args.prompt_length
inputs = tokenizer(
    prompt,
    return_tensors="pt",
    return_token_type_ids=False,
    )
input_ids = inputs.input_ids
length = input_ids.shape[-1] + args.new_token_length

start_time = time.time()
gen_tokens = model.generate(**inputs, min_length=length, max_length=length)
end_time = time.time()

#print(tokenizer.batch_decode(gen_tokens))

elapsed_time = end_time - start_time
print("Elapsed Time:", elapsed_time, "seconds")