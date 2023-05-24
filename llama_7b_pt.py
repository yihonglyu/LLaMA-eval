import gc
import time
import torch
from transformers import LlamaTokenizer, LlamaForCausalLM

n_batch = 10

tokenizer_path = "decapoda-research/llama-7b-hf"
model_path = "decapoda-research/llama-7b-hf"

print(f"Excuting LLaMA models under {model_path}")

prompt_lengths = [ 64, 128, 256, 512, 1024 ]
new_token_lengths= [ 1, 129 ]

tokenizer = LlamaTokenizer.from_pretrained(tokenizer_path)
model = LlamaForCausalLM.from_pretrained(model_path)

for prompt_length in prompt_lengths:
    for new_token_length in new_token_lengths:

        print(f"prompt_numbers = {prompt_length}, new_token_length = {new_token_length}:")

        prompt = "happy " * prompt_length

        ## PyTorch
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
        )
        gen_tokens_length = inputs.input_ids.shape[-1] + new_token_length

        # warmup
        _ = model.generate(
            **inputs,
            min_length=gen_tokens_length,
            max_length=gen_tokens_length
        )

        gc.collect()
        gc.disable()
        #prev_gen_tokens = None
        start = time.time()
        for i in range(n_batch):
            gen_tokens = model.generate(
                **inputs,
                min_length=gen_tokens_length,
                max_length=gen_tokens_length
            )

            #if prev_gen_tokens is not None:
            #    assert torch.equal(prev_gen_tokens, gen_tokens)
            #prev_gen_tokens = gen_tokens
        end = time.time()
        gc.enable()
        print(f"PT: {(end - start) / n_batch:.3f} s")