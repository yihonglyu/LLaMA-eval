import gc
import time
import torch
from transformers import LlamaTokenizer, LlamaForCausalLM
from optimum.onnxruntime import ORTModelForCausalLM

tokenizer_path = "decapoda-research/llama-7b-hf"
pt_model_path = "decapoda-research/llama-7b-hf"
ort_model_path = "/home/yilyu/LLaMA/llama-7b-onnx-opt-nomerged"

print(f"Excuting LLaMA PyTorch models under {pt_model_path}")
print(f"Excuting LLaMA ONNX models under {ort_model_path}")

n_batch = 2

prompt_lengths = [ 64, 128, 256, 512, 1024 ]
new_token_lengths= [ 1, 129 ]

pt_tokenizer = LlamaTokenizer.from_pretrained(tokenizer_path)
pt_model = LlamaForCausalLM.from_pretrained(pt_model_path)
ort_tokenizer = LlamaTokenizer.from_pretrained(tokenizer_path)
ort_model = ORTModelForCausalLM.from_pretrained(
    ort_model_path,
    use_io_binding = True,
)

for prompt_length in prompt_lengths:
    for new_token_length in new_token_lengths:

        print(
            f"prompt_numbers = {prompt_length}, new_token_length = {new_token_length}: ",
            end=""
        )

        prompt = "happy " * prompt_length

        gc.collect()
        ## PyTorch
        pt_inputs = pt_tokenizer(
            prompt,
            return_tensors="pt",
        )
        pt_gen_tokens_length = pt_inputs.input_ids.shape[-1] + new_token_length

        pt_prev_decoded_list = None
        for i in range(n_batch):
            gen_tokens = pt_model.generate(
                **pt_inputs,
                min_length=pt_gen_tokens_length,
                max_length=pt_gen_tokens_length
            )
            decoded_list = pt_tokenizer.batch_decode(gen_tokens)
            #print(f"pt decoded_list: {decoded_list}")

            if pt_prev_decoded_list is not None:
                assert pt_prev_decoded_list == decoded_list
            pt_prev_decoded_list = decoded_list


        gc.collect()
        ## ONNX Runtime
        ort_inputs = ort_tokenizer(
            prompt,
            return_tensors="pt",
            return_token_type_ids=False,
        )
        ort_gen_tokens_length = ort_inputs.input_ids.shape[-1] + new_token_length

        ort_prev_decoded_list = None
        for i in range(n_batch):
            gen_tokens = ort_model.generate(
                **ort_inputs,
                min_length=ort_gen_tokens_length,
                max_length=ort_gen_tokens_length
            )
            decoded_list = ort_tokenizer.batch_decode(gen_tokens)
            #print(f"ort decoded_list: {decoded_list}")

            if ort_prev_decoded_list is not None:
                assert ort_prev_decoded_list == decoded_list
            ort_prev_decoded_list = decoded_list

        if pt_prev_decoded_list == ort_prev_decoded_list:
            print("pass")
        else:
            print("fail")