#!/bin/bash

#file=llama_7b_pt.py
file=llama_7b_onnx.py

log="${file}.log"

rm -f $log

# Define the numbers to iterate
prompt_numbers="64 128 256 512 1024"
new_token_lengths="1 129"

# Iterate over the numbers
for num in $prompt_numbers
do
    for length in $new_token_lengths
    do
        echo "prompt_numbers = $num, new_token_length = $length" >> $log
        for i in {1..3}
        do
            echo -n "$i: " >> $log
            python $file --prompt-length $num --new-token-length $length >> $log
        done
    done
done