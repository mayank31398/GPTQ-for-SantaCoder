dataset=stack
nsamples=128
bits=16

# evaluate perplexity of bf16
python santacoder.py bigcode/gpt_bigcode-santacoder $dataset --nsamples $nsamples --eval --wbits $bits
