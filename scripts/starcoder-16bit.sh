dataset=stack
nsamples=128
bits=16

# evaluate perplexity of bf16
python santacoder.py bigcode/starcoder $dataset --nsamples $nsamples --eval --wbits $bits
