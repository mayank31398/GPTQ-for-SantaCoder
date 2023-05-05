dataset=stack
nsamples=128
bits=32

# evaluate perplexity of fp32
python santacoder.py bigcode/starcoderbase $dataset --nsamples $nsamples --eval --wbits $bits
