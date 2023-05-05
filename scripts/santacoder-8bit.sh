dataset=stack
nsamples=128
opt_level=3
groupsize=-1
bits=8

mkdir -p models/$bits-bit

# remove --eval if you dont want to evaluate perplexity of the model
python santacoder.py bigcode/gpt_bigcode-santacoder $dataset --nsamples $nsamples --eval --wbits $bits --act-order --groupsize $groupsize --optimization-level $opt_level --true-sequential --save models/$bits-bit/model.pt
