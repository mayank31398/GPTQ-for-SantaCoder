dataset=stack
nsamples=150
opt_level=3
groupsize=128
bits=4

mkdir -p models/$bits-bit

# remove --eval if you dont want to evaluate perplexity of the model
python santacoder.py bigcode/starcoder $dataset --nsamples $nsamples --eval --wbits $bits --act-order --groupsize $groupsize --optimization-level $opt_level --true-sequential --save models/$bits-bit/model.pt
