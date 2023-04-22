dataset=stack
nsamples=128
opt_level=3
groupsize=-1

# evaluate perplexity of fp32 and bf16
for bits in 32 16
do
    python santacoder.py bigcode/gpt_bigcode-santacoder $dataset --nsamples $nsamples --eval --wbits $bits |& tee $bits-0.log
done

# 3 and 2 bit models are crap
for bits in 8 4 3 2
do
    mkdir -p models/$bits-bit

    # remove --eval if you dont want to evaluate perplexity of the model
    python santacoder.py bigcode/gpt_bigcode-santacoder $dataset --nsamples $nsamples --eval --wbits $bits --act-order --groupsize $groupsize --optimization-level $opt_level --true-sequential --save models/$bits-bit/model.pt |& tee $bits-$opt_level.log
done
