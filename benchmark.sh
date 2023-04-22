dataset=c4
groupsize=-1
seqlen=2048
load_path=santacoder-GPTQ

# evaluate perplexity of fp32 and bf16
for bits in 32 16
do
    python santacoder.py bigcode/gpt_bigcode-santacoder $dataset --wbits $bits --benchmark $seqlen --check
done

# 3 and 2 bit models are crap
for bits in 8 4 3 2
do
    python santacoder.py bigcode/gpt_bigcode-santacoder $dataset --wbits $bits --groupsize $groupsize --load $load_path/$bits-bit/model.pt --benchmark $seqlen --check
done
