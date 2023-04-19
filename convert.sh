# CUDA_VISIBLE_DEVICES=0 python santacoder.py bigcode/gpt_bigcode-santacoder wikitext2 --wbits 4 --true-uential --act-order --groupsize 128 --save santacoder.pt --nsamples 128
# CUDA_VISIBLE_DEVICES=0 python santacoder.py bigcode/gpt_bigcode-santacoder wikitext2 --nsamples 128 --eval
# CUDA_VISIBLE_DEVICES=0 python santacoder.py bigcode/gpt_bigcode-santacoder wikitext2 --wbits 4 --true-sequential --act-order --groupsize 128 --load santacoder.pt --nsamples 128 --eval
# CUDA_VISIBLE_DEVICES=0 python opt.py facebook/opt-350m wikitext2 --wbits 4 --act-order --groupsize 128 --save opt.pt --nsamples 128

dataset=wikitext2
nsamples=128

for bits in 32 16
do
    CUDA_VISIBLE_DEVICES=0 python santacoder.py bigcode/gpt_bigcode-santacoder $dataset --nsamples $nsamples --eval --wbits $bits |& tee $bits-0.log
done

groupsize=128

for bits in 8 4 3 2
do
    for opt_level in -3 -2 -1 1 2 3
    do
        CUDA_VISIBLE_DEVICES=0 python santacoder.py bigcode/gpt_bigcode-santacoder $dataset --nsamples $nsamples --eval --wbits $bits --act-order --groupsize $groupsize --optimization-level $opt_level --true-sequential |& tee $bits-$opt_level.log
    done
done
# CUDA_VISIBLE_DEVICES=0 python opt.py facebook/opt-350m wikitext2 --nsamples 128 --eval
# CUDA_VISIBLE_DEVICES=0 python opt.py facebook/opt-350m wikitext2 --nsamples 128 --eval --wbits 16 --act-order --groupsize 128
# CUDA_VISIBLE_DEVICES=0 python opt.py facebook/opt-350m wikitext2 --nsamples 128 --eval --wbits 8 --act-order --groupsize 128
# CUDA_VISIBLE_DEVICES=0 python opt.py facebook/opt-350m wikitext2 --nsamples 128 --eval --wbits 4 --act-order --groupsize 128
# CUDA_VISIBLE_DEVICES=0 python opt.py facebook/opt-350m wikitext2 --nsamples 128 --eval --wbits 3 --act-order --groupsize 128
# CUDA_VISIBLE_DEVICES=0 python opt.py facebook/opt-350m wikitext2 --nsamples 128 --eval --wbits 2 --act-order --groupsize 128
