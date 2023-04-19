# GPTQ-for-SantaCoder
Quantization of [SantaCoder](https://arxiv.org/abs/2301.03988) using [GPTQ](https://arxiv.org/abs/2210.17323)

GPTQ is SOTA one-shot weight quantization method

**This code is based on [GPTQ](https://github.com/IST-DASLab/gptq)**

Changed to support new features proposed by [GPTQ](https://github.com/IST-DASLab/gptq#new-features).

* Slightly adjusted preprocessing of C4 and PTB for more realistic evaluations (used in our updated results); can be activated via the flag --new-eval.
* two new tricks:--act-order (quantizing columns in order of decreasing activation size) and --true-sequential (performing sequential quantization even within a single Transformer block). Those fix GPTQ's strangely bad performance on the 7B model (from 7.15 to 6.09 Wiki2 PPL) and lead to slight improvements on most models/settings in general.

## Result
<details>
<summary>SantaCoder (mlp.c_proj is quantized)</summary>

| [SantaCoder](https://arxiv.org/abs/2301.03988)     | Bits | group-size | memory(MiB) | wikitext2 |    ptb     |     c4     | checkpoint size(GB) |
| -------------------------------------------------- | ---- | ---------- | ----------- | --------- | ---------- | ---------- | ------------------- |
| FP32                                               |  32  |     -      |      -      |  24.927   |   38.574   |   27.778   |                     |
| BF16                                               |  16  |     -      |      -      |  24.959   |   38.597   |   27.794   |                     |
| [GPTQ](https://arxiv.org/abs/2210.17323)           |  8   |    128     |      -      |  24.927   |   38.573   |   27.779   |                     |
| [GPTQ](https://arxiv.org/abs/2210.17323)           |  4   |    128     |      -      |  3826.636 |  2649.847  |  2414.551  |                     |
| [GPTQ](https://arxiv.org/abs/2210.17323)           |  3   |    128     |      -      | 62048.417 | 58560.054  |  60491.882 |                     |
| [GPTQ](https://arxiv.org/abs/2210.17323)           |  2   |    128     |      -      | 81707.148 | 92115.679  |  96399.148 |                     |
</details>

<details>
<summary>SantaCoder (mlp.c_fc, mlp.c_proj are quantized)</summary>

| [SantaCoder](https://arxiv.org/abs/2301.03988)     | Bits | group-size | memory(MiB) | wikitext2  |    ptb     |     c4     | checkpoint size(GB) |
| -------------------------------------------------- | ---- | ---------- | ----------- | ---------- | ---------- | ---------- | ------------------- |
| FP32                                               |  32  |     -      |      -      |  24.927    |   38.574   |   27.778   |                     |
| BF16                                               |  16  |     -      |      -      |  24.959    |   38.597   |   27.794   |                     |
| [GPTQ](https://arxiv.org/abs/2210.17323)           |  8   |    128     |      -      |  24.926    |   38.573   |   27.778   |                     |
| [GPTQ](https://arxiv.org/abs/2210.17323)           |  4   |    128     |      -      |  1817.965  |  1447.092  |  1097.042  |                     |
| [GPTQ](https://arxiv.org/abs/2210.17323)           |  3   |    128     |      -      | 30591.978  | 23359.292  |  24260.835 |                     |
| [GPTQ](https://arxiv.org/abs/2210.17323)           |  2   |    128     |      -      | 171096.671 | 146505.640 | 143637.171 |                     |
</details>

<details>
<summary>SantaCoder (attn.c_attn, attn.c_proj, mlp.c_fc, mlp.c_proj are quantized)</summary>

| [SantaCoder](https://arxiv.org/abs/2301.03988)     | Bits | group-size | memory(MiB) | wikitext2 |    ptb     |     c4     | checkpoint size(GB) |
| -------------------------------------------------- | ---- | ---------- | ----------- | --------- | ---------- | ---------- | ------------------- |
| FP32                                               |  32  |     -      |      -      |  24.927   |   38.574   |   27.778   |                     |
| BF16                                               |  16  |     -      |      -      |  24.959   |   38.597   |   27.794   |                     |
| [GPTQ](https://arxiv.org/abs/2210.17323)           |  8   |    128     |      -      |  24.928   |   38.574   |   27.780   |                     |
| [GPTQ](https://arxiv.org/abs/2210.17323)           |  4   |    128     |      -      | 2399.166  |  1790.777  |  1523.385  |                     |
| [GPTQ](https://arxiv.org/abs/2210.17323)           |  3   |    128     |      -      | 62389.542 | 56150.347  | 62935.148  |                     |
| [GPTQ](https://arxiv.org/abs/2210.17323)           |  2   |    128     |      -      | 96986.914 | 117020.460 | 107408.796 |                     |
</details>

<details>
<summary>SantaCoder (attn.c_attn, attn.c_proj are quantized)</summary>

| [SantaCoder](https://arxiv.org/abs/2301.03988)     | Bits | group-size | memory(MiB) | wikitext2 |    ptb     |     c4     | checkpoint size(GB) |
| -------------------------------------------------- | ---- | ---------- | ----------- | --------- | ---------- | ---------- | ------------------- |
| FP32                                               |  32  |     -      |      -      |  24.927   |   38.574   |   27.778   |                     |
| BF16                                               |  16  |     -      |      -      |  24.959   |   38.597   |   27.794   |                     |
| [GPTQ](https://arxiv.org/abs/2210.17323)           |  8   |    128     |      -      |  24.928   |   38.573   |   27.779   |                     |
| [GPTQ](https://arxiv.org/abs/2210.17323)           |  4   |    128     |      -      |  25.001   |   38.721   |   27.842   |                     |
| [GPTQ](https://arxiv.org/abs/2210.17323)           |  3   |    128     |      -      |  25.290   |   39.182   |   28.115   |                     |
| [GPTQ](https://arxiv.org/abs/2210.17323)           |  2   |    128     |      -      |  42.108   |   63.644   |   41.957   |                     |
</details>

<details>
<summary>SantaCoder (attn.c_attn, attn.c_proj, mlp.c_fc are quantized)</summary>
crashed :(
</details>

Quantization requires a large amount of CPU memory. However, the memory required can be reduced by using swap memory.

Depending on the GPUs/drivers, there may be a difference in performance, which decreases as the model size increases.(https://github.com/IST-DASLab/gptq/issues/1)

According to [GPTQ paper](https://arxiv.org/abs/2210.17323), As the size of the model increases, the difference in performance between FP16 and GPTQ decreases.

## Installation
If you don't have [conda](https://docs.conda.io/en/latest/miniconda.html), install it first.
```
conda create --name gptq python=3.9 -y
conda activate gptq
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
# Or, if you're having trouble with conda, use pip with python3.9:
# pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117

pip install -r requirements.txt
python setup_cuda.py install

# Benchmark performance for FC2 layer of LLaMa-7B
CUDA_VISIBLE_DEVICES=0 python test_kernel.py
```
## Dependencies

* `torch`: tested on v2.0.0+cu117
* `transformers`: tested on v4.28.0.dev0
* `datasets`: tested on v2.10.1
* `safetensors`: tested on v0.3.0
* (to run 4-bit kernels: setup for compiling PyTorch CUDA extensions, see also https://pytorch.org/tutorials/advanced/cpp_extension.html, tested on CUDA 11.7)

All experiments were run on a single NVIDIA RTX3090.

# Language Generation
## LLaMA

```
#convert LLaMA to hf
python convert_llama_weights_to_hf.py --input_dir /path/to/downloaded/llama/weights --model_size 7B --output_dir ./llama-hf

# Benchmark language generation with 4-bit LLaMA-7B:

# Save compressed model
CUDA_VISIBLE_DEVICES=0 python llama.py ./llama-hf/llama-7b c4 --wbits 4 --true-sequential --act-order --groupsize 128 --save llama7b-4bit-128g.pt
# Or save compressed `.safetensors` model
CUDA_VISIBLE_DEVICES=0 python llama.py ./llama-hf/llama-7b c4 --wbits 4 --true-sequential --act-order --groupsize 128 --save_safetensors llama7b-4bit-128g.safetensors

# Benchmark generating a 2048 token sequence with the saved model
CUDA_VISIBLE_DEVICES=0 python llama.py ./llama-hf/llama-7b c4 --wbits 4 --groupsize 128 --load llama7b-4bit-128g.pt --benchmark 2048 --check
# Benchmark FP16 baseline, note that the model will be split across all listed GPUs
CUDA_VISIBLE_DEVICES=0,1,2,3,4 python llama.py ./llama-hf/llama-7b c4 --benchmark 2048 --check

# model inference with the saved model
CUDA_VISIBLE_DEVICES=0 python llama_inference.py ./llama-hf/llama-7b --wbits 4 --groupsize 128 --load llama7b-4bit-128g.pt --text "this is llama"
# model inference with the saved model using safetensors loaded direct to gpu
CUDA_VISIBLE_DEVICES=0 python llama_inference.py ./llama-hf/llama-7b --wbits 4 --groupsize 128 --load llama7b-4bit-128g.safetensors --text "this is llama --device=0
# model inference with the saved model with offload(This is very slow. This is a simple implementation and could be improved with technologies like flexgen(https://github.com/FMInference/FlexGen).
CUDA_VISIBLE_DEVICES=0 python llama_inference_offload.py ./llama-hf/llama-7b --wbits 4 --groupsize 128 --load llama7b-4bit-128g.pt --text "this is llama" --pre_layer 16
It takes about 180 seconds to generate 45 tokens(5->50 tokens) on single RTX3090 based on LLaMa-65B. pre_layer is set to 50.
```
Basically, 4-bit quantization and 128 groupsize are recommended.

# Acknowledgements
This code is based on [GPTQ](https://github.com/IST-DASLab/gptq)

Triton GPTQ kernel code is based on [GPTQ-triton](https://github.com/fpgaminer/GPTQ-triton)
