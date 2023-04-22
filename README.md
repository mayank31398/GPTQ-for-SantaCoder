# GPTQ-for-SantaCoder
Quantization of [SantaCoder](https://arxiv.org/abs/2301.03988) using [GPTQ](https://arxiv.org/abs/2210.17323)

GPTQ is SOTA one-shot weight quantization method

**This code is based on [GPTQ](https://github.com/IST-DASLab/gptq)**

Changed to support new features proposed by [GPTQ](https://github.com/IST-DASLab/gptq#new-features).

* Slightly adjusted preprocessing of C4 and PTB for more realistic evaluations (used in our updated results); can be activated via the flag --new-eval.
* two new tricks:--act-order (quantizing columns in order of decreasing activation size) and --true-sequential (performing sequential quantization even within a single Transformer block). Those fix GPTQ's strangely bad performance on the 7B model (from 7.15 to 6.09 Wiki2 PPL) and lead to slight improvements on most models/settings in general.

## Result
| [SantaCoder](https://arxiv.org/abs/2301.03988)     | Bits | group-size | memory(MiB) | wikitext2 |    ptb     |     c4     |   stack    | checkpoint size(MB) |
| -------------------------------------------------- | ---- | ---------- | ----------- | --------- | ---------- | ---------- | ---------- | ------------------- |
| FP32                                               |  32  |     -      |  4344.722   |  24.927   |   38.574   |   27.779   |   2.619    |        4394         |
| BF16                                               |  16  |     -      |  2173.680   |  24.960   |   38.597   |   27.794   |   2.621    |        2195         |
| [GPTQ](https://arxiv.org/abs/2210.17323)           |  8   |     -1     |  1396.548   |  24.936   |   38.592   |   27.785   |   2.619    |        1411         |
| [GPTQ](https://arxiv.org/abs/2210.17323)           |  4   |     -1     |   911.384   |  26.581   |   40.717   |   29.232   |   2.658    |         913         |
| [GPTQ](https://arxiv.org/abs/2210.17323)           |  3   |     -1     |      -      | 11761.473 |  7273.338  |  9124.941  |  2485.844  |         789         |
| [GPTQ](https://arxiv.org/abs/2210.17323)           |  2   |     -1     |      -      | 67976.797 | 68994.484  | 73294.438  | 45370.488  |         649         |

Quantization requires a large amount of CPU memory. However, the memory required can be reduced by using swap memory.

Depending on the GPUs/drivers, there may be a difference in performance, which decreases as the model size increases.(https://github.com/IST-DASLab/gptq/issues/1)

According to [GPTQ paper](https://arxiv.org/abs/2210.17323), As the size of the model increases, the difference in performance between FP16 and GPTQ decreases.

## Installation
If you don't have [conda](https://docs.conda.io/en/latest/miniconda.html), install it first.
```shell
conda create --name gptq python=3.9 -y
conda activate gptq
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
# Or, if you're having trouble with conda, use pip with python3.9:
# pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117

pip install -r requirements.txt
python setup_cuda.py install
```

## Dependencies

* `torch`: tested on v2.0.0+cu117
* `transformers`: tested on v4.28.0.dev0
* `datasets`: tested on v2.10.1
* `safetensors`: tested on v0.3.0
* (to run 4-bit kernels: setup for compiling PyTorch CUDA extensions, see also https://pytorch.org/tutorials/advanced/cpp_extension.html, tested on CUDA 11.7)

All experiments were run on a single NVIDIA RTX3090.

# Language Generation
## SantaCoder
Visit [mayank31398/santacoder-GPTQ](https://huggingface.co/mayank31398/santacoder-GPTQ) for the quantized weights.
Alternatively, you can also use [convert.sh](convert.sh) to get the quantized models and save them to disk.

For benchmarking, use [benchmark.sh](benchmark.sh).

```shell
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
