# GPTQ-for-SantaCoder-and-StarCoder
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

## Result
| StarCoder                                          | Bits | group-size | memory(MiB) | wikitext2 |    ptb     |     c4     |   stack    | checkpoint size(MB) |
| -------------------------------------------------- | ---- | ---------- | ----------- | --------- | ---------- | ---------- | ---------- | ------------------- |
| FP32                                               |  32  |     -      |             |  10.801   |   16.425   |   13.402   |   1.738    |       59195         |
| BF16                                               |  16  |     -      |             |  10.807   |   16.424   |   13.408   |   1.739    |       29597         |
| [GPTQ](https://arxiv.org/abs/2210.17323)           |  8   |    128     |             |  10.805   |   15.453   |   13.408   |   1.739    |       16163         |
| [GPTQ](https://arxiv.org/abs/2210.17323)           |  4   |    128     |             |  10.989   |   16.839   |   13.676   |   1.757    |        8877         |

## Result
| StarCoderBase                                      | Bits | group-size | memory(MiB) | wikitext2 |    ptb     |     c4     |   stack    | checkpoint size(MB) |
| -------------------------------------------------- | ---- | ---------- | ----------- | --------- | ---------- | ---------- | ---------- | ------------------- |
| FP32                                               |  32  |     -      |             |  10.172   |   15.756   |   12.736   |   1.692    |       59195         |
| BF16                                               |  16  |     -      |             |  10.173   |   15.765   |   12.745   |   1.692    |       29597         |
| [GPTQ](https://arxiv.org/abs/2210.17323)           |  8   |    128     |             |  10.174   |   15.767   |   12.739   |   1.692    |       16163         |
| [GPTQ](https://arxiv.org/abs/2210.17323)           |  4   |    128     |             |  10.387   |   16.056   |   13.005   |   1.708    |        8877         |

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

All experiments were run on a single NVIDIA RTX3090.

# Language Generation
## SantaCoder
Visit [mayank31398/santacoder-GPTQ-4bit-128g](https://huggingface.co/mayank31398/santacoder-GPTQ-4bit-128g) for the 4-bit weights.
Visit [mayank31398/santacoder-GPTQ-8bit-128g](https://huggingface.co/mayank31398/santacoder-GPTQ-8bit-128g) for the 8-bit weights.
```shell
# 4-bit
git clone https://huggingface.co/mayank31398/santacoder-GPTQ-4bit-128g
# 8-bit
git clone https://huggingface.co/mayank31398/santacoder-GPTQ-8bit-128g
```
Alternatively, you can also use the [scripts](scripts/) to get the quantized models and save them to disk.

For generation use:
```shell
# fp32
python -m santacoder_inference bigcode/gpt_bigcode-santacoder --wbits 32
# bf16
python -m santacoder_inference bigcode/gpt_bigcode-santacoder --wbits 16

# GPTQ int8
python -m santacoder_inference bigcode/gpt_bigcode-santacoder --wbits 8 --load santacoder-GPTQ-8bit-128g/model.pt
# GPTQ int4
python -m santacoder_inference bigcode/gpt_bigcode-santacoder --wbits 4 --load santacoder-GPTQ-4bit-128g/model.pt
```

## StarCoder
Visit [mayank31398/starcoder-GPTQ-4bit-128g](https://huggingface.co/mayank31398/starcoder-GPTQ-4bit-128g) for the 4-bit weights.
Visit [mayank31398/starcoder-GPTQ-8bit-128g](https://huggingface.co/mayank31398/starcoder-GPTQ-8bit-128g) for the 8-bit weights.
```shell
# 4-bit
git clone https://huggingface.co/mayank31398/starcoder-GPTQ-4bit-128g
# 8-bit
git clone https://huggingface.co/mayank31398/starcoder-GPTQ-8bit-128g
```
Alternatively, you can also use the [scripts](scripts/) to get the quantized models and save them to disk.

For generation use:
```shell
# fp32
python -m santacoder_inference bigcode/starcoder --wbits 32
# bf16
python -m santacoder_inference bigcode/starcoder --wbits 16

# GPTQ int8
python -m santacoder_inference bigcode/starcoder --wbits 8 --load starcoder-GPTQ-8bit-128g/model.pt
# GPTQ int4
python -m santacoder_inference bigcode/starcoder --wbits 4 --load starcoder-GPTQ-4bit-128g/model.pt
```

## StarCoderBase
Visit [mayank31398/starcoderbase-GPTQ-4bit-128g](https://huggingface.co/mayank31398/starcoderbase-GPTQ-4bit-128g) for the 4-bit weights.
Visit [mayank31398/starcoderbase-GPTQ-8bit-128g](https://huggingface.co/mayank31398/starcoderbase-GPTQ-8bit-128g) for the 8-bit weights.
```shell
# 4-bit
git clone https://huggingface.co/mayank31398/starcoderbase-GPTQ-4bit-128g
# 8-bit
git clone https://huggingface.co/mayank31398/starcoderbase-GPTQ-8bit-128g
```
Alternatively, you can also use the [scripts](scripts/) to get the quantized models and save them to disk.

For generation use:
```shell
# fp32
python -m santacoder_inference bigcode/starcoderbase --wbits 32
# bf16
python -m santacoder_inference bigcode/starcoderbase --wbits 16

# GPTQ int8
python -m santacoder_inference bigcode/starcoderbase --wbits 8 --load starcoderbase-GPTQ-8bit-128g/model.pt
# GPTQ int4
python -m santacoder_inference bigcode/starcoderbase --wbits 4 --load starcoderbase-GPTQ-4bit-128g/model.pt
```

# Acknowledgements
This code is based on [GPTQ](https://github.com/IST-DASLab/gptq)

Triton GPTQ kernel code is based on [GPTQ-triton](https://github.com/fpgaminer/GPTQ-triton)
