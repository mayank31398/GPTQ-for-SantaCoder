import time
from argparse import ArgumentParser

import termcolor
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

from gptq import *
from modelutils import *
from quant import *


def disable_torch_init():
    def noop(*args, **kwargs):
        pass

    torch.nn.init.kaiming_uniform_ = noop
    torch.nn.init.uniform_ = noop
    torch.nn.init.normal_ = noop
    transformers.modeling_utils._init_weights = False


def get_santacoder(model, checkpoint, wbits):
    if wbits == 16:
        torch_dtype = torch.bfloat16
    else:
        torch_dtype = torch.float32

    model = AutoModelForCausalLM.from_pretrained(model, torch_dtype=torch_dtype)
    model = model.eval()

    if wbits < 16:
        layers = find_layers(model)
        for name in ["lm_head"]:
            if name in layers:
                del layers[name]
        groupsize = -1
        make_quant(model, layers, wbits, groupsize)

        model.load_state_dict(torch.load(checkpoint))

    model.seqlen = 2048
    model = model.cuda()
    return model


def simple_generation_test(tokenizer, model, prompt):
    batch = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
    batch = {k: v.cuda() for k, v in batch.items()}

    for _ in range(2):
        print("generating...")
        t1 = time.time()
        generated = model.generate(batch["input_ids"], do_sample=False, min_new_tokens=100, max_new_tokens=100)
        t2 = time.time()
        print(termcolor.colored(tokenizer.decode(generated[0]), "yellow"))
        print("generated in %0.2fms" % ((t2 - t1) * 1000))

    print("prompt tokens", len(batch["input_ids"][0]))
    print("all tokens", len(generated[0]))

    generated_tokens = len(generated[0]) - len(batch["input_ids"][0])
    print("%0.1fms per token" % (((t2 - t1) * 1000) / generated_tokens))


def main():
    parser = ArgumentParser()
    parser.add_argument("model", type=str, help="model to load, such as bigcode/gpt_bigcode-santacoder")
    parser.add_argument("--load", type=str, help="load a quantized checkpoint, use normal model if not specified")
    parser.add_argument("--wbits", type=int, default=16, help="bits in quantization checkpoint")
    parser.add_argument("--prompt", type=str, default="pygame example\n\n```", help="prompt the model")
    args = parser.parse_args()

    disable_torch_init()

    t1 = time.time()
    model = get_santacoder(args.model, args.load, args.wbits)
    t2 = time.time()
    print("model load time %0.1fms" % ((t2 - t1) * 1000))

    tokenizer = AutoTokenizer.from_pretrained(args.model)

    simple_generation_test(tokenizer, model, args.prompt)


if __name__ == "__main__":
    main()
