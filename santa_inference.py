import time
import torch
import termcolor
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


def load_quant(model, checkpoint, wbits):
    model = AutoModelForCausalLM.from_pretrained(model, torch_dtype="auto", trust_remote_code=True)
    model.seqlen = 2048

    torch.set_default_dtype(torch.float)
    model = model.eval()
    layers = find_layers(model)
    for name in ['lm_head']:
        if name in layers:
            del layers[name]
    groupsize = -1
    make_quant(model, layers, wbits, groupsize)

    print(f'Loading quantized checkpoint {checkpoint}...')
    model.load_state_dict(torch.load(checkpoint))
    model.seqlen = 2048

    return model


def simple_generation_test(model_name, checkpoint_fn, wbits, prompt):
    t1 = time.time()
    disable_torch_init()  # reduces load time from 15s to 6s (santa)
    if checkpoint_fn:
        model = load_quant(model_name, checkpoint_fn, wbits).to('cuda')
    else:
        print(f'Loading base non-quantized model {model_name}...')
        torch.set_default_dtype(torch.half)
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", trust_remote_code=True)
        model = model.eval()
        model = model.cuda()
        model.seqlen = 2048
    t2 = time.time()
    print("model load time %0.1fms" % ((t2-t1)*1000))

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    batch = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
    batch = {k: v.cuda() for k, v in batch.items()}

    for _ in range(2):
        print("generating...")
        t1 = time.time()
        generated = model.generate(
            batch["input_ids"],
            do_sample=True,
            use_cache=True,
            repetition_penalty=1.1,
            max_new_tokens=200,
            temperature=0.2,
            top_p=0.95,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            )
        t2 = time.time()
        print(termcolor.colored(tokenizer.decode(generated[0]), 'yellow'))
        print("generated in %0.2fms" % ((t2-t1)*1000))
    print("prompt tokens", len(batch["input_ids"][0]))
    print("all tokens", len(generated[0]))
    generated_tokens = len(generated[0]) - len(batch["input_ids"][0])
    print("%0.1fms per token" % (((t2-t1)*1000) / generated_tokens))


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=str, help='model to load, such as bigcode/gpt_bigcode-santacoder')
    parser.add_argument('--load', type=str, help='load a quantized checkpoint, use normal model if not specified')
    parser.add_argument("--wbits", type=int, default=8, help='bits in quantization checkpoint')
    parser.add_argument('--prompt', type=str, help='prompt the model')
    args = parser.parse_args()
    prompt = args.prompt or "pygame example\n\n```"
    simple_generation_test(args.model, args.load, args.wbits, prompt)


if __name__ == '__main__':
    # Example:
    # python my_test.py bigcode/gpt_bigcode-santacoder --load santa-8bit.pt --prompt "import matplotlib"
    main()

