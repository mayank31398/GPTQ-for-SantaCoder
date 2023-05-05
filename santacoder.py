import time

import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM

from gptq import *
from modelutils import *
from quant import *

torch.nn.init.kaiming_uniform_ = lambda *args, **kwargs: None
torch.nn.init.uniform_ = lambda *args, **kwargs: None
torch.nn.init.normal_ = lambda *args, **kwargs: None


def get_santacoder(model, wbits):
    if wbits == 16:
        torch_dtype = torch.bfloat16
    else:
        torch_dtype = torch.float32

    model = AutoModelForCausalLM.from_pretrained(model, torch_dtype=torch_dtype)
    model.seqlen = 2048
    return model


def setup(nsamples, model, batch_iterator, dev):
    model.config.use_cache = False
    layers = model.transformer.h

    model.transformer.wte = model.transformer.wte.to(dev)
    model.transformer.wpe = model.transformer.wpe.to(dev)
    model.transformer.ln_f = model.transformer.ln_f.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros((nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev)
    cache = {"i": 0, "attention_mask": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache["i"]] = inp
            cache["i"] += 1
            cache["attention_mask"] = kwargs["attention_mask"]
            raise ValueError

    layers[0] = Catcher(layers[0])

    for batch in batch_iterator:
        try:
            model(batch.to(dev))
        except ValueError:
            pass

    layers[0] = layers[0].module

    model.transformer.wte = model.transformer.wte.cpu()
    model.transformer.wpe = model.transformer.wpe.cpu()
    model.transformer.ln_f = model.transformer.ln_f.cpu()
    layers[0] = layers[0].cpu()

    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache["attention_mask"].to(dev)

    return layers, inps, outs, attention_mask


@torch.no_grad()
def santacoder_sequential(model, dataloader, dev, level):
    def get_batch_iterator(data, nsamples):
        for batch in data:
            yield batch[0]

    use_cache = model.config.use_cache

    print("Starting ...")
    layers, inps, outs, attention_mask = setup(
        args.nsamples, model, get_batch_iterator(dataloader, args.nsamples), dev
    )
    print("Ready.")

    quantizers = {}
    for i in tqdm(range(len(layers))):
        print(f"layer {i}")

        layer = layers[i].to(dev)
        full = find_layers(layer)
        if args.true_sequential:
            sequential = [
                ["attn.c_attn", "attn.c_proj"],
                ["mlp.c_fc"],
                ["mlp.c_proj"],
            ]
            if level >= 0:
                sequential = sequential[:level]
            else:
                sequential = sequential[level:]
            print("quantization target =", sequential)
        else:
            sequential = [list(full.keys())]

        for names in sequential:
            subset = {n: full[n] for n in names}
            gptq = {}
            for name in subset:
                gptq[name] = GPTQ(subset[name])
                gptq[name].quantizer = Quantizer()
                gptq[name].quantizer.configure(args.wbits, perchannel=True, sym=args.sym, mse=False)

            def add_batch(name):
                def tmp(_, inp, out):
                    gptq[name].add_batch(inp[0].data, out.data)

                return tmp

            handles = []
            for name in subset:
                handles.append(subset[name].register_forward_hook(add_batch(name)))
            for j in range(args.nsamples):
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
            for h in handles:
                h.remove()

            for name in subset:
                print(f"Quantizing {name} in layer {i+1}/{len(layers)}...")
                scale, zero, g_idx = gptq[name].fasterquant(
                    percdamp=args.percdamp, groupsize=args.groupsize, actorder=args.act_order
                )
                quantizers["transformer.h.%d.%s" % (i, name)] = (
                    gptq[name].quantizer.cpu(),
                    scale.cpu(),
                    zero.cpu(),
                    g_idx.cpu(),
                )
                gptq[name].free()

        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]

        layers[i] = layer.cpu()
        del layer
        del gptq
        torch.cuda.empty_cache()

        inps, outs = outs, inps

    model.config.use_cache = use_cache

    return quantizers


@torch.no_grad()
def santacoder_eval(model, testenc, dev, dataset_name):
    def get_batch_iterator(data, nsamples):
        for i in range(nsamples):
            yield data[:, (i * model.seqlen) : ((i + 1) * model.seqlen)]

    print("Evaluating ...")

    if dataset_name != "stack":
        testenc = testenc.input_ids
    nsamples = testenc.numel() // model.seqlen

    use_cache = model.config.use_cache

    layers, inps, outs, attention_mask = setup(nsamples, model, get_batch_iterator(testenc, nsamples), dev)

    for i in range(len(layers)):
        layer = layers[i].to(dev)

        if args.nearest:
            subset = find_layers(layer)
            for name in subset:
                quantizer = Quantizer()
                quantizer.configure(args.wbits, perchannel=True, sym=args.sym, mse=False)
                W = subset[name].weight.data
                quantizer.find_params(W, weight=True)
                subset[name].weight.data = quantize(W, quantizer.scale, quantizer.zero, quantizer.maxq).to(
                    next(iter(layer.parameters())).dtype
                )

        for j in range(nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
        layers[i] = layer.cpu()
        del layer
        torch.cuda.empty_cache()
        inps, outs = outs, inps

    if model.transformer.ln_f is not None:
        model.transformer.ln_f = model.transformer.ln_f.to(dev)
    model.lm_head = model.lm_head.to(dev)

    testenc = testenc.to(dev)
    nlls = []
    for i in range(nsamples):
        hidden_states = inps[i].unsqueeze(0)
        if model.transformer.ln_f is not None:
            hidden_states = model.transformer.ln_f(hidden_states)
        lm_logits = model.lm_head(hidden_states)
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = testenc[:, (i * model.seqlen) : ((i + 1) * model.seqlen)][:, 1:]
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        neg_log_likelihood = loss.float() * model.seqlen
        nlls.append(neg_log_likelihood)
    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))
    print(ppl.item())

    model.config.use_cache = use_cache


# TODO: perform packing on GPU
def santacoder_pack(model, quantizers, wbits, groupsize):
    layers = find_layers(model)
    layers = {n: layers[n] for n in quantizers}
    make_quant(model, quantizers, wbits, groupsize)
    qlayers = find_layers(model, [QuantLinear])
    print("Packing ...")
    for name in qlayers:
        print(name)
        quantizers[name], scale, zero, g_idx = quantizers[name]
        qlayers[name].pack(layers[name], scale, zero, g_idx)
    print("Done.")
    return model


def load_quant(model, checkpoint, wbits, groupsize=-1):
    config = AutoConfig.from_pretrained(model)

    torch.set_default_dtype(torch.half)
    transformers.modeling_utils._init_weights = False
    torch.set_default_dtype(torch.half)
    model = AutoModelForCausalLM.from_config(config)
    torch.set_default_dtype(torch.float)
    model = model.eval()
    layers = find_layers(model)
    for name in ["lm_head"]:
        if name in layers:
            del layers[name]
    make_quant(model, layers, wbits, groupsize)

    del layers

    print("Loading model ...")
    if checkpoint.endswith(".safetensors"):
        from safetensors.torch import load_file as safe_load

        model.load_state_dict(safe_load(checkpoint), strict=False)
    else:
        model.load_state_dict(torch.load(checkpoint), strict=False)
    model.seqlen = 2048
    print("Done.")

    return model


def benchmark(model, input_ids, check=False):
    input_ids = input_ids.to(model.gpus[0] if hasattr(model, "gpus") else DEV)
    torch.cuda.synchronize()

    cache = {"past": None}

    def clear_past(i):
        def tmp(layer, inp, out):
            if cache["past"]:
                cache["past"][i] = None

        return tmp

    for i, layer in enumerate(model.transformer.h):
        layer.register_forward_hook(clear_past(i))

    print("Benchmarking ...")

    if check:
        loss = nn.CrossEntropyLoss()
        tot = 0.0

    def sync():
        if hasattr(model, "gpus"):
            for gpu in model.gpus:
                torch.cuda.synchronize(gpu)
        else:
            torch.cuda.synchronize()

    max_memory = 0
    with torch.no_grad():
        attention_mask = torch.ones((1, input_ids.numel()), device=DEV)
        times = []
        for i in range(input_ids.numel()):
            tick = time.time()
            out = model(
                input_ids[:, i : i + 1],
                past_key_values=cache["past"],
                attention_mask=attention_mask[:, : (i + 1)].reshape((1, -1)),
            )
            sync()
            times.append(time.time() - tick)
            print(i, times[-1])
            max_memory = max(max_memory, torch.cuda.memory_allocated() / 1024 / 1024)
            if check and i != input_ids.numel() - 1:
                tot += loss(out.logits[0].to(DEV), input_ids[:, (i + 1)].to(DEV)).float()
            cache["past"] = list(out.past_key_values)
            del out
        sync()
        import numpy as np

        print("Median:", np.median(times))
        if check:
            print("PPL:", torch.exp(tot / (input_ids.numel() - 1)).item())
            print("max memory(MiB):", max_memory)


if __name__ == "__main__":
    import argparse

    from datautils import *

    parser = argparse.ArgumentParser()

    parser.add_argument("model", type=str, help="model to load")
    parser.add_argument(
        "dataset",
        type=str,
        choices=["wikitext2", "ptb", "c4", "stack"],
        help="Where to extract calibration data from.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Seed for sampling the calibration data.")
    parser.add_argument("--nsamples", type=int, default=128, help="Number of calibration data samples.")
    parser.add_argument(
        "--percdamp", type=float, default=0.01, help="Percent of the average Hessian diagonal to use for dampening."
    )
    parser.add_argument("--nearest", action="store_true", help="Whether to run the RTN baseline.")
    parser.add_argument(
        "--wbits",
        type=int,
        default=32,
        choices=[2, 3, 4, 8, 16, 32],
        help="#bits to use for quantization; use 16 for evaluating base model.",
    )
    parser.add_argument("--trits", action="store_true", help="Whether to use trits for quantization.")
    parser.add_argument(
        "--groupsize", type=int, default=-1, help="Groupsize to use for quantization; default uses full row."
    )
    parser.add_argument("--eval", action="store_true", help="evaluate quantized model.")
    parser.add_argument("--save", type=str, default="", help="Save quantized checkpoint under this name.")
    parser.add_argument(
        "--save_safetensors", type=str, default="", help="Save quantized `.safetensors` checkpoint under this name."
    )
    parser.add_argument("--load", type=str, default="", help="Load quantized model.")
    parser.add_argument("--benchmark", type=int, default=0, help="Number of tokens to use for benchmarking.")
    parser.add_argument(
        "--check", action="store_true", help="Whether to compute perplexity during benchmarking for verification."
    )
    parser.add_argument("--sym", action="store_true", help="Whether to perform symmetric quantization.")
    parser.add_argument(
        "--act-order", action="store_true", help="Whether to apply the activation order GPTQ heuristic"
    )
    parser.add_argument("--true-sequential", action="store_true", help="Whether to run in true sequential model.")
    parser.add_argument(
        "--optimization-level",
        type=int,
        choices=[-2, -1, 1, 2, 3],
        help="Whether to run in true sequential model.",
    )
    parser.add_argument("--new-eval", action="store_true", help="Whether to use the new PTB and C4 eval")

    args = parser.parse_args()

    if type(args.load) is not str:
        args.load = args.load.as_posix()

    if args.load:
        model = load_quant(args.model, args.load, args.wbits, args.groupsize)
    else:
        model = get_santacoder(args.model, args.wbits)
        model.eval()

    dataloader, testloader = get_loaders(
        args.dataset, nsamples=args.nsamples, seed=args.seed, model=args.model, seqlen=model.seqlen
    )

    if not args.load and args.wbits < 16 and not args.nearest:
        tick = time.time()
        quantizers = santacoder_sequential(model, dataloader, DEV, args.optimization_level)
        print(time.time() - tick)
        santacoder_pack(model, quantizers, args.wbits, args.groupsize)

    if args.benchmark:
        model = model.to(DEV)
        if args.benchmark:
            input_ids = next(iter(dataloader))[0][:, : args.benchmark]
            benchmark(model, input_ids, check=args.check)

    if args.eval:
        datasets = ["wikitext2", "ptb", "c4", "stack"]
        if args.new_eval:
            datasets = ["wikitext2", "ptb-new", "c4-new"]
        for dataset in datasets:
            dataloader, testloader = get_loaders(dataset, seed=args.seed, model=args.model, seqlen=model.seqlen)
            print(dataset)
            santacoder_eval(model, testloader, DEV, dataset)

    if args.save:
        torch.save(model.state_dict(), args.save)

    if args.save_safetensors:
        from safetensors.torch import save_file as safe_save

        safe_save(model.state_dict(), args.save_safetensors)
