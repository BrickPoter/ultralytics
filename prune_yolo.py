import argparse
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from ultralytics.models.yolo.model import YOLO
from ultralytics.utils.torch_utils import select_device


def collect_modules(model, include_bias=False):
    mods = []
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            mods.append((m, "weight"))
            if include_bias and m.bias is not None:
                mods.append((m, "bias"))
    return mods


def prune_model(model, amount=0.3, include_bias=False):
    modules = collect_modules(model, include_bias)
    prune.global_unstructured(modules, pruning_method=prune.L1Unstructured, amount=amount)
    for m, p in modules:
        prune.remove(m, p)
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--amount", type=float, default=0.3)
    parser.add_argument("--include_bias", action="store_true")
    parser.add_argument("--device", type=str, default="")
    parser.add_argument("--save", type=str, default="yolo_pruned.pt")
    args = parser.parse_args()
    select_device(args.device, 1)
    y = YOLO(args.model)
    m = y.model
    m.eval()
    m = prune_model(m, amount=args.amount, include_bias=args.include_bias)
    torch.save({"model": m}, args.save)


if __name__ == "__main__":
    main()
