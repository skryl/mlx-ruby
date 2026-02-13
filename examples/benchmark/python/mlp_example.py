import argparse
import json
import time

import mlx.core as mx
from mlx.nn.layers.activations import ReLU
from mlx.nn.layers.linear import Linear

MLP_FACTOR = 4


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=["cpu", "gpu"], required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--dims", type=int, required=True)
    parser.add_argument("--warmup", type=int, required=True)
    parser.add_argument("--iterations", type=int, required=True)
    return parser.parse_args()


def set_device(device_name):
    mx.set_default_device(mx.cpu if device_name == "cpu" else mx.gpu)


def main():
    args = parse_args()
    set_device(args.device)

    input_size = args.dims * MLP_FACTOR
    hidden_size = args.dims * MLP_FACTOR
    output_size = args.dims
    warmup_every = max(1, args.warmup // 5)
    iter_every = max(1, args.iterations // 5)

    layer1 = Linear(input_size, hidden_size)
    layer2 = Linear(hidden_size, hidden_size)
    layer3 = Linear(hidden_size, output_size)
    relu = ReLU()

    x = mx.random.uniform(
        low=-1.0,
        high=1.0,
        shape=(args.batch_size, input_size),
        dtype=mx.float32,
    )

    out = None
    for i in range(args.warmup):
        y = layer1(x)
        y = relu(y)
        y = layer2(y)
        y = relu(y)
        out = layer3(y)
        mx.eval(out)
        if (i + 1) == args.warmup or (i + 1) % warmup_every == 0:
            print(f"[python/mlp] warmup {i + 1}/{args.warmup}", flush=True)

    start = time.perf_counter()
    for i in range(args.iterations):
        y = layer1(x)
        y = relu(y)
        y = layer2(y)
        y = relu(y)
        out = layer3(y)
        mx.eval(out)
        if (i + 1) == args.iterations or (i + 1) % iter_every == 0:
            print(f"[python/mlp] iter {i + 1}/{args.iterations}", flush=True)
    elapsed = time.perf_counter() - start

    print(
        json.dumps(
            {
                "average_ms": (elapsed / args.iterations) * 1000.0,
                "iterations": args.iterations,
                "warmup": args.warmup,
                "output_shape": list(out.shape),
            }
        )
    )


if __name__ == "__main__":
    main()
