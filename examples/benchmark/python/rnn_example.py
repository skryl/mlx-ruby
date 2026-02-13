import argparse
import json
import time

import mlx.core as mx
from mlx.nn.layers.recurrent import RNN

RNN_HIDDEN_MULTIPLIER = 2


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=["cpu", "gpu"], required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--sequence-length", type=int, required=True)
    parser.add_argument("--dims", type=int, required=True)
    parser.add_argument("--warmup", type=int, required=True)
    parser.add_argument("--iterations", type=int, required=True)
    return parser.parse_args()


def set_device(device_name):
    mx.set_default_device(mx.cpu if device_name == "cpu" else mx.gpu)


def main():
    args = parse_args()
    set_device(args.device)

    hidden_size = args.dims * RNN_HIDDEN_MULTIPLIER
    warmup_every = max(1, args.warmup // 5)
    iter_every = max(1, args.iterations // 5)

    rnn = RNN(args.dims, hidden_size)
    x = mx.random.uniform(
        low=-1.0,
        high=1.0,
        shape=(args.batch_size, args.sequence_length, args.dims),
        dtype=mx.float32,
    )

    out = None
    for i in range(args.warmup):
        out = rnn(x)
        mx.eval(out)
        if (i + 1) == args.warmup or (i + 1) % warmup_every == 0:
            print(f"[python/rnn] warmup {i + 1}/{args.warmup}", flush=True)

    start = time.perf_counter()
    for i in range(args.iterations):
        out = rnn(x)
        mx.eval(out)
        if (i + 1) == args.iterations or (i + 1) % iter_every == 0:
            print(f"[python/rnn] iter {i + 1}/{args.iterations}", flush=True)
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
