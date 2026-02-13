import argparse
import json
import time

import mlx.core as mx
from mlx.nn.layers.activations import ReLU
from mlx.nn.layers.convolution import Conv2d
from mlx.nn.layers.linear import Linear
from mlx.nn.layers.pooling import MaxPool2d

CNN_CHANNELS = 3
CNN_HEIGHT = 64
CNN_WIDTH = 64
CNN_CLASSES = 1024


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=["cpu", "gpu"], required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--warmup", type=int, required=True)
    parser.add_argument("--iterations", type=int, required=True)
    return parser.parse_args()


def set_device(device_name):
    mx.set_default_device(mx.cpu if device_name == "cpu" else mx.gpu)


def main():
    args = parse_args()
    set_device(args.device)

    warmup_every = max(1, args.warmup // 5)
    iter_every = max(1, args.iterations // 5)
    flattened = 32 * (CNN_HEIGHT // 4) * (CNN_WIDTH // 4)

    conv1 = Conv2d(CNN_CHANNELS, 16, 3, stride=1, padding=1)
    conv2 = Conv2d(16, 32, 3, stride=1, padding=1)
    relu = ReLU()
    pool = MaxPool2d(2, stride=2)
    linear = Linear(flattened, CNN_CLASSES)

    x = mx.random.uniform(
        low=-1.0,
        high=1.0,
        shape=(args.batch_size, CNN_HEIGHT, CNN_WIDTH, CNN_CHANNELS),
        dtype=mx.float32,
    )

    out = None
    for i in range(args.warmup):
        y = conv1(x)
        y = relu(y)
        y = pool(y)
        y = conv2(y)
        y = relu(y)
        y = pool(y)
        y = mx.reshape(y, (args.batch_size, flattened))
        out = linear(y)
        mx.eval(out)
        if (i + 1) == args.warmup or (i + 1) % warmup_every == 0:
            print(f"[python/cnn] warmup {i + 1}/{args.warmup}", flush=True)

    start = time.perf_counter()
    for i in range(args.iterations):
        y = conv1(x)
        y = relu(y)
        y = pool(y)
        y = conv2(y)
        y = relu(y)
        y = pool(y)
        y = mx.reshape(y, (args.batch_size, flattened))
        out = linear(y)
        mx.eval(out)
        if (i + 1) == args.iterations or (i + 1) % iter_every == 0:
            print(f"[python/cnn] iter {i + 1}/{args.iterations}", flush=True)
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
