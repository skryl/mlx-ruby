import argparse
import json
import time

import mlx.core as mx
from benchmark_digest import assign_deterministic_parameters
from benchmark_digest import deterministic_tensor
from benchmark_digest import digest_array
from mlx.nn.layers.activations import relu
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

    x = deterministic_tensor(
        (args.batch_size, CNN_HEIGHT, CNN_WIDTH, CNN_CHANNELS),
        mx.float32,
        offset=0,
    )
    input_shape = list(x.shape)
    input_digest = digest_array(x)

    conv1 = Conv2d(CNN_CHANNELS, 16, 3, stride=1, padding=1)
    conv2 = Conv2d(16, 32, 3, stride=1, padding=1)
    pool = MaxPool2d(2, stride=2)
    linear = Linear(flattened, CNN_CLASSES)
    assign_deterministic_parameters([conv1, conv2, linear])

    def run_step():
        y = conv1(x)
        y = relu(y)
        y = pool(y)
        y = conv2(y)
        y = relu(y)
        y = pool(y)
        y = mx.reshape(y, (args.batch_size, flattened))
        return linear(y)

    reference_output_digest = digest_array(run_step())
    path_signature = "forward_only_eval_output"

    out = None
    for i in range(args.warmup):
        out = run_step()
        mx.eval(out)
        if (i + 1) == args.warmup or (i + 1) % warmup_every == 0:
            print(f"[python/cnn] warmup {i + 1}/{args.warmup}", flush=True)

    start = time.perf_counter()
    for i in range(args.iterations):
        out = run_step()
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
                "input_shape": input_shape,
                "output_shape": list(out.shape),
                "input_digest": input_digest,
                "reference_output_digest": reference_output_digest,
                "path_signature": path_signature,
            }
        )
    )


if __name__ == "__main__":
    main()
