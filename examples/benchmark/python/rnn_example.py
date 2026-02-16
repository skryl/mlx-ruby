import argparse
import json
import time

import mlx.core as mx
from benchmark_digest import assign_deterministic_parameters
from benchmark_digest import deterministic_tensor
from benchmark_digest import digest_array
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

    x = deterministic_tensor(
        (args.batch_size, args.sequence_length, args.dims),
        mx.float32,
        offset=0,
    )
    input_shape = list(x.shape)
    input_digest = digest_array(x)

    rnn = RNN(args.dims, hidden_size)
    assign_deterministic_parameters(rnn)

    def run_step():
        return rnn(x)

    reference_output_digest = digest_array(run_step())
    path_signature = "forward_only_eval_output"

    out = None
    for i in range(args.warmup):
        out = run_step()
        mx.eval(out)
        if (i + 1) == args.warmup or (i + 1) % warmup_every == 0:
            print(f"[python/rnn] warmup {i + 1}/{args.warmup}", flush=True)

    start = time.perf_counter()
    for i in range(args.iterations):
        out = run_step()
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
