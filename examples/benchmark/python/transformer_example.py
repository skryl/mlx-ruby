import argparse
import json
import time

import mlx.core as mx
from benchmark_digest import INPUT_BASE_OFFSET
from benchmark_digest import assign_deterministic_parameters
from benchmark_digest import deterministic_tensor
from benchmark_digest import digest_array
from benchmark_digest import digest_value
from mlx.nn.layers.transformer import MultiHeadAttention
from mlx.nn.layers.transformer import Transformer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=["cpu", "gpu"], required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--source-sequence-length", type=int, required=True)
    parser.add_argument("--target-sequence-length", type=int, required=True)
    parser.add_argument("--dims", type=int, required=True)
    parser.add_argument("--num-heads", type=int, required=True)
    parser.add_argument("--num-layers", type=int, required=True)
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

    src = deterministic_tensor(
        (args.batch_size, args.source_sequence_length, args.dims),
        mx.float32,
        offset=0,
    )
    tgt = deterministic_tensor(
        (args.batch_size, args.target_sequence_length, args.dims),
        mx.float32,
        offset=INPUT_BASE_OFFSET,
    )
    src_mask = MultiHeadAttention.create_additive_causal_mask(
        args.source_sequence_length, mx.float32
    )
    tgt_mask = MultiHeadAttention.create_additive_causal_mask(
        args.target_sequence_length, mx.float32
    )
    input_digest = digest_value([src, tgt, src_mask, tgt_mask])

    model = Transformer(
        dims=args.dims,
        num_heads=args.num_heads,
        num_encoder_layers=args.num_layers,
        num_decoder_layers=args.num_layers,
        mlp_dims=args.dims * 4,
        dropout=0.0,
    )
    assign_deterministic_parameters(model)

    def run_step():
        return model(src, tgt, src_mask, tgt_mask, None)

    reference_output_digest = digest_array(run_step())
    path_signature = "forward_only_eval_output"

    out = None
    for i in range(args.warmup):
        out = run_step()
        mx.eval(out)
        if (i + 1) == args.warmup or (i + 1) % warmup_every == 0:
            print(f"[python/transformer] warmup {i + 1}/{args.warmup}", flush=True)

    start = time.perf_counter()
    for i in range(args.iterations):
        out = run_step()
        mx.eval(out)
        if (i + 1) == args.iterations or (i + 1) % iter_every == 0:
            print(f"[python/transformer] iter {i + 1}/{args.iterations}", flush=True)
    elapsed = time.perf_counter() - start

    print(
        json.dumps(
            {
                "average_ms": (elapsed / args.iterations) * 1000.0,
                "iterations": args.iterations,
                "warmup": args.warmup,
                "output_shape": list(out.shape),
                "input_digest": input_digest,
                "reference_output_digest": reference_output_digest,
                "path_signature": path_signature,
            }
        )
    )


if __name__ == "__main__":
    main()
