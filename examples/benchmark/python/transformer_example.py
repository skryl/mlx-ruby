import argparse
import json
import time

import mlx.core as mx
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

    src = mx.random.uniform(
        low=-1.0,
        high=1.0,
        shape=(args.batch_size, args.source_sequence_length, args.dims),
        dtype=mx.float32,
    )
    tgt = mx.random.uniform(
        low=-1.0,
        high=1.0,
        shape=(args.batch_size, args.target_sequence_length, args.dims),
        dtype=mx.float32,
    )
    src_mask = MultiHeadAttention.create_additive_causal_mask(
        args.source_sequence_length, mx.float32
    )
    tgt_mask = MultiHeadAttention.create_additive_causal_mask(
        args.target_sequence_length, mx.float32
    )

    model = Transformer(
        dims=args.dims,
        num_heads=args.num_heads,
        num_encoder_layers=args.num_layers,
        num_decoder_layers=args.num_layers,
        mlp_dims=args.dims * 4,
        dropout=0.0,
    )

    out = None
    for i in range(args.warmup):
        out = model(src, tgt, src_mask, tgt_mask, None)
        mx.eval(out)
        if (i + 1) == args.warmup or (i + 1) % warmup_every == 0:
            print(f"[python/transformer] warmup {i + 1}/{args.warmup}", flush=True)

    start = time.perf_counter()
    for i in range(args.iterations):
        out = model(src, tgt, src_mask, tgt_mask, None)
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
            }
        )
    )


if __name__ == "__main__":
    main()
