import argparse
import json
import os
import random
import time

import mlx.core as mx
import mlx.optimizers as optim
from mlx import nn
from mlx.nn import losses
from mlx.nn.layers.dropout import Dropout
from mlx.nn.layers.embedding import Embedding
from mlx.nn.layers.linear import Linear
from mlx.nn.layers.normalization import LayerNorm
from mlx.nn.layers.transformer import MultiHeadAttention
from mlx.nn.layers.transformer import TransformerEncoderLayer
from mlx.nn.utils import value_and_grad


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=["cpu", "gpu"], required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--sequence-length", type=int, required=True)
    parser.add_argument("--dims", type=int, required=True)
    parser.add_argument("--num-heads", type=int, required=True)
    parser.add_argument("--num-layers", type=int, required=True)
    parser.add_argument("--warmup", type=int, required=True)
    parser.add_argument("--iterations", type=int, required=True)
    parser.add_argument("--dataset-path", required=True)
    return parser.parse_args()


def set_device(device_name):
    mx.set_default_device(mx.cpu if device_name == "cpu" else mx.gpu)


def load_dataset(path):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Karpathy GPT-2 fixture missing at {path}. "
            "Add benchmark/fixtures/karpathy.txt before running this benchmark."
        )

    with open(path, "r", encoding="utf-8") as handle:
        text = handle.read()

    chars = sorted(set(text))
    if len(chars) == 0:
        raise ValueError(f"Dataset at {path} is empty.")

    stoi = {char: idx for idx, char in enumerate(chars)}
    data = [stoi[char] for char in text]
    vocab_size = len(chars)
    split = len(data) * 9 // 10
    return data[:split], vocab_size


def main():
    args = parse_args()
    set_device(args.device)
    train_data, vocab_size = load_dataset(args.dataset_path)

    warmup_every = max(1, args.warmup // 5)
    iter_every = max(1, args.iterations // 5)

    class GPT2Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.tok_embedding = Embedding(vocab_size, args.dims)
            self.pos_embedding = Embedding(args.sequence_length, args.dims)
            self.dropout = Dropout(0.0)
            self.blocks = [
                TransformerEncoderLayer(
                    args.dims,
                    args.num_heads,
                    mlp_dims=args.dims * 4,
                    dropout=0.0,
                    norm_first=True,
                )
                for _ in range(args.num_layers)
            ]
            self.ln = LayerNorm(args.dims)
            self.lm_head = Linear(args.dims, vocab_size)
            self.causal_mask = MultiHeadAttention.create_additive_causal_mask(
                args.sequence_length, mx.float32
            )

        def __call__(self, idx):
            seq_len = idx.shape[1]
            positions = mx.arange(0, seq_len, dtype=mx.int32)
            x = self.tok_embedding(idx) + self.pos_embedding(positions)
            x = self.dropout(x)
            for block in self.blocks:
                x = block(x, self.causal_mask)
            x = self.ln(x)
            return self.lm_head(x)

    def get_batch():
        max_start = len(train_data) - args.sequence_length - 1
        if max_start <= 0:
            raise ValueError(
                f"Sequence length {args.sequence_length} is too large for dataset size {len(train_data)}."
            )
        starts = [random.randrange(max_start) for _ in range(args.batch_size)]
        x = [train_data[start : start + args.sequence_length] for start in starts]
        y = [train_data[start + 1 : start + args.sequence_length + 1] for start in starts]
        return mx.array(x, mx.int32), mx.array(y, mx.int32)

    model = GPT2Model()

    def loss_fn(x, y):
        logits = model(x)
        batch_size, seq_len, local_vocab_size = logits.shape
        reshaped_logits = mx.reshape(
            logits[:, : (seq_len - 1), :],
            (batch_size * (seq_len - 1), local_vocab_size),
        )
        targets = mx.reshape(y[:, 1:seq_len], (batch_size * (seq_len - 1),))
        return losses.cross_entropy(reshaped_logits, targets, reduction="mean")

    step = value_and_grad(model, loss_fn)
    optimizer = optim.AdamW(learning_rate=1e-3)

    sample_input, _ = get_batch()
    out = model(sample_input)

    for i in range(args.warmup):
        x, y = get_batch()
        loss, grads = step(x, y)
        optimizer.update(model, grads)
        mx.eval(loss)
        if (i + 1) == args.warmup or (i + 1) % warmup_every == 0:
            print(f"[python/karpathy_gpt2] warmup {i + 1}/{args.warmup}", flush=True)

    start = time.perf_counter()
    for i in range(args.iterations):
        x, y = get_batch()
        loss, grads = step(x, y)
        optimizer.update(model, grads)
        mx.eval(loss)
        if (i + 1) == args.iterations or (i + 1) % iter_every == 0:
            print(f"[python/karpathy_gpt2] iter {i + 1}/{args.iterations}", flush=True)
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
