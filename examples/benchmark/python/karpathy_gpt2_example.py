import argparse
import json
import os
import time

import mlx.core as mx
import mlx.optimizers as optim
from benchmark_digest import BATCH_OFFSET
from benchmark_digest import BATCH_STRIDE
from benchmark_digest import assign_deterministic_parameters
from benchmark_digest import digest_array
from benchmark_digest import digest_value
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

    with open(path, "rb") as handle:
        raw = list(handle.read())

    vocab = sorted(set(raw))
    if len(vocab) == 0:
        raise ValueError(f"Dataset at {path} is empty.")

    stoi = {value: idx for idx, value in enumerate(vocab)}
    data = [stoi[value] for value in raw]
    vocab_size = len(vocab)
    split = len(data) * 9 // 10
    return data[:split], vocab_size


def deterministic_starts(max_start, batch_size, step_index):
    base = step_index * BATCH_STRIDE
    return [((base + (idx * BATCH_OFFSET)) % max_start) for idx in range(batch_size)]


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

    def get_batch_for_step(step_index):
        max_start = len(train_data) - args.sequence_length - 1
        if max_start <= 0:
            raise ValueError(
                f"Sequence length {args.sequence_length} is too large for dataset size {len(train_data)}."
            )
        starts = deterministic_starts(max_start, args.batch_size, step_index)
        x = [train_data[start : start + args.sequence_length] for start in starts]
        y = [train_data[start + 1 : start + args.sequence_length + 1] for start in starts]
        return mx.array(x, mx.int32), mx.array(y, mx.int32)

    model = GPT2Model()
    assign_deterministic_parameters(model)

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

    sample_input, sample_target = get_batch_for_step(0)
    out = model(sample_input)
    input_digest = digest_value(
        {
            "train_data": train_data,
            "batch_size": args.batch_size,
            "sequence_length": args.sequence_length,
            "batch_stride": BATCH_STRIDE,
            "batch_offset": BATCH_OFFSET,
            "sample_input": sample_input,
            "sample_target": sample_target,
        }
    )
    reference_output_digest = digest_array(out)
    path_signature = "value_and_grad_update_eval_loss"

    def run_step(step_index):
        x, y = get_batch_for_step(step_index)
        loss, grads = step(x, y)
        optimizer.update(model, grads)
        return loss

    step_index = 0
    for i in range(args.warmup):
        loss = run_step(step_index)
        step_index += 1
        mx.eval(loss)
        if (i + 1) == args.warmup or (i + 1) % warmup_every == 0:
            print(f"[python/karpathy_gpt2] warmup {i + 1}/{args.warmup}", flush=True)

    start = time.perf_counter()
    for i in range(args.iterations):
        loss = run_step(step_index)
        step_index += 1
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
                "input_digest": input_digest,
                "reference_output_digest": reference_output_digest,
                "path_signature": path_signature,
            }
        )
    )


if __name__ == "__main__":
    main()
