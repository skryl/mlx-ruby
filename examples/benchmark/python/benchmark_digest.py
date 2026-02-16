import hashlib
import numbers
import struct

import mlx.core as mx

INPUT_BASE_OFFSET = 1_000_000
BATCH_STRIDE = 9973
BATCH_OFFSET = 7919


def digest_value(value):
    return hashlib.sha256(_encode_value(value)).hexdigest()


def digest_array(array_like):
    return digest_value(array_like.tolist())


def deterministic_tensor(shape, dtype, offset=0):
    shape_tuple = tuple(int(dim) for dim in shape)
    flat_count = 1
    for dim in shape_tuple:
        flat_count *= dim

    dtype_name = str(dtype)
    if "int" in dtype_name:
        values = [int((offset + idx) % 97) for idx in range(flat_count)]
        array = mx.array(values, dtype=mx.int32)
    else:
        values = [(((offset + idx) % 1024) - 512) / 257.0 for idx in range(flat_count)]
        array = mx.array(values, dtype=mx.float32)

    if array.dtype != dtype:
        array = array.astype(dtype)
    return mx.reshape(array, shape_tuple)


def assign_deterministic_parameters(module_or_modules):
    modules = module_or_modules
    if not isinstance(module_or_modules, (list, tuple)):
        modules = [module_or_modules]

    offset = 0
    for module in modules:
        params = module.parameters()
        deterministic_params, offset = _deterministic_tree(params, offset)
        module.update(deterministic_params)


def _encode_value(value):
    if isinstance(value, bool):
        return b"b" + (b"\x01" if value else b"\x00")
    if value is None:
        return b"n"
    if isinstance(value, numbers.Integral):
        return b"i" + struct.pack("<q", int(value))
    if isinstance(value, numbers.Real):
        return b"f" + struct.pack("<d", float(value))
    if isinstance(value, str):
        encoded = value.encode("utf-8")
        return b"s" + struct.pack("<I", len(encoded)) + encoded
    if isinstance(value, (list, tuple)):
        payload = bytearray()
        payload.extend(b"a")
        payload.extend(struct.pack("<I", len(value)))
        for item in value:
            payload.extend(_encode_value(item))
        return bytes(payload)
    if isinstance(value, dict):
        payload = bytearray()
        items = sorted(((str(k), v) for k, v in value.items()), key=lambda kv: kv[0])
        payload.extend(b"h")
        payload.extend(struct.pack("<I", len(items)))
        for key, child in items:
            payload.extend(_encode_value(key))
            payload.extend(_encode_value(child))
        return bytes(payload)
    if hasattr(value, "tolist"):
        return _encode_value(value.tolist())
    raise TypeError(f"Unsupported value for digest encoding: {type(value)!r}")


def _deterministic_tree(tree, offset):
    if isinstance(tree, dict):
        output = {}
        for key in sorted(tree.keys(), key=lambda item: str(item)):
            child, offset = _deterministic_tree(tree[key], offset)
            output[key] = child
        return output, offset
    if isinstance(tree, list):
        output = []
        for element in tree:
            child, offset = _deterministic_tree(element, offset)
            output.append(child)
        return output, offset
    if isinstance(tree, tuple):
        output = []
        for element in tree:
            child, offset = _deterministic_tree(element, offset)
            output.append(child)
        return tuple(output), offset
    if tree is None:
        return None, offset
    if hasattr(tree, "shape") and hasattr(tree, "dtype"):
        shape = tuple(int(dim) for dim in tree.shape)
        flat_count = 1
        for dim in shape:
            flat_count *= dim
        return deterministic_tensor(shape, tree.dtype, offset=offset), offset + flat_count
    raise TypeError(f"Unsupported parameter leaf for deterministic tree: {type(tree)!r}")
