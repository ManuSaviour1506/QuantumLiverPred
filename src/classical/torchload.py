"""Load a PyTorch checkpoint (.pth) into NumPy arrays WITHOUT PyTorch.

Supports the legacy (non-zip) serialization used by the timm GitHub release weights and the newer zip format.
Only state_dict-like content (nested dicts of tensors and plain Python values) is supported; any other class is
refused, so loading a checkpoint cannot execute arbitrary code.
"""
from __future__ import annotations

import io
import pickle
import struct
import zipfile
from collections import OrderedDict

import numpy as np

_DT = {"FloatStorage": np.float32, "DoubleStorage": np.float64, "HalfStorage": np.float16, "LongStorage": np.int64,
       "IntStorage": np.int32, "ShortStorage": np.int16, "CharStorage": np.int8, "ByteStorage": np.uint8,
       "BoolStorage": np.bool_}


class _StorageType:
    def __init__(self, name):
        if name not in _DT:
            raise pickle.UnpicklingError(f"unsupported storage {name}")
        self.dtype = np.dtype(_DT[name])


class _Lazy:
    """A tensor whose storage bytes are not read yet."""
    __slots__ = ("key", "offset", "size", "stride")

    def __init__(self, key, offset, size, stride):
        self.key, self.offset, self.size, self.stride = key, offset, tuple(size), tuple(stride)


def _rebuild_tensor(storage, offset, size, stride, *args):
    return _Lazy(storage, offset, size, stride)


def _rebuild_param(data, *args):
    return data


class _Unpickler(pickle.Unpickler):
    def __init__(self, f, on_storage):
        super().__init__(f)
        self.on_storage = on_storage

    def find_class(self, module, name):
        if module == "collections" and name == "OrderedDict":
            return OrderedDict
        if module.startswith("torch"):
            if name in ("_rebuild_tensor_v2", "_rebuild_tensor"):
                return _rebuild_tensor
            if name == "_rebuild_parameter":
                return _rebuild_param
            if name.endswith("Storage"):
                return _StorageType(name)
        raise pickle.UnpicklingError(f"refusing to load {module}.{name}")

    def persistent_load(self, pid):
        if pid[0] != "storage":
            raise pickle.UnpicklingError(f"unknown persistent id {pid[0]}")
        stype, key = pid[1], str(pid[2])
        self.on_storage(key, stype)
        return key


def _materialize(obj, buffers, dtypes):
    if isinstance(obj, _Lazy):
        data, dt = buffers[obj.key], dtypes[obj.key]
        if len(obj.size) == 0:
            return np.array(data[obj.offset], dtype=dt)
        view = np.lib.stride_tricks.as_strided(data[obj.offset:], shape=obj.size,
                                               strides=tuple(s * dt.itemsize for s in obj.stride))
        return np.array(view, dtype=np.float32 if dt == np.float16 else dt, copy=True)
    if isinstance(obj, dict):
        return OrderedDict((k, _materialize(v, buffers, dtypes)) for k, v in obj.items())
    if isinstance(obj, (list, tuple)):
        return type(obj)(_materialize(v, buffers, dtypes) for v in obj)
    return obj


def load_state_dict(path: str) -> "OrderedDict[str, np.ndarray]":
    with open(path, "rb") as fh:
        raw = fh.read()
    dtypes: dict[str, np.dtype] = {}

    def on_storage(key, stype):
        dtypes[key] = stype.dtype

    buffers: dict[str, np.ndarray] = {}
    if raw[:2] == b"PK":  # zip format (torch >= 1.6)
        z = zipfile.ZipFile(io.BytesIO(raw))
        pkl = next(n for n in z.namelist() if n.endswith("data.pkl"))
        root = pkl[: -len("data.pkl")]
        obj = _Unpickler(io.BytesIO(z.read(pkl)), on_storage).load()
        for k, dt in dtypes.items():
            buffers[k] = np.frombuffer(z.read(f"{root}data/{k}"), dtype=dt)
    else:  # legacy format: magic, protocol, sys_info, object, storage keys, raw storages
        f = io.BytesIO(raw)
        for _ in range(3):
            pickle.load(f)
        obj = _Unpickler(f, on_storage).load()
        for k in pickle.load(f):
            k = str(k)
            n = struct.unpack("<q", f.read(8))[0]
            dt = dtypes[k]
            buffers[k] = np.frombuffer(f.read(n * dt.itemsize), dtype=dt)
    out = _materialize(obj, buffers, dtypes)
    if isinstance(out, dict) and "state_dict" in out and isinstance(out["state_dict"], dict):
        out = out["state_dict"]
    return out
