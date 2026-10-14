"""ctypes binding for libinfer.so (tracker/native). ctypes.CDLL releases the GIL during calls,
so inference overlaps with capture/web threads."""
import ctypes
import json
import os

import numpy as np

_LIB_PATH = os.environ.get("LIBINFER", os.path.join(os.path.dirname(__file__), "..", "lib", "libinfer.so"))
_lib = None
_DT = {"float32": np.float32, "float16": np.float16, "int8": np.int8, "uint8": np.uint8,
       "int32": np.int32, "int64": np.int64, "bool": np.bool_}


def lib():
    global _lib
    if _lib is None:
        _lib = ctypes.CDLL(_LIB_PATH)
        _lib.infer_load.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_void_p)]
        _lib.infer_run.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_void_p)]
        _lib.infer_info.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int]
        _lib.infer_dump.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        _lib.infer_free.argtypes = [ctypes.c_void_p]
    return _lib


class Model:
    def __init__(self, path, threads=0, etdump=False):
        self.path = path
        self.h = ctypes.c_void_p()
        rc = lib().infer_load(path.encode(), threads, int(etdump), ctypes.byref(self.h))
        if rc:
            raise RuntimeError(f"infer_load({path}) failed rc={rc}")
        buf = ctypes.create_string_buffer(16384)
        lib().infer_info(self.h, buf, len(buf))
        self.info = json.loads(buf.value.decode())
        self.in_specs = [(tuple(t["shape"]), _DT[t["dtype"]]) for t in self.info["inputs"]]
        self.outs = [np.empty(t["shape"], _DT[t["dtype"]]) for t in self.info["outputs"]]
        self._out_ptrs = (ctypes.c_void_p * len(self.outs))(*[o.ctypes.data for o in self.outs])

    def run(self, *inputs):
        """Returns the internal output buffers (overwritten by the next run; copy to keep)."""
        arrs = []
        for x, (shape, dt) in zip(inputs, self.in_specs):
            a = np.ascontiguousarray(x, dtype=dt)
            if a.shape != shape:
                a = a.reshape(shape)
            arrs.append(a)
        in_ptrs = (ctypes.c_void_p * len(arrs))(*[a.ctypes.data for a in arrs])
        rc = lib().infer_run(self.h, in_ptrs, self._out_ptrs)
        if rc:
            raise RuntimeError(f"infer_run({self.path}) failed rc={rc}")
        return self.outs

    def dump(self, path):
        return lib().infer_dump(self.h, path.encode())

    def close(self):
        if self.h:
            lib().infer_free(self.h)
            self.h = ctypes.c_void_p()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
