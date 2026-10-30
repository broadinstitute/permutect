"""
Counting calls that copy tensor values to the CPU, which make the CPU wait for the GPU when the tensors are on a GPU.
Without a GPU the waits can't be observed, but the calls can.
"""

import sys
from contextlib import contextmanager

import torch


@contextmanager
def permutect_calls(monkeypatch, *method_names: str):
    """
    Within the context, record the calls by permutect's own code to the given torch.Tensor methods, eg "item".  PyTorch
    also calls them, for example optimizers call .item() every step, but on step counters, which are on the CPU.
    """
    calls = []
    with monkeypatch.context() as patch:
        for name in method_names:

            def counted(tensor, *args, _name=name, _method=getattr(torch.Tensor, name), **kwargs):
                if sys._getframe(1).f_globals.get("__name__", "").startswith("permutect"):
                    calls.append(_name)
                return _method(tensor, *args, **kwargs)

            patch.setattr(torch.Tensor, name, counted)
        yield calls
