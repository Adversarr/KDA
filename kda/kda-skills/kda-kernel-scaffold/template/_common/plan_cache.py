"""Bounded thread-owned nvmath plans with explicit operand lifetimes."""

from collections import OrderedDict
import threading

import torch


def operand_key(value):
    """Return scalar layout and alignment metadata without retaining a tensor."""
    if value is None:
        return None
    pointer = value.data_ptr()
    return (tuple(value.shape), tuple(value.stride()), value.storage_offset(), str(value.dtype), str(value.device),
            min(256, pointer & -pointer) if pointer else 0)


class PlanCache(threading.local):
    """Thread-local bounded plans; owners must clear at request teardown."""

    def __init__(self, capacity=8):
        if capacity < 1:
            raise ValueError("plan capacity must be positive")
        self.capacity = capacity
        self.plans = OrderedDict()

    def clear(self):
        """Free all owned plans, attempting every release even after an error."""
        error = None
        while self.plans:
            _, plan = self.plans.popitem(last=False)
            try:
                plan.free()
            except Exception as exc:
                error = error or exc
        if error is not None:
            raise error

    def execute(self, a, b, *, bias=None, epilog=None, options=None):
        """Execute a compatible plan and release operand references afterwards."""
        from nvmath.linalg.advanced import Matmul

        if not hasattr(Matmul, "release_operands"):
            raise ValueError("nvmath Matmul.release_operands is required; validated with nvmath-python 1.0")
        if torch.cuda.is_current_stream_capturing():
            raise ValueError("nvmath graph capture is outside the accelerated contract")
        stream = torch.cuda.current_stream(a.device)
        options = dict(options or {})
        key = (operand_key(a), operand_key(b), operand_key(bias), epilog,
               stream.cuda_stream, tuple(sorted((k, repr(v)) for k, v in options.items())))
        plan = self.plans.pop(key, None)
        inputs = {"bias": bias} if bias is not None else None
        if plan is None:
            if len(self.plans) >= self.capacity:
                _, oldest = self.plans.popitem(last=False)
                oldest.free()
            plan = Matmul(a, b, options=options, stream=stream)
            try:
                plan.plan(epilog=epilog, epilog_inputs=inputs)
            except BaseException:
                try:
                    plan.free()
                except Exception:
                    pass  # Preserve the original planning exception.
                raise
        self.plans[key] = plan
        try:
            plan.reset_operands(a=a, b=b, epilog_inputs=inputs, stream=stream)
            for value in (a, b, bias):
                if value is not None:
                    value.record_stream(stream)
            result = plan.execute(stream=stream)
        except BaseException:
            self.plans.pop(key, None)
            try:
                plan.free()
            except Exception:
                pass  # Preserve the execute/reset exception.
            raise
        try:
            plan.release_operands()
        except BaseException:
            self.plans.pop(key, None)
            try:
                plan.free()
            except Exception:
                pass
            raise
        return result
