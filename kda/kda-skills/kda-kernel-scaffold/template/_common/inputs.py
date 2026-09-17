"""Storage-preserving verification inputs and observable tensor contracts."""

from typing import Dict

import torch


class UnsupportedLayout(ValueError):
    """An input layout that cannot be faithfully reconstructed."""


def tensor_metadata(tensors: Dict[str, torch.Tensor]) -> dict:
    """Describe layouts and canonical alias groups without persisting addresses."""
    groups = {}
    result = {}
    for name, value in tensors.items():
        if value.layout != torch.strided or value.is_conj() or value.is_neg():
            raise UnsupportedLayout(f"{name}: unsupported layout or view bits")
        key = (str(value.device), value.untyped_storage()._cdata)
        group = groups.setdefault(key, len(groups))
        result[name] = dict(shape=list(value.shape), stride=list(value.stride()),
                            offset=value.storage_offset(), dtype=str(value.dtype),
                            device=str(value.device), storage_group=group)
    return result


def clone_inputs(tensors: Dict[str, torch.Tensor], needs_grad=None) -> Dict[str, torch.Tensor]:
    """Copy each backing storage once and reconstruct its original views.

    Args:
        tensors: Named strided tensors, possibly sharing storage.
        needs_grad: Optional predicate selecting differentiable input leaves.

    Returns:
        Independent storage with the same layouts and within-group aliases.
    """
    before = tensor_metadata(tensors)
    buffers = {}
    result = {}
    for name, value in tensors.items():
        key = before[name]["storage_group"]
        if key not in buffers:
            storage = value.untyped_storage()
            raw = torch.empty(0, dtype=torch.uint8, device=value.device)
            buffers[key] = raw.set_(storage, 0, (storage.nbytes(),), (1,)).clone()
        view = torch.empty(0, dtype=value.dtype, device=value.device).set_(
            buffers[key].untyped_storage(), value.storage_offset(), value.shape, value.stride())
        result[name] = view.requires_grad_(bool(needs_grad and needs_grad(name) and value.is_floating_point()))
    if tensor_metadata(result) != before:
        raise UnsupportedLayout("copy changed tensor metadata")
    return result


def check_workload(workload, tensors: Dict[str, torch.Tensor]) -> dict:
    """Check actual inputs against declared geometry and storage relationships."""
    actual = tensor_metadata(tensors)
    if set(actual) != set(workload.shapes):
        raise UnsupportedLayout("input names differ from workload")
    declared_groups = {}
    actual_groups = {}
    for name, shape in workload.shapes.items():
        row = actual[name]
        strides = workload.strides.get(name)
        if strides is None:
            strides, size = [], 1
            for dim in reversed(shape):
                strides.insert(0, size)
                size *= max(1, dim)
        group = workload.storage_groups.get(name, "input:" + name)
        declared = declared_groups.setdefault(group, len(declared_groups))
        observed = actual_groups.setdefault(row["storage_group"], len(actual_groups))
        if (row["shape"] != list(shape) or row["stride"] != list(strides)
                or row["offset"] != workload.storage_offsets.get(name, 0)
                or row["dtype"] != str(workload.dtype_of(name)) or declared != observed):
            raise UnsupportedLayout(f"{name}: actual layout differs from workload: {row}")
    return actual


def unchanged(before: Dict[str, torch.Tensor], after: Dict[str, torch.Tensor]) -> bool:
    """Check metadata and values, including unchanged NaNs, after a call."""
    if tensor_metadata(before) != tensor_metadata(after):
        return False
    return all(torch.equal(a, after[n]) or bool(torch.all(
        (a == after[n]) | (torch.isnan(a) & torch.isnan(after[n]))))
        for n, a in before.items())


def output_checks(outputs, reference, inputs, ref_inputs, contracts):
    """Check declared output stride, contiguity and storage aliases."""
    result = {}
    for index, (actual, expected) in enumerate(zip(outputs, reference)):
        name = f"out{index}"
        contract = contracts.get(name, {})
        passed = True
        if "stride" in contract:
            stride = expected.stride() if contract["stride"] == "reference" else contract["stride"]
            passed &= tuple(actual.stride()) == tuple(stride)
        if "storage_offset" in contract:
            offset = expected.storage_offset() if contract["storage_offset"] == "reference" else contract["storage_offset"]
            passed &= actual.storage_offset() == offset
        if contract.get("contiguous"):
            passed &= actual.is_contiguous()
        if "aliases" in contract:
            actual_all = dict(inputs, **{f"out{i}": v for i, v in enumerate(outputs) if i != index})
            ref_all = dict(ref_inputs, **{f"out{i}": v for i, v in enumerate(reference) if i != index})
            aliases = lambda value, values: sorted(k for k, v in values.items()
                if value.untyped_storage()._cdata == v.untyped_storage()._cdata)
            wanted = aliases(expected, ref_all) if contract["aliases"] == "reference" else sorted(contract["aliases"])
            passed &= aliases(actual, actual_all) == wanted
        if contract:
            result[name + "_metadata"] = {"passed": bool(passed), "contract": contract}
    return result
