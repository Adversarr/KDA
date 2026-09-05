"""Peak-throughput lookup for speed-of-light (SOL) estimates.

Resolution order, per quantity:

* HBM bandwidth: derived exactly from ``torch.cuda.get_device_properties``
  (``memory_clock_rate x memory_bus_width x 2 / 8``); falls back to the SKU table.
* Peak FLOP/s: SKU table when the device name matches, because SKUs sharing a compute
  capability can differ wildly (H20 is sm_90 like H100 with ~15% of its tensor throughput);
  else ``SMs x SM clock x per-CC FLOP/SM/cycle``; else ``None`` and SOL is reported as n/a.

When ``flops_source`` is ``cc-derived`` for a SKU not in the table, the scaffold skill asks
the agent to confirm the numbers by web search and record them in ``SPEC.md``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Optional, Tuple

import torch

# Dense peak TFLOP/s (fp32 CUDA-core, bf16 tensor-core) and HBM GB/s per SKU. Keys are
# device-name substrings; longer keys are matched first.
_SKU_TABLE: Dict[str, Tuple[float, float, float]] = {
    "A800": (19.5, 312.0, 2039.0),
    "A100": (19.5, 312.0, 2039.0),
    "H20": (44.0, 148.0, 4000.0),
    "H800": (67.0, 989.0, 3350.0),
    "H100 80GB HBM3": (67.0, 989.0, 3350.0),
    "H100 PCIe": (51.0, 756.0, 2000.0),
    "H100 NVL": (60.0, 835.0, 3900.0),
    "H200": (67.0, 989.0, 4800.0),
    "B200": (80.0, 2250.0, 8000.0),
    "L20": (59.8, 119.5, 864.0),
    "L40S": (91.6, 362.0, 864.0),
}

# FLOP per SM per cycle: (fp32 CUDA-core, bf16 dense tensor-core), by compute capability.
_CC_FLOP_PER_SM_CLK: Dict[str, Tuple[int, int]] = {
    "8.0": (128, 2048),
    "8.6": (256, 512),
    "8.9": (256, 1024),
    "9.0": (256, 4096),
    "10.0": (256, 8192),
}


@dataclass(frozen=True)
class GpuInfo:
    name: str
    cc: str
    sms: int
    sm_clock_ghz: float
    l2_bytes: int
    total_mem_bytes: int
    hbm_gbps: Optional[float]
    fp32_tflops: Optional[float]
    bf16_tflops: Optional[float]
    flops_source: str  # "table" | "cc-derived" | "unknown"

    def as_dict(self) -> dict:
        return asdict(self)


def _sku_entry(name: str) -> Optional[Tuple[float, float, float]]:
    for key in sorted(_SKU_TABLE, key=len, reverse=True):
        if key in name:
            return _SKU_TABLE[key]
    return None


def get_gpu_info(device: Optional[torch.device] = None) -> GpuInfo:
    """Describe the GPU behind ``device`` (default: current CUDA device)."""
    props = torch.cuda.get_device_properties(device if device is not None else torch.cuda.current_device())
    cc = f"{props.major}.{props.minor}"
    sm_clock_ghz = props.clock_rate / 1e6  # kHz -> GHz
    sku = _sku_entry(props.name)

    mem_clock_khz = getattr(props, "memory_clock_rate", None)
    bus_bits = getattr(props, "memory_bus_width", None)
    if mem_clock_khz and bus_bits:
        hbm_gbps = mem_clock_khz * 1e3 * bus_bits * 2 / 8 / 1e9
    else:
        hbm_gbps = sku[2] if sku else None

    if sku:
        fp32, bf16, source = sku[0], sku[1], "table"
    elif cc in _CC_FLOP_PER_SM_CLK:
        f32_per_clk, bf16_per_clk = _CC_FLOP_PER_SM_CLK[cc]
        fp32 = props.multi_processor_count * sm_clock_ghz * f32_per_clk / 1e3
        bf16 = props.multi_processor_count * sm_clock_ghz * bf16_per_clk / 1e3
        source = "cc-derived"
    else:
        fp32, bf16, source = None, None, "unknown"

    return GpuInfo(
        name=props.name,
        cc=cc,
        sms=props.multi_processor_count,
        sm_clock_ghz=sm_clock_ghz,
        l2_bytes=getattr(props, "L2_cache_size", 0),
        total_mem_bytes=props.total_memory,
        hbm_gbps=hbm_gbps,
        fp32_tflops=fp32,
        bf16_tflops=bf16,
        flops_source=source,
    )


def compute_ms(flops: float, info: GpuInfo, *, roof: str = "fp32") -> Optional[float]:
    """Time at peak throughput of the unit doing the math (``fp32`` CUDA cores or ``bf16`` tensor cores)."""
    tflops = info.fp32_tflops if roof == "fp32" else info.bf16_tflops
    return None if tflops is None else flops / (tflops * 1e12) * 1e3


def sol_ms(bytes_moved: float, flops: float, info: GpuInfo, *, roof: str = "fp32") -> Optional[float]:
    """Theoretical speed-of-light time in ms: the slower of the datasheet memory and compute roofs.

    Returns ``None`` if either peak is unknown. For the gate, `_run_dev.py` pairs this with
    an achievable roof (`bench.copy_ms` at the same byte count) because HBM peak is only
    approached asymptotically.
    """
    c = compute_ms(flops, info, roof=roof)
    if info.hbm_gbps is None or c is None:
        return None
    return max(bytes_moved / (info.hbm_gbps * 1e9) * 1e3, c)


__all__ = ["GpuInfo", "get_gpu_info", "compute_ms", "sol_ms"]
