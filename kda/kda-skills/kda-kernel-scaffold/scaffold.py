#!/usr/bin/env python3
"""Render a new kernel package from the KDA template into a user's repo.

    python scaffold.py --op fused_residual_rmsnorm --dest /path/to/repo/kda_kernels
    python scaffold.py --op foo --dest ... --kernel-backend triton      # or nvmath (GEMM + epilogue via cuBLASLt), tilelang (experimental)
    python scaffold.py --op foo --dest ... --force        # re-render an existing op dir
    python scaffold.py --dest ... --sync-common           # refresh _common only

``template/_common`` is copied once per destination and never overwritten unless
``--sync-common`` is passed (each op's SPEC records the version it was generated against).
``template/op`` is copied with ``{{op}}``, ``{{Op}}``, ``{{common_version}}``,
``{{kernel_backend}}`` and ``{{date}}`` substituted in file contents and names. Only the
kernel directory of the chosen backend (``_triton/``, ``_nvmath/`` or ``_tilelang/``) is rendered: one
kernel backend per package. Deterministic: no LLM is involved.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import re
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
TEMPLATE = HERE / "template"
OP_NAME_RE = re.compile(r"^[a-z][a-z0-9_]*$")
IGNORE = shutil.ignore_patterns("__pycache__", "*.pyc")
KERNEL_BACKENDS = ("triton", "nvmath", "tilelang")


def _camel(op: str) -> str:
    return "".join(part.capitalize() for part in op.split("_"))


def _template_common_version() -> str:
    return (TEMPLATE / "_common" / "VERSION").read_text().strip()


def sync_common(dest: Path, *, force: bool) -> str:
    """Install or refresh ``dest/_common``; returns the installed version."""
    src, dst = TEMPLATE / "_common", dest / "_common"
    template_version = _template_common_version()
    if not dst.exists():
        shutil.copytree(src, dst, ignore=IGNORE)
        print(f"installed _common {template_version} -> {dst}")
        return template_version
    installed = (dst / "VERSION").read_text().strip() if (dst / "VERSION").exists() else "unknown"
    if force:
        shutil.rmtree(dst)
        shutil.copytree(src, dst, ignore=IGNORE)
        print(f"updated _common {installed} -> {template_version}")
        return template_version
    if installed != template_version:
        print(
            f"note: {dst} is version {installed}, template is {template_version}; "
            "pass --sync-common to update (existing ops keep working)."
        )
    return installed


def render_op(op: str, dest: Path, *, common_version: str, force: bool, kernel_backend: str = "triton") -> Path:
    """Copy ``template/op`` to ``dest/<op>`` with placeholders substituted.

    Only ``_<kernel_backend>/`` is rendered among the kernel directories.
    """
    if kernel_backend not in KERNEL_BACKENDS:
        sys.exit(f"error: --kernel-backend must be one of {KERNEL_BACKENDS}, got {kernel_backend!r}")
    dst = dest / op
    if dst.exists():
        if not force:
            sys.exit(f"error: {dst} already exists; pass --force to re-render it")
        shutil.rmtree(dst)
    subst = {
        "{{op}}": op,
        "{{Op}}": _camel(op),
        "{{common_version}}": common_version,
        "{{kernel_backend}}": kernel_backend,
        "{{date}}": _dt.date.today().isoformat(),
    }
    skipped_dirs = {f"_{b}" for b in KERNEL_BACKENDS if b != kernel_backend}

    def render(text: str) -> str:
        for key, value in subst.items():
            text = text.replace(key, value)
        return text

    for src in sorted((TEMPLATE / "op").rglob("*")):
        if "__pycache__" in src.parts or src.suffix == ".pyc":
            continue
        if skipped_dirs & set(src.relative_to(TEMPLATE / "op").parts):
            continue
        rel = Path(render(str(src.relative_to(TEMPLATE / "op"))))
        target = dst / rel
        if src.is_dir():
            target.mkdir(parents=True, exist_ok=True)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(render(src.read_text()))
    print(f"rendered op {op!r} -> {dst}")
    return dst


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--op", help="snake_case op name; omit to only (re)install _common")
    p.add_argument("--dest", required=True, help="the kda_kernels package directory inside the user's repo")
    p.add_argument("--force", action="store_true", help="re-render an existing op directory")
    p.add_argument("--sync-common", action="store_true", help="overwrite dest/_common with the template version")
    p.add_argument(
        "--kernel-backend",
        default="triton",
        choices=KERNEL_BACKENDS,
        help="kernel backend of this package (SPEC kernel_backend; triton is the default, nvmath for a GEMM + epilogue that cuBLASLt expresses, tilelang is experimental); only its _<backend>/ directory is rendered",
    )
    args = p.parse_args(argv)

    dest = Path(args.dest).resolve()
    dest.mkdir(parents=True, exist_ok=True)
    init = dest / "__init__.py"
    if not init.exists():
        init.write_text('"""Kernels generated by KDA. Each subpackage is one fused op; `_common` is shared runtime."""\n')

    version = sync_common(dest, force=args.sync_common)
    if not args.op:
        return 0
    if not OP_NAME_RE.match(args.op):
        sys.exit(f"error: op name {args.op!r} must be snake_case ([a-z][a-z0-9_]*)")

    op_dir = render_op(args.op, dest, common_version=version, force=args.force, kernel_backend=args.kernel_backend)
    print(
        "\nnext steps:\n"
        f"  1. fill {op_dir / 'SPEC.md'} (front matter + prose) and {op_dir / '_eager.py'} (verbatim user code)\n"
        f"  2. write the signature and the op-specific contract checks in {op_dir / 'interface.py'},"
        f" the roofline in {op_dir / '_speed_of_light.py'}\n"
        f"  3. sanity check (verify + contract + infer), run from the repo root {dest.parent}:\n"
        f"     python -m {dest.name}.{args.op}._run_dev --verify --backend eager"
        f" --json {dest.name}/{args.op}/report.eager.json --md {dest.name}/{args.op}/report.eager.md\n"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
