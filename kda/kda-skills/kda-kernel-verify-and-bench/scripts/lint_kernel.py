#!/usr/bin/env python3
"""Mechanical lint of one KDA kernel package against the implementer conventions.

    python .agents/skills/kda-kernel-verify-and-bench/scripts/lint_kernel.py <op_dir> [--json out.json]
    python .agents/skills/kda-kernel-verify-and-bench/scripts/lint_kernel.py path/to/kernel.py ...

Reads ``<op_dir>/SPEC.md`` (kernel backend, saved-for-backward names, workloads) and
``STATUS.md`` (whether M5 is ticked), scans ``<op_dir>/_<kernel_backend>/*.py`` and prints a
markdown table of findings. Exit code 1 when any **hard** finding exists, else 0. Given
``.py`` files instead (reference snippets), the same code rules run without the SPEC-driven
ones (aux names are then the ``aux``-named pointers only).

Hard rules (fail the M3 verdict):

* ``copy``: ``.contiguous()`` or ``.reshape(...)`` on a launcher parameter, a saved tensor
  or an incoming gradient (a hidden copy the roofline does not count; pass strides instead).
* ``int64``: a Triton ``tl.program_id(...)`` used without ``.to(tl.int64)``.
* ``aux_guard``: a store to an aux tensor (SPEC ``saved_for_backward`` names that are not
  launcher inputs or user-visible outputs) outside ``if SAVE_AUX:`` / ``if save_aux:``.
* ``mma_operand``: an operand of ``tl.dot`` / ``T.gemm`` upcast to fp32 (``.to(tl.float32)``,
  ``.to(COMPUTE)``, an ``accum_dtype`` buffer, ``T.cast(..., float32)``): tensor cores need
  the storage dtype.
* ``autotune``: ``@triton.autotune`` / ``tilelang.autotune`` present although STATUS ticks M5.
* ``raw_launcher``: ``_register.py`` calls a launcher imported from ``_impl_*`` directly
  instead of the ``register_kernel`` op (``_fwd``/``_bwd``); torch.compile then traces into
  the launch and the compile probe fails.

Soft rules (notes for the verifier):

* ``view``: ``.view(...)`` on an input or gradient (never copies, but raises on layouts it
  cannot express; strides are the robust path).
* ``next_power_of_2``: a workload has a non-power-of-two last dim and no ``next_power_of_2`` in
  the Triton launchers.
* ``default_unused``: ``_configs.py`` defines ``_DEFAULT`` but ``select_config`` never returns it.
* ``wide_rows``: a workload's last dim exceeds the one-program-per-row register budget (32768
  elements, see compute-patterns.md) and the kernel files show no split-D geometry
  (``N_CHUNKS`` / ``SPLIT_D`` / ``n_chunks``).

Only stdlib + PyYAML; runs in the user's python.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

ONE_ROW_MAX_D = 32768  # widest row one program keeps in registers (rmsnorm reference, A800)
COPY_METHODS = {"contiguous": "hard", "reshape": "hard", "view": "soft"}
JIT_DECORATORS = ("triton.jit", "tilelang.jit", "jit")


@dataclass
class Finding:
    rule: str
    severity: str  # hard | soft
    where: str  # file:line
    detail: str


# ---------------------------------------------------------------- inputs


def parse_front_matter(text: str) -> dict:
    import yaml

    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}
    end = next((i for i in range(1, len(lines)) if lines[i].strip() == "---"), None)
    if end is None:
        return {}
    data = yaml.safe_load("\n".join(lines[1:end])) or {}
    return data if isinstance(data, dict) else {}


def m5_ticked(status_text: str) -> bool:
    return re.search(r"^- \[x\] M5\b", status_text, flags=re.MULTILINE | re.IGNORECASE) is not None


def _rows_and_d(shape: Sequence[int]) -> Tuple[int, int]:
    """(rows, last dim) of a shape list; (0, 0) when it is empty or still holds TODOs."""
    if not shape or not all(isinstance(s, int) for s in shape):
        return 0, 0
    rows = 1
    for s in shape[:-1]:
        rows *= s
    return rows, shape[-1]


def workload_facts(spec: dict) -> Dict[str, bool]:
    """Which soft rules the workload list activates."""
    non_pow2 = wide_rows = False
    for w in spec.get("workloads") or []:
        shapes = w.get("shapes") or {}
        if not shapes:
            continue
        _, d = _rows_and_d(next(iter(shapes.values())))
        if d > 0 and (d & (d - 1)) != 0:
            non_pow2 = True
        if d > ONE_ROW_MAX_D:
            wide_rows = True
    return {"non_pow2": non_pow2, "wide_rows": wide_rows}


# ---------------------------------------------------------------- ast helpers


def _decorator_names(fn: ast.FunctionDef) -> List[str]:
    names = []
    for d in fn.decorator_list:
        node = d.func if isinstance(d, ast.Call) else d
        names.append(ast.unparse(node))
    return names


def is_kernel(fn: ast.FunctionDef) -> bool:
    return any(name.endswith(JIT_DECORATORS) for name in _decorator_names(fn))


def has_autotune(fn: ast.FunctionDef) -> bool:
    return any("autotune" in name for name in _decorator_names(fn))


def root_name(node: ast.AST) -> Optional[str]:
    """``x`` for ``x``, ``x.float()``, ``x.T.contiguous``, ``x[..., 0]``, ``x_ptr + off``; None otherwise."""
    while True:
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            node = node.value
        elif isinstance(node, ast.Call):
            node = node.func
        elif isinstance(node, ast.Subscript):
            node = node.value
        elif isinstance(node, ast.BinOp):
            node = node.left  # pointer arithmetic: the base pointer is on the left by convention
        else:
            return None


def all_params(fn: ast.FunctionDef) -> Set[str]:
    """Parameters of ``fn`` and of every function nested in it (TileLang's ``@T.prim_func``)."""
    names: Set[str] = set()
    for node in ast.walk(fn):
        if isinstance(node, ast.FunctionDef):
            names |= {a.arg for a in node.args.args + node.args.kwonlyargs}
    return names


def input_names(fn: ast.FunctionDef) -> Set[str]:
    """Launcher parameters plus names unpacked from them or from ``ctx.saved_tensors``."""
    params = [a.arg for a in fn.args.args + fn.args.kwonlyargs]
    if fn.args.vararg:
        params.append(fn.args.vararg.arg)
    names = {p for p in params if p != "ctx"}
    for node in ast.walk(fn):
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target, value = node.targets[0], node.value
        src_root = root_name(value)
        from_saved = isinstance(value, ast.Attribute) and value.attr == "saved_tensors"
        if not (from_saved or src_root in names):
            continue
        if isinstance(target, (ast.Tuple, ast.List)):
            names |= {e.id for e in target.elts if isinstance(e, ast.Name)}
        elif isinstance(target, ast.Name) and isinstance(value, ast.Attribute) and from_saved:
            names.add(target.id)
    return names


def _inside_guard(path: List[ast.AST]) -> bool:
    for node in path:
        if isinstance(node, ast.If) and re.search(r"save_aux", ast.unparse(node.test), re.IGNORECASE):
            return True
    return False


def walk_with_path(node: ast.AST, path: Optional[List[ast.AST]] = None) -> Iterable[Tuple[ast.AST, List[ast.AST]]]:
    path = path or []
    yield node, path
    for child in ast.iter_child_nodes(node):
        yield from walk_with_path(child, path + [node])


# ---------------------------------------------------------------- rules


class Linter:
    """Lint a kernel package (``op_dir`` with SPEC.md) or a single snippet file (``files``).

    In file mode there is no SPEC: the backend is guessed from the imports, aux names are the
    ``*_ptr`` / buffer names containing ``aux`` and the STATUS/workload rules are off.
    """

    def __init__(self, op_dir: Path, files: Optional[Sequence[Path]] = None):
        self.op_dir = op_dir
        self.files = [Path(f).resolve() for f in files] if files else None
        self.findings: List[Finding] = []
        self.spec = parse_front_matter((op_dir / "SPEC.md").read_text()) if (op_dir / "SPEC.md").exists() else {}
        self.backend = str(self.spec.get("kernel_backend", "triton"))
        if self.files and not self.spec:
            texts = [f.read_text() for f in self.files]
            self.backend = "tilelang" if any("import tilelang" in t for t in texts) else ("nvmath" if any(re.search(r"^\s*(from|import) nvmath", t, re.M) for t in texts) else "triton")
        status = (op_dir / "STATUS.md").read_text() if (op_dir / "STATUS.md").exists() else ""
        self.frozen = m5_ticked(status)
        self.saved_names = {str(s.get("name")) for s in (self.spec.get("saved_for_backward") or []) if isinstance(s, dict)}
        self.kernel_dir = op_dir / f"_{self.backend}"
        self.sources: Dict[Path, str] = {}
        self.trees: Dict[Path, ast.Module] = {}

    def add(self, rule: str, severity: str, path: Path, line: int, detail: str) -> None:
        where = path.relative_to(self.op_dir) if path.is_relative_to(self.op_dir) else path.name
        self.findings.append(Finding(rule, severity, f"{where}:{line}", detail))

    def run(self) -> List[Finding]:
        if self.files is not None:
            paths = list(self.files)
        elif self.kernel_dir.is_dir():
            paths = sorted(self.kernel_dir.glob("*.py"))
        else:
            self.add("layout", "hard", self.op_dir / "SPEC.md", 1, f"kernel directory _{self.backend}/ not found")
            return self.findings
        for path in paths:
            text = path.read_text()
            self.sources[path] = text
            try:
                self.trees[path] = ast.parse(text)
            except SyntaxError as exc:
                self.add("syntax", "hard", path, exc.lineno or 1, str(exc.msg))
        aux_names = self._aux_names()
        for path, tree in self.trees.items():
            for fn in (n for n in tree.body if isinstance(n, ast.FunctionDef)):
                if self.frozen and has_autotune(fn):
                    self.add("autotune", "hard", path, fn.lineno, f"{fn.name} is autotuned but STATUS ticks M5")
                if is_kernel(fn):
                    self._check_kernel(path, fn, aux_names)
                elif not fn.name.endswith("_fake"):
                    self._check_launcher(path, fn)
        for path, tree in self.trees.items():
            if path.name == "_register.py":
                self._check_register(path, tree)
        self._soft_rules()
        return self.findings

    def _check_register(self, path: Path, tree: ast.Module) -> None:
        """Launchers imported from ``_impl_*`` may only be passed to ``register_kernel``.

        Calling one directly from ``_backward`` (instead of the registered ``_bwd`` op) is
        invisible to every numerics check and fails only the compile probe: inductor traces
        into the raw launch.
        """
        launchers: Set[str] = set()
        for node in tree.body:
            if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("_impl_"):
                launchers |= {(a.asname or a.name) for a in node.names if not a.name.endswith("_fake")}
        if not launchers:
            return
        registered: Set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and ast.unparse(node.func) == "register_kernel":
                registered |= {ast.unparse(a) for a in node.args}
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in launchers:
                self.add("raw_launcher", "hard", path, node.lineno,
                         f"`{node.func.id}(...)` called directly; call the registered op (`_fwd`/`_bwd`) so torch.compile sees an opaque op")
        for name in sorted(launchers - registered):
            self.add("raw_launcher", "soft", path, 1, f"launcher `{name}` imported but never passed to register_kernel")

    # -- which pointers are aux: saved names that are neither launcher inputs nor user-visible outputs
    def _aux_names(self) -> Set[str]:
        names = set(self.saved_names) | {"aux"}
        n_outputs = 1
        for text in self.sources.values():
            m = re.search(r"^N_OUTPUTS\s*=\s*(\d+)", text, flags=re.MULTILINE)
            if m:
                n_outputs = int(m.group(1))
        for tree in self.trees.values():
            for fn in (n for n in tree.body if isinstance(n, ast.FunctionDef)):
                if not fn.name.endswith("_fwd") or is_kernel(fn):
                    continue
                names -= {a.arg for a in fn.args.args + fn.args.kwonlyargs}
                for node in ast.walk(fn):
                    if isinstance(node, ast.Return) and isinstance(node.value, (ast.Tuple, ast.List)):
                        outs = {root_name(e) for e in node.value.elts[:n_outputs]}
                        names -= {o for o in outs if o}
        return names

    def _check_launcher(self, path: Path, fn: ast.FunctionDef) -> None:
        inputs = input_names(fn)
        for node in ast.walk(fn):
            if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
                continue
            method = node.func.attr
            if method not in COPY_METHODS:
                continue
            root = root_name(node.func.value)
            if root in inputs:
                severity = COPY_METHODS[method]
                what = "hidden copy" if severity == "hard" else "raises on layouts it cannot express; pass strides"
                rule = "copy" if severity == "hard" else "view"
                self.add(rule, severity, path, node.lineno, f"`{ast.unparse(node)}` on input `{root}` ({what})")

    def _check_kernel(self, path: Path, fn: ast.FunctionDef, aux_names: Set[str]) -> None:
        params = all_params(fn)
        aux_ptrs = {p for p in params if p in aux_names or p.removesuffix("_ptr") in aux_names}
        if self.backend in ("triton", "nvmath"):  # an nvmath package's adjoint is a Triton kernel
            self._check_program_id(path, fn)
        for node, stack in walk_with_path(fn):
            # aux stores outside the SAVE_AUX guard
            target: Optional[str] = None
            if isinstance(node, ast.Call) and ast.unparse(node.func) in ("tl.store", "T.store") and node.args:
                target = root_name(node.args[0])
            elif isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Subscript):
                target = root_name(node.targets[0])
            if target in aux_ptrs and not _inside_guard(stack):
                self.add("aux_guard", "hard", path, node.lineno, f"store to aux `{target}` outside `if SAVE_AUX:`")
            # MMA operands upcast to fp32
            if isinstance(node, ast.Call) and ast.unparse(node.func) in ("tl.dot", "T.gemm"):
                for operand in node.args[:2]:
                    reason = self._upcast_reason(fn, operand)
                    if reason:
                        self.add("mma_operand", "hard", path, node.lineno, f"`{ast.unparse(node.func)}` operand `{ast.unparse(operand)}` {reason}")

    def _check_program_id(self, path: Path, fn: ast.FunctionDef) -> None:
        src = self.sources[path].splitlines()
        body = "\n".join(src[fn.lineno - 1 : fn.end_lineno])
        for node in ast.walk(fn):
            if not (isinstance(node, ast.Call) and ast.unparse(node.func) == "tl.program_id"):
                continue
            line = src[node.lineno - 1]
            if ".to(tl.int64)" in line:
                continue
            m = re.match(r"\s*(\w+)\s*=\s*tl\.program_id", line)
            if m and re.search(rf"\b{m.group(1)}\.to\(tl\.int64\)", body):
                continue
            self.add("int64", "hard", path, node.lineno, f"`{line.strip()}` without `.to(tl.int64)` (int32 offsets overflow past 2^31 elements)")

    def _upcast_reason(self, fn: ast.FunctionDef, operand: ast.AST) -> Optional[str]:
        text = ast.unparse(operand)
        if re.search(r"\.to\((tl\.float32|COMPUTE)\)", text) or re.search(r"T\.cast\(.*(float32|accum_dtype)", text):
            return "is upcast inline"
        name = root_name(operand)
        if name is None:
            return None
        for node in ast.walk(fn):
            if isinstance(node, ast.Assign):
                targets = [root_name(t) for t in node.targets]
                if name not in targets:
                    continue
                value = ast.unparse(node.value)
                if re.search(r"\.to\((tl\.float32|COMPUTE)\)", value):
                    return f"is defined as `{value}`"
                if re.search(r"T\.alloc_\w+\(.*,\s*(accum_dtype|T\.float32|\"float32\"|'float32')\s*\)", value):
                    return f"is an fp32 buffer `{value}`"
                if isinstance(node.targets[0], ast.Subscript) and re.search(r"T\.cast\(.*(accum_dtype|float32)", value):
                    return f"is filled with `{value}`"
        return None

    def _soft_rules(self) -> None:
        facts = workload_facts(self.spec)
        all_text = "\n".join(self.sources.values())
        if facts["non_pow2"] and self.backend in ("triton", "nvmath") and "next_power_of_2" not in all_text:
            self.add("next_power_of_2", "soft", self.kernel_dir / "_impl_fwd.py", 1, "a workload has a non-power-of-two last dim but no launcher uses triton.next_power_of_2")
        if facts["wide_rows"] and not re.search(r"N_CHUNKS|SPLIT_D|n_chunks", all_text):
            self.add("wide_rows", "soft", self.kernel_dir / "_impl_fwd.py", 1, f"a workload has D > {ONE_ROW_MAX_D} (one program per row spills) but no split-D geometry is present")
        cfg = self.kernel_dir / "_configs.py"
        if cfg in self.sources:
            text = self.sources[cfg]
            if "_DEFAULT" in text and not re.search(r"return\s+dict\(_DEFAULT\)|return\s+_DEFAULT", text):
                self.add("default_unused", "soft", cfg, 1, "_DEFAULT is defined but select_config never returns it")


# ---------------------------------------------------------------- output


def render(findings: List[Finding], op_dir: Path, backend: str) -> str:
    hard = sum(f.severity == "hard" for f in findings)
    lines = [f"# lint: `{op_dir.name}` (`_{backend}/`) - {hard} hard, {len(findings) - hard} soft", ""]
    if not findings:
        lines.append("no findings")
        return "\n".join(lines)
    lines += ["| rule | severity | where | detail |", "|---|---|---|---|"]
    for f in sorted(findings, key=lambda f: (f.severity != "hard", f.where)):
        lines.append(f"| {f.rule} | {f.severity} | `{f.where}` | {f.detail} |")
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("paths", nargs="+", help="one kda_kernels/<op> directory, or one or more snippet .py files")
    p.add_argument("--json", help="also write the findings as JSON here")
    args = p.parse_args(argv)
    paths = [Path(a).resolve() for a in args.paths]
    if all(x.is_file() and x.suffix == ".py" for x in paths):
        op_dir = paths[0].parent
        linter = Linter(op_dir, files=paths)
    elif len(paths) == 1 and paths[0].is_dir():
        op_dir = paths[0]
        linter = Linter(op_dir)
    else:
        print("error: pass one package directory or only .py files", file=sys.stderr)
        return 2
    findings = linter.run()
    print(render(findings, op_dir, linter.backend))
    if args.json:
        Path(args.json).write_text(json.dumps([asdict(f) for f in findings], indent=2))
    return 1 if any(f.severity == "hard" for f in findings) else 0


if __name__ == "__main__":
    sys.exit(main())
