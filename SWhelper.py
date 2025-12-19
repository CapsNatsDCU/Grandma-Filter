"""
SWhelper.py

Grandma Filter helper script (STRICT GATEKEEPER).

Locked decisions:
1) faster-whisper is the primary/default transcription backend.
   Word-level timing is preferred via faster-whisper (when available).
   Additional word-timing backends may be added later, but are OPTIONAL.
2) Strict gatekeeper: missing *required* items => non-zero exit.
3) Must be run from repo root (main.py, processText.py, whisperCalls.py, FFMcalls.py).

Usage:
  python SWhelper.py check
  python SWhelper.py check --json
  python SWhelper.py check --install
  python SWhelper.py cleanup

Exit codes:
  0 = all required checks passed
  2 = missing required dependency / contract failure
  3 = error during checks
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import importlib.metadata
import platform
import shutil
import subprocess
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ---------- Result types ----------

@dataclass
class CheckItem:
    name: str
    ok: bool
    details: str = ""
    recommendation: str = ""


@dataclass
class CheckReport:
    required_ok: bool
    items: List[CheckItem]
    python: Dict[str, Any]
    pip: Dict[str, Any]

    def to_json(self) -> str:
        return json.dumps(
            {
                "required_ok": self.required_ok,
                "items": [asdict(i) for i in self.items],
                "python": self.python,
                "pip": self.pip,
            },
            indent=2,
        )


# ---------- Small helpers ----------

def _run(cmd, timeout_s: int | None = 15):
    """Run a command and return (rc, combined_output). Never hang."""
    try:
        p = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
            timeout=timeout_s,
        )
        return p.returncode, (p.stdout or "").strip()
    except subprocess.TimeoutExpired:
        return 124, f"TIMEOUT after {timeout_s}s: {' '.join(cmd)}"
    except Exception as e:
        return 3, f"{type(e).__name__}: {e}"


def _is_venv() -> bool:
    return bool(os.environ.get("VIRTUAL_ENV")) or (
        getattr(sys, "base_prefix", sys.prefix) != sys.prefix
    )


def _python_identity() -> Dict[str, Any]:
    return {
        "executable": sys.executable,
        "version": sys.version.split()[0],
        "prefix": sys.prefix,
        "base_prefix": getattr(sys, "base_prefix", sys.prefix),
        "venv": _is_venv(),
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "mac_ver": platform.mac_ver()[0],
        },
    }


def _pip_identity() -> Dict[str, Any]:
    pip_path = shutil.which("pip")
    # Never let pip identity wedge the gatekeeper.
    rc, out = _run([sys.executable, "-m", "pip", "--version"], timeout_s=8)
    return {
        "pip_on_path": pip_path,
        "pip_module": out if rc == 0 else None,
        "pip_error": out if rc != 0 else None,
    }


def _brew_python_warning(pyexe: str) -> Optional[str]:
    p = pyexe.replace("\\", "/")
    if p.startswith("/opt/homebrew/") or p.startswith("/usr/local/"):
        return (
            "Your Python looks like Homebrew Python. "
            "Prefer a project venv so installs don’t pollute system Python."
        )
    return None


# ---------- Checks ----------

# Repo-root contract (STRICT)
REPO_ROOT_REQUIRED_FILES = [
    "main.py",
    "processText.py",
    "whisperCalls.py",
    "FFMcalls.py",
]

# Required Python modules (STRICT)
REQUIRED_MODULES = [
    "torch",
    "faster_whisper",
]

OPTIONAL_MODULES = [
    "whisper",  # legacy
    "whisper_timestamped",  # optional word-level timestamps backend
    "stable_ts",  # optional word-level timestamps backend
]


def _check_repo_root_contract() -> CheckItem:
    root = Path.cwd()
    missing = [name for name in REPO_ROOT_REQUIRED_FILES if not (root / name).exists()]
    if missing:
        return CheckItem(
            name="repo:root_contract",
            ok=False,
            details=f"Not at repo root (missing: {', '.join(missing)})",
            recommendation=(
                "cd into the Grandma Filter repo root (the folder containing "
                "main.py, processText.py, whisperCalls.py, FFMcalls.py) and re-run."
            ),
        )
    return CheckItem(name="repo:root_contract", ok=True, details="Repo root files detected.")


def _ffmpeg_reco() -> str:
    sysname = platform.system()
    if sysname == "Darwin":
        return "Install ffmpeg via Homebrew: `brew install ffmpeg`"
    if sysname == "Linux":
        return "Install ffmpeg via apt/yum (example): `sudo apt-get update && sudo apt-get install ffmpeg`"
    if sysname == "Windows":
        return "Install ffmpeg and add it to PATH (e.g., via winget or a static build)."
    return "Install ffmpeg and ensure it’s on PATH."


def _check_executable(name: str, version_args: List[str]) -> CheckItem:
    path = shutil.which(name)
    if not path:
        return CheckItem(
            name=name,
            ok=False,
            details=f"{name} not found on PATH.",
            recommendation=_ffmpeg_reco() if name == "ffmpeg" else "Install it and ensure it’s on PATH.",
        )

    rc, out = _run([name] + version_args)
    if rc == 0:
        first_line = out.splitlines()[0] if out else ""
        return CheckItem(name=name, ok=True, details=f"{path} | {first_line}".strip())

    return CheckItem(
        name=name,
        ok=False,
        details=f"Found at {path}, but running it failed: {out}",
        recommendation="Try reinstalling it and/or check permissions.",
    )


def _check_module(mod: str, required: bool) -> CheckItem:
    # Fast path: “is it installed?”
    pkg_name = mod.replace("_", "-")  # common convention
    installed = False
    try:
        importlib.metadata.version(pkg_name)
        installed = True
    except Exception:
        try:
            importlib.metadata.version(mod)
            installed = True
        except Exception:
            installed = False

    if not installed:
        reco = (
            f"{sys.executable} -m pip install -U {mod}"
            if required
            else f"Optional: {sys.executable} -m pip install -U {mod}"
        )
        return CheckItem(
            name=f"python:{mod}",
            ok=False,
            details="not installed",
            recommendation=reco,
        )

    # Don’t hang: skip expensive imports for faster_whisper
    if mod == "faster_whisper":
        return CheckItem(
            name="python:faster_whisper",
            ok=True,
            details="installed (import check skipped to avoid long transformers scan)",
            recommendation="If runtime still hangs, switch to Python 3.12 or pin dependency versions.",
        )

    # For everything else, keep the subprocess import check
    try:
        p = subprocess.run(
            [sys.executable, "-c", f"import {mod}"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=25,
            check=False,
        )
        if p.returncode == 0:
            return CheckItem(name=f"python:{mod}", ok=True, details="import ok")

        out = (p.stdout or "").strip()
        return CheckItem(
            name=f"python:{mod}",
            ok=False,
            details=f"import failed (rc={p.returncode}): {out[:4000]}",
            recommendation=(
                f"{sys.executable} -m pip install -U {mod}"
                if required
                else f"Optional: {sys.executable} -m pip install -U {mod}"
            ),
        )
    except subprocess.TimeoutExpired:
        return CheckItem(
            name=f"python:{mod}",
            ok=False,
            details=f"import timed out (>{25}s): {mod}",
            recommendation="This usually means a broken/unsupported build for your Python version.",
        )



def _check_torch_accel_strict() -> CheckItem:
    """Strict check: on Apple Silicon, require MPS.

    IMPORTANT: do this in a subprocess with a timeout so we never hang during torch import.
    """

    code = r"""
import json, platform
try:
    import torch
    sysname = platform.system()
    machine = platform.machine()
    info = {
        'torch_version': getattr(torch, '__version__', 'unknown'),
        'system': sysname,
        'machine': machine,
    }

    if sysname == 'Darwin' and machine in ('arm64', 'aarch64'):
        mps = getattr(torch.backends, 'mps', None)
        info['mps_built'] = bool(mps is not None and mps.is_built())
        info['mps_available'] = bool(mps is not None and mps.is_available())
    else:
        cuda = getattr(torch, 'cuda', None)
        info['cuda_available'] = bool(cuda is not None and cuda.is_available())

    print(json.dumps({'ok': True, 'info': info}))
except Exception as e:
    print(json.dumps({'ok': False, 'error': f"{type(e).__name__}: {e}"}))
"""

    try:
        p = subprocess.run(
            [sys.executable, "-c", code],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=25,
            check=False,
        )

        raw = (p.stdout or "").strip()
        last_line = raw.splitlines()[-1] if raw else ""
        data = json.loads(last_line)

        if not data.get("ok"):
            return CheckItem(
                name="torch:accel",
                ok=False,
                details=f"torch probe failed: {data.get('error', 'unknown error')}",
                recommendation="Reinstall PyTorch for your platform (macOS + Apple Silicon): https://pytorch.org/get-started/locally/",
            )

        info = data.get("info", {})
        sysname = info.get("system")
        machine = info.get("machine")
        version = info.get("torch_version")

        details_parts = [f"torch={version}"]

        if sysname == "Darwin" and machine in ("arm64", "aarch64"):
            mps_built = bool(info.get("mps_built"))
            mps_avail = bool(info.get("mps_available"))
            details_parts.append(f"mps_built={mps_built}")
            details_parts.append(f"mps_available={mps_avail}")

            if mps_avail:
                return CheckItem(name="torch:accel", ok=True, details=" | ".join(details_parts))

            return CheckItem(
                name="torch:accel",
                ok=False,
                details=" | ".join(details_parts) + " | MPS not available",
                recommendation=(
                    "Install a PyTorch build with MPS and ensure macOS 12.3+. "
                    "See: https://pytorch.org/get-started/locally/"
                ),
            )

        cuda_avail = bool(info.get("cuda_available"))
        details_parts.append(f"cuda_available={cuda_avail}")
        return CheckItem(name="torch:accel", ok=True, details=" | ".join(details_parts))

    except subprocess.TimeoutExpired:
        return CheckItem(
            name="torch:accel",
            ok=False,
            details="torch probe timed out (>25s) — torch import may be wedged on this Python/build",
            recommendation="Strongly consider using Python 3.12 for Grandma Filter (torch is often slow/broken on newer Python builds).",
        )
    except Exception as e:
        return CheckItem(
            name="torch:accel",
            ok=False,
            details=f"torch probe errored: {type(e).__name__}: {e}",
            recommendation="Reinstall PyTorch for your platform: https://pytorch.org/get-started/locally/",
        )

def _check_ffmcalls_strict() -> List[CheckItem]:
    """Strictly verify FFMcalls resolves to local repo file and has make_mini_audio."""
    items: List[CheckItem] = []
    try:
        import importlib.util

        spec = importlib.util.find_spec("FFMcalls")
        if spec is None or spec.origin is None:
            return [
                CheckItem(
                    name="import:FFMcalls",
                    ok=False,
                    details="Could not locate module 'FFMcalls' on PYTHONPATH.",
                    recommendation="Run from repo root and ensure PYTHONPATH is not shadowing the local module.",
                )
            ]

        origin = Path(spec.origin).resolve()
        expected = (Path.cwd() / "FFMcalls.py").resolve()

        if origin != expected:
            items.append(
                CheckItem(
                    name="import:FFMcalls",
                    ok=False,
                    details=f"FFMcalls resolved to {origin}, expected {expected}",
                    recommendation="Fix working directory / PYTHONPATH so the local repo module is imported.",
                )
            )
        else:
            items.append(CheckItem(name="import:FFMcalls", ok=True, details=f"Local import ok: {origin}"))

        mod = importlib.import_module("FFMcalls")
        if hasattr(mod, "make_mini_audio"):
            items.append(CheckItem(name="import:FFMcalls.make_mini_audio", ok=True, details="Function exists."))
        else:
            items.append(
                CheckItem(
                    name="import:FFMcalls.make_mini_audio",
                    ok=False,
                    details="make_mini_audio is missing (this caused your previous AttributeError)",
                    recommendation="Add def make_mini_audio(...) to FFMcalls.py or update callers to the correct function name.",
                )
            )

    except Exception as e:
        items.append(
            CheckItem(
                name="import:FFMcalls",
                ok=False,
                details=f"Error while checking FFMcalls: {type(e).__name__}: {e}",
                recommendation="Fix import paths and ensure there isn’t a duplicate/shadowing module.",
            )
        )
    return items


def _install_module(module_name: str) -> None:
    print(f"[install] Installing {module_name} into {sys.executable} …")
    rc, out = _run([sys.executable, "-m", "pip", "install", "-U", module_name])
    if rc == 0:
        print(f"[install] ✅ Installed {module_name}")
    else:
        print(f"[install] ❌ Failed to install {module_name}: {out}")


def run_checks(install: bool = False) -> CheckReport:
    items: List[CheckItem] = []
    pyinfo = _python_identity()
    pipinfo = _pip_identity()

    # STRICT: must be in venv
    if not pyinfo["venv"]:
        items.append(
            CheckItem(
                name="env:venv",
                ok=False,
                details="Not running inside a virtual environment.",
                recommendation=(
                    "Create/use a venv:\n"
                    "  python3 -m venv .venv\n"
                    "  source .venv/bin/activate\n"
                    "  python -m pip install -U pip\n"
                    "Then re-run: python SWhelper.py check"
                ),
            )
        )
    else:
        items.append(CheckItem(name="env:venv", ok=True, details="Virtual environment active."))

    brew_warn = _brew_python_warning(pyinfo["executable"])
    if brew_warn:
        items.append(CheckItem(name="env:brew_python", ok=True, details=brew_warn))

    # STRICT: must be repo root
    items.append(_check_repo_root_contract())

    # Required executables
    items.append(_check_executable("ffmpeg", ["-version"]))

    # Required Python modules
    for m in REQUIRED_MODULES:
        ci = _check_module(m, required=True)
        items.append(ci)
        if install and (not ci.ok):
            _install_module(m)
            items.append(_check_module(m, required=True))

    # Optional modules
    for m in OPTIONAL_MODULES:
        items.append(_check_module(m, required=False))

    # Torch accel (STRICT on Apple Silicon)
    items.append(_check_torch_accel_strict())

    # Local import strictness for known failure point
    items.extend(_check_ffmcalls_strict())

    required_names = {
        "env:venv",
        "repo:root_contract",
        "ffmpeg",
        "torch:accel",
        "import:FFMcalls",
        "import:FFMcalls.make_mini_audio",
        *{f"python:{m}" for m in REQUIRED_MODULES},
    }

    required_ok = True
    for it in items:
        if it.name in required_names and not it.ok:
            required_ok = False

    return CheckReport(required_ok=required_ok, items=items, python=pyinfo, pip=pipinfo)


# ---------- Cleanup ----------

def clean_up(project_root: Optional[Path] = None) -> int:
    root = project_root or Path.cwd()
    removed: List[str] = []
    errors: List[str] = []

    mini = root / "mini_audio"
    if mini.exists() and mini.is_dir():
        for p in mini.glob("*"):
            try:
                if p.is_file():
                    p.unlink()
                    removed.append(str(p))
            except Exception as e:
                errors.append(f"{p}: {type(e).__name__}: {e}")

    for p in root.glob("output.*"):
        try:
            if p.is_file():
                p.unlink()
                removed.append(str(p))
        except Exception as e:
            errors.append(f"{p}: {type(e).__name__}: {e}")

    print(f"Removed {len(removed)} files.")
    for r in removed:
        print(f"  - {r}")

    if errors:
        print(f"Errors ({len(errors)}):")
        for err in errors:
            print(f"  - {err}")
        return 3

    return 0


# ---------- CLI ----------

def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Grandma Filter environment helper (strict)")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_check = sub.add_parser("check", help="Verify environment + dependencies")
    p_check.add_argument("--install", action="store_true", help="Attempt to pip install missing REQUIRED modules")
    p_check.add_argument("--json", action="store_true", help="Emit JSON report (app-friendly)")

    sub.add_parser("cleanup", help="Remove temporary files (mini_audio, output.*)")

    args = parser.parse_args(argv)

    if args.cmd == "cleanup":
        return clean_up()

    report = run_checks(install=bool(args.install))

    if args.json:
        print(report.to_json())
    else:
        print("--- Grandma Filter: STRICT Software Check ---")
        print(f"Python: {report.python['executable']} ({report.python['version']})")
        print(f"Venv:   {report.python['venv']}")
        if report.pip.get("pip_module"):
            print(f"Pip:    {report.pip['pip_module']}")
        print("")

        for it in report.items:
            status = "✅" if it.ok else "❌"
            print(f"{status} {it.name}: {it.details}")
            if (not it.ok) and it.recommendation:
                print(f"   -> {it.recommendation}")

        print("")
        print("✅ Required dependencies look good." if report.required_ok else "❌ Gatekeeper failed: fix the ❌ items above.")

    return 0 if report.required_ok else 2


if __name__ == "__main__":
    raise SystemExit(main())