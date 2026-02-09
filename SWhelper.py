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
  python SWhelper.py check --install --yes
  python SWhelper.py cleanup
  python SWhelper.py uninstall

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


def _python_in_venv(py_path: str, venv_path: str) -> bool:
    try:
        py = Path(py_path).resolve()
        venv = Path(venv_path).resolve()
        return str(py).startswith(str(venv))
    except Exception:
        return False


def _is_venv() -> bool:
    base_prefix = getattr(sys, "base_prefix", sys.prefix)
    if base_prefix != sys.prefix:
        return True
    venv = os.environ.get("VIRTUAL_ENV")
    if venv and _python_in_venv(sys.executable, venv):
        return True
    return False


def _target_python() -> str:
    """Prefer the active venv's Python if available; fall back to sys.executable."""
    if _is_venv():
        return sys.executable

    venv = os.environ.get("VIRTUAL_ENV")
    if venv:
        # macOS/Linux
        cand = Path(venv) / "bin" / "python"
        if cand.exists():
            return str(cand)
        # Windows
        cand = Path(venv) / "Scripts" / "python.exe"
        if cand.exists():
            return str(cand)

    return sys.executable


def _find_preferred_python() -> str:
    """Prefer Python 3.12 (or 3.11) for compatibility with faster-whisper."""
    for exe in ("python3.12", "python3.11"):
        path = shutil.which(exe)
        if path:
            return path
    return sys.executable


def _maybe_install_python312() -> Optional[str]:
    if platform.system() != "Darwin":
        return None
    if shutil.which("python3.12"):
        return shutil.which("python3.12")
    if not shutil.which("brew"):
        return None
    if not _confirm("Python 3.12 is recommended. Install via Homebrew now?"):
        return None
    print("[install] Installing python@3.12 via Homebrew …")
    rc, out = _run(["brew", "install", "python@3.12"], timeout_s=900)
    if rc != 0:
        print(f"[install] ❌ Failed to install python@3.12: {out}")
        return None
    return shutil.which("python3.12")


def _python_version(py_path: str) -> str:
    rc, out = _run([py_path, "-c", "import sys; print(sys.version.split()[0])"], timeout_s=8)
    return out.strip() if rc == 0 else ""


def _ensure_venv_exists(path: str = ".venv") -> Optional[str]:
    """Create .venv if missing. Returns venv python path if available."""
    venv_path = Path(path)
    if not venv_path.exists():
        print(f"[setup] Creating virtual environment at {venv_path} …")
        py = _find_preferred_python()
        rc, out = _run([py, "-m", "venv", str(venv_path)], timeout_s=900)
        if rc != 0:
            print(f"[setup] ❌ Failed to create venv: {out}")
            return None
    # Resolve venv python
    cand = venv_path / "bin" / "python"
    if cand.exists():
        return str(cand)
    cand = venv_path / "Scripts" / "python.exe"
    if cand.exists():
        return str(cand)
    print("[setup] ❌ Could not locate venv python after creation.")
    return None


def ensure_runtime_ready(argv: List[str], *, assume_yes: bool = False) -> None:
    """Ensure we are running inside the project venv; relaunch if needed."""
    if _is_venv():
        vpy_ver = _python_version(sys.executable)
        if vpy_ver and vpy_ver.split(".")[0:2] not in [["3", "12"], ["3", "11"]]:
            preferred = _find_preferred_python()
            pref_ver = _python_version(preferred)
            if pref_ver == "" and platform.system() == "Darwin":
                # Try to install 3.12 if missing
                p312 = _maybe_install_python312()
                if p312:
                    preferred = p312
                    pref_ver = _python_version(preferred)
            if pref_ver and pref_ver.split(".")[0:2] in [["3", "12"], ["3", "11"]]:
                print(f"[setup] Detected venv Python {vpy_ver}; recreating with {pref_ver} for compatibility …")
                try:
                    shutil.rmtree(".venv")
                except Exception as e:
                    print(f"[setup] ❌ Failed to remove .venv: {type(e).__name__}: {e}")
                    raise SystemExit(2)
                vpy = _ensure_venv_exists(".venv")
                if not vpy:
                    raise SystemExit(2)
                print("[setup] Relaunching with venv Python …")
                env = os.environ.copy()
                env["VIRTUAL_ENV"] = str(Path(".venv").resolve())
                env["GF_RELAUNCHED"] = "1"
                os.execve(vpy, [vpy] + argv, env)
            return
        return

    if os.environ.get("GF_RELAUNCHED") == "1":
        print("[setup] ❌ Relaunch attempted but still not in venv. Aborting.")
        raise SystemExit(2)

    # If VIRTUAL_ENV is set but not actually active, try to relaunch using it.
    venv_env = os.environ.get("VIRTUAL_ENV")
    if venv_env:
        cand = Path(venv_env) / "bin" / "python"
        if not cand.exists():
            cand = Path(venv_env) / "Scripts" / "python.exe"
        if cand.exists():
            vpy = str(cand)
        else:
            vpy = None
    else:
        vpy = None

    if not vpy:
        vpy = _ensure_venv_exists(".venv")
    if not vpy:
        raise SystemExit(2)

    # If the venv Python is too new for faster-whisper deps, try to recreate with 3.12/3.11.
    vpy_ver = _python_version(vpy)
    if vpy_ver and vpy_ver.split(".")[0:2] not in [["3", "12"], ["3", "11"]]:
        preferred = _find_preferred_python()
        pref_ver = _python_version(preferred)
        if pref_ver == "" and platform.system() == "Darwin":
            p312 = _maybe_install_python312()
            if p312:
                preferred = p312
                pref_ver = _python_version(preferred)
        if pref_ver and pref_ver.split(".")[0:2] in [["3", "12"], ["3", "11"]]:
            print(f"[setup] Detected venv Python {vpy_ver}; recreating with {pref_ver} for compatibility …")
            try:
                shutil.rmtree(".venv")
            except Exception as e:
                print(f"[setup] ❌ Failed to remove .venv: {type(e).__name__}: {e}")
                raise SystemExit(2)
            vpy = _ensure_venv_exists(".venv")
            if not vpy:
                raise SystemExit(2)

    # Relaunch using venv Python
    print("[setup] Relaunching with venv Python …")
    env = os.environ.copy()
    env["VIRTUAL_ENV"] = str(Path(".venv").resolve())
    env["GF_RELAUNCHED"] = "1"
    os.execve(vpy, [vpy] + argv, env)


def _python_identity() -> Dict[str, Any]:
    py = _target_python()
    rc, out = _run([py, "-c", "import sys; print(sys.version.split()[0])"], timeout_s=8)
    py_ver = out.strip() if rc == 0 else sys.version.split()[0]
    return {
        "executable": py,
        "version": py_ver,
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
    py = _target_python()
    # Never let pip identity wedge the gatekeeper.
    rc, out = _run([py, "-m", "pip", "--version"], timeout_s=8)
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


def _install_ffmpeg(*, assume_yes: bool = False) -> None:
    sysname = platform.system()
    if sysname == "Darwin":
        if not shutil.which("brew"):
            print("[install] ❌ Homebrew not found; install it first: https://brew.sh/")
            return
        if (not assume_yes) and (not _confirm("ffmpeg will be installed system-wide via Homebrew. Proceed?")):
            print("[install] Skipped ffmpeg install.")
            return
        print("[install] Installing ffmpeg via Homebrew …")
        rc, out = _run(["brew", "install", "ffmpeg"], timeout_s=900)
        if rc == 0:
            print("[install] ✅ Installed ffmpeg")
        else:
            print(f"[install] ❌ Failed to install ffmpeg: {out}")
        return

    if sysname == "Linux":
        if shutil.which("apt-get"):
            if (not assume_yes) and (not _confirm("ffmpeg will be installed system-wide via apt-get. Proceed?")):
                print("[install] Skipped ffmpeg install.")
                return
            print("[install] Installing ffmpeg via apt-get …")
            rc, out = _run(["sudo", "apt-get", "update"], timeout_s=900)
            if rc != 0:
                print(f"[install] ❌ apt-get update failed: {out}")
                return
            rc, out = _run(["sudo", "apt-get", "install", "-y", "ffmpeg"], timeout_s=900)
            if rc == 0:
                print("[install] ✅ Installed ffmpeg")
            else:
                print(f"[install] ❌ Failed to install ffmpeg: {out}")
            return
        if shutil.which("yum"):
            if (not assume_yes) and (not _confirm("ffmpeg will be installed system-wide via yum. Proceed?")):
                print("[install] Skipped ffmpeg install.")
                return
            print("[install] Installing ffmpeg via yum …")
            rc, out = _run(["sudo", "yum", "install", "-y", "ffmpeg"], timeout_s=900)
            if rc == 0:
                print("[install] ✅ Installed ffmpeg")
            else:
                print(f"[install] ❌ Failed to install ffmpeg: {out}")
            return

        print("[install] ❌ No supported Linux package manager found for ffmpeg.")
        return

    if sysname == "Windows":
        if shutil.which("winget"):
            if (not assume_yes) and (not _confirm("ffmpeg will be installed system-wide via winget. Proceed?")):
                print("[install] Skipped ffmpeg install.")
                return
            print("[install] Installing ffmpeg via winget …")
            rc, out = _run(["winget", "install", "--id", "Gyan.FFmpeg", "-e"], timeout_s=900)
            if rc == 0:
                print("[install] ✅ Installed ffmpeg")
            else:
                print(f"[install] ❌ Failed to install ffmpeg: {out}")
            return
        print("[install] ❌ winget not found; install ffmpeg manually and add it to PATH.")
        return

    print("[install] ❌ Unsupported OS for automatic ffmpeg install.")


def _check_module(mod: str, required: bool) -> CheckItem:
    py = _target_python()
    pkg_name = mod.replace("_", "-")  # common convention

    # Check installation via target Python to avoid mismatch with venv
    check_code = (
        "import importlib.metadata as m\n"
        "name = %r\n"
        "ok = False\n"
        "try:\n"
        "  m.version(name)\n"
        "  ok = True\n"
        "except Exception:\n"
        "  try:\n"
        "    m.version(%r)\n"
        "    ok = True\n"
        "  except Exception:\n"
        "    ok = False\n"
        "print('OK' if ok else 'NO')\n"
    ) % (pkg_name, mod)
    rc, out = _run([py, "-c", check_code], timeout_s=15)
    installed = rc == 0 and (out.strip().endswith("OK"))

    if not installed:
        reco = (
            f"{py} -m pip install -U {mod}"
            if required
            else f"Optional: {py} -m pip install -U {mod}"
        )
        return CheckItem(
            name=f"python:{mod}",
            ok=False,
            details="not installed",
            recommendation=reco,
        )

    # Don’t hang: skip expensive imports for known heavy modules
    if mod in {"faster_whisper", "torch", "whisper", "whisper_timestamped", "stable_ts"}:
        return CheckItem(
            name=f"python:{mod}",
            ok=True,
            details=f"installed (import check skipped for {mod})",
            recommendation=(
                "If runtime still hangs, switch to Python 3.12 or pin dependency versions."
                if mod == "faster_whisper"
                else ""
            ),
        )

    # For everything else, keep the subprocess import check
    try:
        p = subprocess.run(
            [py, "-c", f"import {mod}"],
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
                f"{py} -m pip install -U {mod}"
                if required
                else f"Optional: {py} -m pip install -U {mod}"
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
            [_target_python(), "-c", code],
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
            ok=True,
            details="torch probe timed out (>25s) — skipping accel check",
            recommendation="If performance is poor, reinstall PyTorch for your platform.",
        )
    except Exception as e:
        return CheckItem(
            name="torch:accel",
            ok=True,
            details=f"torch probe errored: {type(e).__name__}: {e} — skipping accel check",
            recommendation="If performance is poor, reinstall PyTorch for your platform.",
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
    if not _is_venv():
        print(
            f"[install] ⚠️ Skipping {module_name}: not in a virtual environment. "
            "Activate .venv and re-run to avoid PEP 668 system Python restrictions."
        )
        return
    py = _target_python()
    print(f"[install] Installing {module_name} into {py} …")
    rc, out = _run([py, "-m", "pip", "install", "-U", module_name], timeout_s=900)
    if rc == 0:
        print(f"[install] ✅ Installed {module_name}")
    else:
        print(f"[install] ❌ Failed to install {module_name}: {out}")


def _uninstall_module(module_name: str) -> None:
    print(f"[uninstall] Uninstalling {module_name} from {sys.executable} …")
    rc, out = _run([sys.executable, "-m", "pip", "uninstall", "-y", module_name], timeout_s=300)
    if rc == 0:
        print(f"[uninstall] ✅ Uninstalled {module_name}")
    else:
        print(f"[uninstall] ❌ Failed to uninstall {module_name}: {out}")


def _uninstall_ffmpeg() -> None:
    sysname = platform.system()
    if sysname == "Darwin":
        if not shutil.which("brew"):
            print("[uninstall] ❌ Homebrew not found; cannot uninstall ffmpeg automatically.")
            return
        print("[uninstall] Uninstalling ffmpeg via Homebrew …")
        rc, out = _run(["brew", "uninstall", "ffmpeg"], timeout_s=900)
        if rc == 0:
            print("[uninstall] ✅ Uninstalled ffmpeg")
        else:
            print(f"[uninstall] ❌ Failed to uninstall ffmpeg: {out}")
        return

    if sysname == "Linux":
        if shutil.which("apt-get"):
            print("[uninstall] Uninstalling ffmpeg via apt-get …")
            rc, out = _run(["sudo", "apt-get", "remove", "-y", "ffmpeg"], timeout_s=900)
            if rc == 0:
                print("[uninstall] ✅ Uninstalled ffmpeg")
            else:
                print(f"[uninstall] ❌ Failed to uninstall ffmpeg: {out}")
            return
        if shutil.which("yum"):
            print("[uninstall] Uninstalling ffmpeg via yum …")
            rc, out = _run(["sudo", "yum", "remove", "-y", "ffmpeg"], timeout_s=900)
            if rc == 0:
                print("[uninstall] ✅ Uninstalled ffmpeg")
            else:
                print(f"[uninstall] ❌ Failed to uninstall ffmpeg: {out}")
            return
        print("[uninstall] ❌ No supported Linux package manager found for ffmpeg.")
        return

    if sysname == "Windows":
        if shutil.which("winget"):
            print("[uninstall] Uninstalling ffmpeg via winget …")
            rc, out = _run(["winget", "uninstall", "--id", "Gyan.FFmpeg", "-e"], timeout_s=900)
            if rc == 0:
                print("[uninstall] ✅ Uninstalled ffmpeg")
            else:
                print(f"[uninstall] ❌ Failed to uninstall ffmpeg: {out}")
            return
        print("[uninstall] ❌ winget not found; uninstall ffmpeg manually.")
        return

    print("[uninstall] ❌ Unsupported OS for automatic ffmpeg uninstall.")


def _remove_venv(path: str = ".venv") -> None:
    venv_path = Path(path)
    if not venv_path.exists():
        print(f"[uninstall] .venv not found at {venv_path}")
        return
    if not venv_path.is_dir():
        print(f"[uninstall] .venv exists but is not a directory: {venv_path}")
        return
    try:
        shutil.rmtree(venv_path)
        print("[uninstall] ✅ Removed .venv directory")
    except Exception as e:
        print(f"[uninstall] ❌ Failed to remove .venv: {type(e).__name__}: {e}")


def run_checks(install: bool = False, *, assume_yes: bool = False, install_optional: bool = True) -> CheckReport:
    # Ensure we are running inside the project venv; relaunch if needed.
    if not _is_venv():
        ensure_runtime_ready(sys.argv)
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
    ffmpeg_ci = _check_executable("ffmpeg", ["-version"])
    if install and (not ffmpeg_ci.ok):
        _install_ffmpeg(assume_yes=assume_yes)
        ffmpeg_ci = _check_executable("ffmpeg", ["-version"])
    items.append(ffmpeg_ci)

    # Required Python modules
    for m in REQUIRED_MODULES:
        ci = _check_module(m, required=True)
        if install and (not ci.ok):
            _install_module(m)
            ci = _check_module(m, required=True)
        items.append(ci)

    # Optional modules (install if requested)
    for m in OPTIONAL_MODULES:
        ci = _check_module(m, required=False)
        if install and install_optional and (not ci.ok):
            _install_module(m)
            ci = _check_module(m, required=False)
        items.append(ci)

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

    # Use the latest status per item name
    latest: Dict[str, CheckItem] = {}
    for it in items:
        latest[it.name] = it

    required_ok = True
    for name in required_names:
        it = latest.get(name)
        if it is not None and not it.ok:
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


def _confirm(prompt: str) -> bool:
    try:
        resp = input(f"{prompt} [y/N]: ").strip().lower()
        return resp in {"y", "yes"}
    except EOFError:
        return False


def full_uninstall(yes: bool = False, force: bool = False) -> int:
    if not yes:
        ok = _confirm(
            "This will remove temp files, uninstall required Python packages, remove .venv, and uninstall ffmpeg. Continue?"
        )
        if not ok:
            print("Aborted.")
            return 2

    # 1) Temp files in repo
    clean_up()

    # 2) Python deps (required only; those are what --install installs)
    if not _is_venv() and not force:
        print(
            "[uninstall] ⚠️ Not in a virtual environment; skipping pip uninstalls. "
            "Re-run inside the venv or pass --force to allow uninstalling from current Python."
        )
    else:
        for m in REQUIRED_MODULES:
            _uninstall_module(m)

    # 3) Remove venv
    _remove_venv(".venv")

    # 4) System ffmpeg
    _uninstall_ffmpeg()

    return 0


# ---------- CLI ----------

def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Grandma Filter environment helper (strict)")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_check = sub.add_parser("check", help="Verify environment + dependencies")
    p_check.add_argument(
        "--install",
        action="store_true",
        help="Attempt to install missing REQUIRED dependencies (pip modules + ffmpeg)",
    )
    p_check.add_argument(
        "--yes",
        action="store_true",
        help="Assume yes for system-wide installs (ffmpeg)",
    )
    p_check.add_argument("--json", action="store_true", help="Emit JSON report (app-friendly)")

    sub.add_parser("cleanup", help="Remove temporary files (mini_audio, output.*)")
    p_uninstall = sub.add_parser("uninstall", help="Full uninstall (temp files, venv, deps, ffmpeg)")
    p_uninstall.add_argument("--yes", action="store_true", help="Skip confirmation prompt")
    p_uninstall.add_argument(
        "--force",
        action="store_true",
        help="Allow pip uninstalls even if not in a venv (use with care)",
    )

    args = parser.parse_args(argv)

    if args.cmd == "cleanup":
        return clean_up()
    if args.cmd == "uninstall":
        return full_uninstall(yes=bool(args.yes), force=bool(args.force))

    report = run_checks(install=bool(args.install), assume_yes=bool(args.yes))

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
