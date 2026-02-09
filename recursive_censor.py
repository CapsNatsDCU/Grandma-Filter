import argparse
import os
import shutil
import sys
import time
import json

import SWhelper
import main as gf_main


DEFAULT_EXTS = ["mp4", "mov", "mkv", "avi"]


def _iter_media_files_recursive(root: str, extensions: list[str], skip_censored: bool = True) -> list[str]:
    exts = {e.lower() if e.startswith(".") else f".{e.lower()}" for e in extensions}
    out: list[str] = []
    for dirpath, _, filenames in os.walk(root):
        for name in filenames:
            base, ext = os.path.splitext(name)
            if ext.lower() not in exts:
                continue
            if skip_censored and base.endswith("_censored"):
                continue
            out.append(os.path.join(dirpath, name))
    out.sort()
    return out


def _default_output_path(input_file: str) -> str:
    base, ext = os.path.splitext(input_file)
    return f"{base}_censored{ext}"


def _render_bar(pct: float, width: int = 20) -> str:
    pct = max(0.0, min(1.0, pct))
    filled = int(round(pct * width))
    return "[" + ("#" * filled) + ("-" * (width - filled)) + f"] {int(pct*100):3d}%"


def main() -> int:
    SWhelper.ensure_runtime_ready(sys.argv)

    parser = argparse.ArgumentParser(
        description="Grandma Filter – recursive directory censoring with mirrored output"
    )
    parser.add_argument("input_dir", help="Root directory to scan recursively")
    parser.add_argument(
        "--out-dir",
        dest="out_dir",
        default=None,
        help="Output root directory (defaults to <input_dir>_censored)",
    )
    parser.add_argument("-b", "--beep", action="store_true", help="Use beep instead of mute")
    parser.add_argument(
        "--ext",
        dest="extensions",
        action="append",
        default=None,
        help="File extension to include (repeatable). Example: --ext mp4 --ext mov.",
    )
    parser.add_argument(
        "--no-skip-censored",
        action="store_true",
        help="Also process files that already end in _censored",
    )
    parser.add_argument(
        "--keep-original",
        action="store_true",
        help="Keep original files instead of deleting them",
    )
    parser.add_argument(
        "--subs-only",
        action="store_true",
        help="Only write transcript SRT next to original files (no video output, no deletes, no mirroring)",
    )
    parser.add_argument(
        "--yes-install",
        action="store_true",
        help="Assume yes for system-wide installs during the auto-check (ffmpeg)",
    )
    parser.add_argument(
        "--no-check",
        action="store_true",
        help="Skip the automatic dependency check/installer (not recommended)",
    )
    parser.add_argument(
        "--low-conf",
        type=float,
        default=0.6,
        help="Low-confidence cutoff for target words; below this triggers verification (default: 0.6)",
    )
    parser.add_argument(
        "--verify-models",
        type=str,
        default="small,medium,large",
        help="Comma-separated models for low-confidence verification (default: small,medium,large)",
    )
    parser.add_argument(
        "--verify-pad",
        type=float,
        default=0.30,
        help="Seconds to pad around low-confidence word for verification (default: 0.30)",
    )

    args = parser.parse_args()

    input_dir = args.input_dir
    if not os.path.isdir(input_dir):
        print(f"❌ Not a directory: {input_dir}")
        return 2

    out_dir = args.out_dir or f"{input_dir}_censored"
    extensions = args.extensions or DEFAULT_EXTS

    if not args.no_check:
        report = SWhelper.run_checks(
            install=True,
            assume_yes=bool(args.yes_install),
            install_optional=False,
        )
        if not report.required_ok:
            print("❌ Gatekeeper failed. Run: python SWhelper.py check --json")
            return 2

    files = _iter_media_files_recursive(
        input_dir, extensions, skip_censored=(not args.no_skip_censored)
    )
    if not files:
        print(f"No matching media files found in: {input_dir}")
        return 0

    print(f"Found {len(files)} file(s) under {input_dir}")
    total_files = len(files)
    censor_mode = "beep" if args.beep else "mute"

    ok = 0
    failed = 0
    reports: list[dict] = []
    failures: list[dict] = []
    start = time.time()

    def _batch_bar(i: int) -> str:
        pct = i / total_files if total_files else 1.0
        width = 24
        filled = int(round(pct * width))
        return "[" + ("#" * filled) + ("-" * (width - filled)) + f"] {int(pct*100):3d}%"

    for idx, path in enumerate(files, start=1):
        rel = os.path.relpath(path, input_dir)
        rel_dir = os.path.dirname(rel)
        default_out = _default_output_path(path)
        out_subdir = os.path.join(out_dir, rel_dir)
        target_out = os.path.join(out_subdir, os.path.basename(default_out))

        print(f"[{idx}/{len(files)}] {path}")
        print(f"Batch {_batch_bar(idx-1)}")

        try:
            if args.subs_only:
                report = gf_main.main(
                    path,
                    censor_mode,
                    low_conf_threshold=float(args.low_conf),
                    verify_models=[m.strip() for m in args.verify_models.split(",") if m.strip()],
                    verify_pad=float(args.verify_pad),
                    no_subs=False,
                    subs_only=True,
                )
                default_srt = f"{os.path.splitext(default_out)[0]}.srt"
                target_srt = f"{os.path.splitext(path)[0]}.srt"
                if os.path.exists(default_srt):
                    shutil.move(default_srt, target_srt)
                ok += 1
                line = f"Batch {_batch_bar(idx)}"
                print("\r" + line[:72].ljust(72), end="", flush=True)
                print("")
                if report:
                    report["subtitle_only"] = True
                    report["subtitle_file"] = target_srt
                    reports.append(report)
                continue

            os.makedirs(out_subdir, exist_ok=True)

            report = gf_main.main(
                path,
                censor_mode,
                low_conf_threshold=float(args.low_conf),
                verify_models=[m.strip() for m in args.verify_models.split(",") if m.strip()],
                verify_pad=float(args.verify_pad),
            )
            if not os.path.exists(default_out):
                print(f"❌ Missing output: {default_out}")
                failed += 1
                continue

            shutil.move(default_out, target_out)
            default_srt = f"{os.path.splitext(default_out)[0]}.srt"
            target_srt = f"{os.path.splitext(target_out)[0]}.srt"
            if os.path.exists(default_srt):
                shutil.move(default_srt, target_srt)

            if not args.keep_original:
                os.remove(path)

            ok += 1
            print(f"Batch {_batch_bar(idx)}")
            if report:
                report["mirrored_output_file"] = target_out
                reports.append(report)
        except Exception as e:
            print(f"❌ Failed: {path} -> {type(e).__name__}: {e}")
            failed += 1
            failures.append(
                {
                    "input_file": path,
                    "error_type": type(e).__name__,
                    "error": str(e),
                }
            )

    elapsed = time.time() - start
    if reports:
        total_media_s = sum(r.get("media_duration_s", 0.0) for r in reports)
        total_proc_s = sum(r.get("processing_time_s", 0.0) for r in reports)
        total_hits = sum(r.get("muted_hit_count", 0) for r in reports)
        avg_media_s = total_media_s / len(reports) if reports else 0.0
        avg_proc_s = total_proc_s / len(reports) if reports else 0.0

        os.makedirs("reports", exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        super_path = os.path.join("reports", f"recursive_report.{stamp}.json")
        with open(super_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "failures": failures,
                    "summary": {
                        "files_total": len(reports),
                        "files_ok": ok,
                        "files_failed": failed,
                        "muted_hit_count": total_hits,
                        "media_duration_s_total": round(total_media_s, 3),
                        "media_duration_s_avg": round(avg_media_s, 3),
                        "processing_time_s_total": round(total_proc_s, 3),
                        "processing_time_s_avg": round(avg_proc_s, 3),
                        "wall_time_s_total": round(elapsed, 3),
                        "censor_mode": censor_mode,
                        "input_dir": input_dir,
                        "output_dir": out_dir,
                    },
                    "files": reports,
                },
                f,
                indent=2,
            )
        print(f"Super report -> {super_path}")
        print(
            f"Summary: files={len(reports)} ok={ok} failed={failed} "
            f"media_s={total_media_s:.1f} processed_s={total_proc_s:.1f} muted_hits={total_hits}"
        )
    else:
        print(f"Done. ok={ok} failed={failed} elapsed={elapsed:.1f}s")
    return 0 if failed == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
