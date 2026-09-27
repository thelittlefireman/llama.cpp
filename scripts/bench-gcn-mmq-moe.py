#!/usr/bin/env python3
"""Compare GCN MoE MMQ target widths with the existing backend test runner."""

import argparse
import csv
import json
import os
from pathlib import Path
import random
import re
import statistics
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", default="build-feature_GCN_MOE_MMQ_DIAGNOSTICS")
    parser.add_argument("--backend", default="ROCm0")
    parser.add_argument("--mode", choices=("test", "perf"), default="perf")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--params", default=r"^mmq_moe=1,.*n_mats=40,")
    parser.add_argument("--widths", default="legacy,auto,8,16,24,32,40,48,64,128")
    parser.add_argument("--out")
    args = parser.parse_args()
    if args.runs < 1:
        parser.error("--runs must be positive")
    widths = args.widths.split(",")
    allowed = {"legacy", "auto"} | {str(j) for j in range(8, 129, 8)}
    if not widths or len(set(widths)) != len(widths) or any(j not in allowed for j in widths):
        parser.error("--widths requires distinct entries: legacy, auto, or multiples of 8 from 8 to 128")

    binary = (Path(args.build) / "bin/test-backend-ops").resolve()
    if not binary.is_file():
        parser.error(f"missing binary: {binary}")
    print(f"Runner: {binary} (backend: {args.backend})", flush=True)
    out = Path(args.out or f"mmq-moe-{args.mode}")
    out.mkdir(parents=True, exist_ok=False)
    command = [str(binary), args.mode, "-b", args.backend, "-o", "MUL_MAT_ID", "-p", args.params]
    metadata = {"args": vars(args), "command": command, "env": {
        k: os.environ[k] for k in ("HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "GGML_CUDA_DISABLE_GRAPHS", "GGML_CUDA_GRAPH_OPT") if k in os.environ
    }}
    (out / "run.json").write_text(json.dumps(metadata, indent=2) + "\n")

    ansi = re.compile(r"\x1b\[[0-9;]*m")
    perf_pattern = re.compile(r"MUL_MAT_ID\((mmq_moe=1,.*?)\):\s+\d+ runs -\s+([0-9.]+) us/run")
    test_pattern = re.compile(r"MUL_MAT_ID\((mmq_moe=1,.*?)\):[^\n]*\bOK\b")
    rng = random.Random(1234)
    samples = {}
    expected_cases = None
    runs = 1 if args.mode == "test" else args.runs
    with (out / "samples.csv").open("w", newline="") as sample_file:
        writer = csv.writer(sample_file)
        writer.writerow(["repeat", "target", "params", "time_us"])
        for repeat in range(1, runs + 1):
            order = widths.copy()
            rng.shuffle(order)
            for width in order:
                env = os.environ.copy()
                env.pop("GGML_CUDA_MMQ_MOE_NCOLS", None)
                if width != "auto":
                    env["GGML_CUDA_MMQ_MOE_NCOLS"] = "0" if width == "legacy" else width
                env["GGML_CUDA_MMQ_MOE_TRACE"] = "1"
                stem = out / f"{repeat:02d}-{width}"
                print(f"{args.mode}: repeat {repeat}/{runs}, target {width}", flush=True)
                with stem.with_suffix(".log").open("w") as stderr:
                    result = subprocess.run(command, env=env, stdout=subprocess.PIPE, stderr=stderr, text=True)
                stem.with_suffix(".txt").write_text(result.stdout)
                stderr_text = stem.with_suffix(".log").read_text()

                def fail(message):
                    print(f"Runner: {binary}", file=sys.stderr)
                    for label, contents in (("stdout", result.stdout), ("stderr", stderr_text)):
                        print(f"--- {label} (last 40 lines) ---", file=sys.stderr)
                        print("\n".join(contents.splitlines()[-40:]), file=sys.stderr)
                    raise SystemExit(f"{message}; full output: {stem}.txt and {stem}.log")

                if result.returncode:
                    fail(f"test runner failed ({result.returncode})")
                output = ansi.sub("", result.stdout)
                if "MUL_MAT_ID(mmq_moe=1," not in output:
                    fail("no diagnostic cases executed: check --build, --backend and --params; rebuild test-backend-ops from feature_GCN_MOE_MMQ_DIAGNOSTICS")
                if "not supported" in output or "skipping large tensors" in output:
                    fail("some cases were skipped")
                if "mmq-moe:" not in result.stdout + stderr_text:
                    fail("diagnostic cases ran without a GCN MoE MMQ trace: check the GPU, rebuild the HIP backend and check which backend library is loaded")
                if args.mode == "perf":
                    matches = perf_pattern.findall(output)
                    cases = {case for case, _ in matches}
                    if len(cases) != len(matches):
                        fail("duplicate cases")
                    for case, time_us in matches:
                        time_us = float(time_us)
                        if time_us <= 0:
                            fail("invalid timing")
                        writer.writerow([repeat, width, case, time_us])
                        samples.setdefault(case, {}).setdefault(width, []).append(time_us)
                    sample_file.flush()
                else:
                    cases = set(test_pattern.findall(output))
                if not cases:
                    fail("no measured/passed cases")
                if expected_cases is None:
                    expected_cases = cases
                elif cases != expected_cases:
                    fail("case set changed")

    if args.mode == "perf":
        with (out / "summary.csv").open("w", newline="") as summary_file:
            writer = csv.writer(summary_file)
            writer.writerow(["params", "target", "median_us", "min_us", "max_us", "speed_vs_legacy_pct"])
            for case, measurements in sorted(samples.items()):
                baseline = statistics.median(measurements["legacy"]) if "legacy" in measurements else None
                for width, values in measurements.items():
                    median = statistics.median(values)
                    gain = 100 * (baseline / median - 1) if baseline is not None else ""
                    writer.writerow([case, width, median, min(values), max(values), gain])
        print(f"Results: {out / 'summary.csv'}")
    else:
        print(f"Passed {len(expected_cases)} cases for each target. Logs: {out}")


if __name__ == "__main__":
    main()
