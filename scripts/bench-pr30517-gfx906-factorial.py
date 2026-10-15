#!/usr/bin/env python3
"""PR #30517 gfx906 2x2 experiment: baseline / __vsub4 / nwarps=1 / both (+ master).
Launch directly from GitHub: curl -fsSL URL | python3 - --quick
Requires a llama.cpp git checkout, ROCm, a gfx906 GPU, CMake, C++ compiler.
Changes are confined to isolated worktrees under --out; current branch is untouched.
"""
import argparse
from collections import defaultdict
from datetime import datetime
from pathlib import Path
import os
import re
import shutil
import statistics
import subprocess
import sys

FORK = "https://github.com/thelittlefireman/llama.cpp.git"
VERSIONS = {
    "master":  "8c2134a875",
    "original": "c869d1e38a3f541bf8475eec263926f06dd398e0",
    "vsub4":  "4d5f0a39112a7ae429199470f9003a5ef126bae1",
    "nwarps1": "4b9ce5b902a8f3df2544e450be5b92453a87311e",
    "both":    "2f410fe14b67fa43eba5a7bd846b3f07c21e400f",
}
TARGETS = (("q3_K", 2), ("q3_K", 3), ("q6_K", 4))
CONTROLS = (("q2_K", 3), ("q3_K", 4), ("q6_K", 3))
TYPES = {"q2_K": "GGML_TYPE_Q2_K", "q3_K": "GGML_TYPE_Q3_K", "q6_K": "GGML_TYPE_Q6_K"}
ANSI = re.compile(r"\x1b\[[0-9;]*m")
PERF = re.compile(
    r"MUL_MAT\(type_a=(q[236]_K),type_b=f32,m=(\d+),n=(\d+),k=(\d+),[^\n)]*\)"
    r":\s*\d+ runs\s*-\s*([0-9]+(?:\.[0-9]+)?) us/run")
TAG = "    // PR30517 GFX906 FACTORIAL TEST GENERATED"
BLOCK = re.compile(
    r"(?m)^    // PR30517 GFX906 FACTORIAL TEST GENERATED\n"
    r"(?:    test_cases\.emplace_back\(new test_mul_mat\([^\n]+\);\n)+\n")


def shell(args, cwd=None, log=None, env=None):
    print("+", " ".join(map(str, args)), flush=True)
    if log is None:
        subprocess.run(args, cwd=cwd, env=env, check=True)
        return ""
    with log.open("w") as f:
        rc = subprocess.run(args, cwd=cwd, env=env, stdout=f, stderr=subprocess.STDOUT).returncode
    if rc:
        raise RuntimeError(f"Command failed ({rc}), see {log}")
    return log.read_text(errors="replace")


def matrix(quick):
    ms = (4095, 4096, 4097) if quick else (1023, 4095, 4096, 4097)
    ks = (256, 1024, 14336) if quick else (256, 512, 1024, 4096, 14336)
    cases = {(t, m, n, k) for t, n in TARGETS for m in ms for k in ks}
    cases |= {(t, 4096, n, k) for t, n in CONTROLS for k in (256, 14336)}
    return sorted(cases)


def patch_test_cases(file, cases):
    src = file.read_text()
    src = BLOCK.sub("", src)
    eval_marker = "    // m == 1, with n on both sides of MMVF_MAX_BATCH_SIZE"
    perf_marker = "    for (int bs : {1, 2, 3, 4, 5, 8, 512}) {"
    inserts = TAG + "\n" + "".join(
        f"    test_cases.emplace_back(new test_mul_mat({TYPES[t]}, GGML_TYPE_F32, "
        f"{m}, {n}, {k}, {{1, 1}}, {{1, 1}}));\n" for t, m, n, k in cases
    ) + "\n"
    for marker in (eval_marker, perf_marker):
        if src.count(marker) != 1:
            raise RuntimeError(f"Test source has unexpected marker count: {marker}")
        src = src.replace(marker, inserts + marker, 1)
    file.write_text(src)


def filter_expr(cases):
    return "(?:" + "|".join(f"type_a={t},type_b=f32,m={m},n={n},k={k},"
                            for t,m,n,k in cases) + ")"


def measure(log_text, cases):
    out = {}
    for typ,m,n,k,us in PERF.findall(ANSI.sub("", log_text)):
        key = (typ,int(m),int(n),int(k))
        if key in cases:
            out[key] = float(us)
    return out


def markdown(cases, versions, samples, threshold, errors):
    cols = " | ".join(f"{v} us" for v in versions)
    lines = [
        "# PR30517 gfx906 factorial results", "",
        "Each number is the median GPU kernel time in microseconds (lower is better).",
        "Speedups are original/variant - 1; positive numbers mean faster.",
        f"Regression warning threshold: {threshold}%.", "",
        f"| Quant | M | n | K | {cols} | vsub4 gain | nwarps gain | combined gain | Best PR variant |",
        "|---|---:|---:|---:|" + "|---:"*len(versions) + "|---:|---:|---:|---|"
    ]
    fail = list(errors)
    comparisons = defaultdict(lambda: {"wins": 0, "regressions": 0, "missing": 0})
    for t,m,n,k in cases:
        case=(t,m,n,k)
        med={name: statistics.median(samples[name][case])
             if samples[name][case] else None for name in versions}
        if any(v is None for v in med.values()):
            fail.append(f"Missing measurements: {case}: {med}")
            for name,v in med.items():
                if v is None: comparisons[name]["missing"]+=1
            continue
        original=med["original"]
        gains={name:100*(original/med[name]-1) for name in ("vsub4","nwarps1","both")}
        for name,gain in gains.items():
            if gain>0: comparisons[name]["wins"]+=1
            if gain < -threshold:
                comparisons[name]["regressions"]+=1
                fail.append(f"{case}: {name} {gain:.2f}% vs original")
        best=min(("original","vsub4","nwarps1","both"), key=lambda x:med[x])
        if med["both"] > min(med["vsub4"], med["nwarps1"])*(1+threshold/100):
            fail.append(f"{case}: both is >{threshold}% slower than best single optimization")
        if "master" in med and med["both"]>med["master"]*(1+threshold/100):
            fail.append(f"{case}: both slower than master by >{threshold}%")
        parts=[t,str(m),str(n),str(k)]
        parts += [f"{med[name]:.2f}" for name in versions]
        parts += [f"{gains[name]:+.2f}%" for name in ("vsub4","nwarps1","both")]
        parts += [best]
        lines.append("| " + " | ".join(parts) + " |")
    lines += ["", "## Summary", "", "| Variant | Faster than original | >threshold regressions | Missing |",
              "|---|---:|---:|---:|"]
    for name in ("vsub4","nwarps1","both"):
        counts=comparisons[name]
        lines.append(f"| {name} | {counts['wins']} | {counts['regressions']} | {counts['missing']} |")
    lines += ["", f"Covered cases: {len(cases)}.", f"Alerts: {len(fail)}."]
    if fail:
        lines += ["", "## Alerts", ""] + ["- "+a for a in fail]
    return "\n".join(lines)+"\n", fail


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", type=Path, default=Path.cwd())
    ap.add_argument("--out", type=Path, default=Path.home() / "bench-pr30517-factorial")
    ap.add_argument("--quick", action="store_true", help="33 cases instead of 66")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--jobs", type=int, default=2, help="Parallel build jobs; default 2 for limited RAM")
    ap.add_argument("--regression-pct", type=float, default=5.0)
    ap.add_argument("--skip-correctness", action="store_true")
    ap.add_argument("--prepare-only", action="store_true")
    ap.add_argument("--without-master", action="store_true")
    args=ap.parse_args()
    if args.rounds < 1 or args.jobs < 1:
        ap.error("--rounds and --jobs must be positive")
    repo=args.repo.resolve()
    out=args.out.expanduser().resolve()
    if subprocess.check_output(["git","-C",str(repo),"rev-parse","--show-toplevel"],
                               text=True).strip()!=str(repo):
        ap.error("--repo must be the root of a llama.cpp git checkout")
    out.mkdir(parents=True, exist_ok=True)
    versions={k:v for k,v in VERSIONS.items() if k!="master" or not args.without_master}
    cases=matrix(args.quick)
    case_set=set(cases)
    filt=filter_expr(cases)
    # Fetch the fork by URL; origin may point to upstream ggml-org/llama.cpp.
    shell(["git","-C",str(repo),"fetch","--no-tags",FORK,
           "master","test_PR_30517_MMVQ",
           "test_PR_30517_VSUB4","test_PR_30517_VSUB4_NWARPS1"])
    paths={}
    for name,sha in versions.items():
        wt=out / "worktrees" / name
        if not wt.is_dir():
            wt.parent.mkdir(parents=True,exist_ok=True)
            shell(["git","-C",str(repo),"worktree","add","--detach",str(wt),sha])
        else:
            got=subprocess.check_output(["git","-C",str(wt),"rev-parse","HEAD"],text=True).strip()
            expect=subprocess.check_output(["git","-C",str(repo),"rev-parse",sha],text=True).strip()
            if got!=expect:
                raise RuntimeError(f"Unexpected worktree HEAD for {name}: {got}")
        test_cpp=wt / "tests" / "test-backend-ops.cpp"
        patch_test_cases(test_cpp,cases)
        build=wt/"build-gfx906"
        configure=["cmake","-S",str(wt),"-B",str(build),
                   "-DCMAKE_BUILD_TYPE=Release","-DGGML_HIP=ON",
                   "-DGPU_TARGETS=gfx906","-DGGML_HIP_NO_VMM=ON",
                   "-DLLAMA_BUILD_TESTS=ON"]
        if shutil.which("ninja"): configure+=["-G","Ninja"]
        shell(configure)
        shell(["cmake","--build",str(build),"--target","test-backend-ops",
               "-j",str(args.jobs)])
        exe=build/"bin"/"test-backend-ops"
        if not exe.exists(): raise RuntimeError(f"Missing executable: {exe}")
        paths[name]=exe
    if args.prepare_only:
        print(f"Prepared {len(cases)} cases and {len(versions)} builds in {out}")
        return 0
    env=dict(os.environ, HIP_VISIBLE_DEVICES="0")
    errors=[]
    if not args.skip_correctness:
        for name,exe in paths.items():
            log=out/f"test-{name}.log"
            try:
                output=shell([str(exe),"test","-b","ROCm0","-o","MUL_MAT","-p",filt],
                             env=env,log=log)
                if not re.search(r"Backend ROCm0:\s*(?:\x1b\[[^m]*m)*OK",output):
                    raise RuntimeError("ROCm0 success not found")
            except RuntimeError as e:
                errors.append(f"{name} correctness: {e}")
    samples={v:defaultdict(list) for v in versions}
    names=list(versions)
    for round_id in range(args.rounds):
        ordered=names if round_id%2==0 else list(reversed(names))
        for name in ordered:
            log=out/f"perf-{round_id+1}-{name}.log"
            output=shell([str(paths[name]),"perf","-b","ROCm0","-o","MUL_MAT","-p",filt],
                         env=env,log=log)
            if "gfx906" not in output:
                raise RuntimeError(f"gfx906 not detected in {log}")
            got=measure(output,case_set)
            if len(got)!=len(cases):
                errors.append(f"{name} round {round_id+1}: only {len(got)}/{len(cases)} cases ({log})")
            for case,value in got.items():
                samples[name][case].append(value)
    report,alerts=markdown(cases,versions,samples,args.regression_pct,errors)
    (out/"report.md").write_text(report)
    print(report)
    print(f"Logs and report: {out}")
    return 2 if alerts else 0


if __name__=="__main__":
    try: sys.exit(main())
    except (subprocess.CalledProcessError,OSError,RuntimeError) as exc:
        print(f"ERROR: {exc}",file=sys.stderr)
        sys.exit(1)
