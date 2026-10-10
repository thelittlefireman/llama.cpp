#!/usr/bin/env python3
"""Trace gfx906 ds_bpermute address registers used by FP32 additions.

Candidates need manual lane/EXEC and floating-point validation before modifying kernels.
"""
import argparse
import csv
import re
from bisect import bisect_left
from collections import Counter, defaultdict
from pathlib import Path

INST = re.compile(r"^\s*([a-z][a-z0-9_]*)\s+(.+?)\s*$")
LABEL = re.compile(r"^([.$A-Za-z_][\w.$@]*):")
REG = re.compile(r"^[vs]\d+$")
DPP_XOR = {1: "quad_perm:[1,0,3,2]", 2: "quad_perm:[2,3,0,1]", 3: "quad_perm:[3,2,1,0]", 8: "row_ror:8"}


def integer(value):
    try:
        return int(value, 0)
    except ValueError:
        return None


def parse(line, number, function, block):
    clean = line.split(";", 1)[0].strip()
    match = INST.match(clean)
    if not match or not match[1].startswith(("v_", "ds_", "s_", "flat_", "global_", "buffer_")):
        return None
    op, args = match.groups()
    parts = [item.strip().split(" ", 1)[0] for item in re.split(r",(?![^\[]*\])", args)]
    return dict(line=number, op=op, args=args, parts=parts, function=function, block=block)


def defines(ins, reg):
    return ins["op"].startswith(("v_", "ds_")) and bool(ins["parts"]) and ins["parts"][0] == reg


def expand(name, inst, definitions, index, block, function, depth=0, chain=None):
    if chain is None:
        chain = []
    if depth >= 14:
        return ("limit",), chain, False
    val = integer(name)
    if val is not None:
        return ("const", val), chain, True
    if not REG.fullmatch(name):
        return ("unknown", name), chain, False
    entries = definitions.get((function, name), [])
    pos = bisect_left(entries, index) - 1
    if pos < 0:
        return ("entry", name), chain, False
    k = entries[pos]
    item = inst[k]
    local = item["block"] == block
    trace = chain + [f'{item["line"]}: {item["op"]} {item["args"]}']
    op, p = item["op"], item["parts"]
    if op.startswith("v_mbcnt_hi_u32_b32"):
        return ("lane",), trace, local
    if op.startswith("v_mbcnt_lo_u32_b32"):
        return ("lane_low",), trace, local

    def arg(n):
        if n >= len(p):
            return ("unknown",), trace, False
        return expand(p[n], inst, definitions, k, block, function, depth + 1, trace)

    if op.startswith("v_cndmask_b32") and len(p) >= 4:
        a, ta, sa = arg(1)
        b, tb, sb = arg(2)
        return ("select", a, b), ta + tb[len(trace):], local and sa and sb
    if op.startswith("v_mov_b32"):
        expr, track, trusted = arg(1)
        return expr, track, local and trusted
    if op.startswith(("v_lshlrev_b32", "v_lshrrev_b32")) and len(p) >= 3:
        amount = integer(p[1])
        if amount is None:
            return ("unknown", op), trace, False
        expr, track, trusted = arg(2)
        return ("shl" if op.startswith("v_lshlrev_b32") else "shr", expr, amount), track, local and trusted
    binary = (("v_xor_b32", "xor"), ("v_and_b32", "and"), ("v_or_b32", "or"),
              ("v_add_u32", "add"), ("v_add_co_u32", "add"), ("v_sub_u32", "sub"),
              ("v_mul_lo_u32", "mul"))
    for prefix, symbol in binary:
        if op.startswith(prefix) and len(p) >= 3:
            a, ta, sa = arg(1)
            b, tb, sb = arg(2)
            return (symbol, a, b), ta + tb[len(trace):], local and sa and sb
    return ("unknown", op), trace, False


def expression_text(expr):
    if expr[0] == "const":
        return hex(expr[1])
    if expr[0] == "lane":
        return "lane_id"
    if expr[0] in ("entry", "unknown"):
        return ":".join(str(x) for x in expr)
    if len(expr) == 3:
        second = expression_text(expr[2]) if isinstance(expr[2], tuple) else str(expr[2])
        return f"{expr[0]}({expression_text(expr[1])}, {second})"
    return expr[0]


def lane_base(expr):
    if expr[0] == "lane":
        return "lane_id"
    if expr[0] == "and":
        a, b = expr[1:3]
        if a[0] == "const":
            a, b = b, a
        if b[0] == "const" and b[1] in (15, 31, 63) and lane_base(a):
            return f"lane_mask_{b[1]}"
    if expr[0] == "entry":
        return "entry_unverified"
    return None


def address_kind(expr):
    if expr[0] != "shl" or expr[2] != 2:
        return "unknown_address", "", ""
    index = expr[1]
    if index[0] == "select":
        a, b = index[1:3]
        if b[0] != "xor":
            return "unknown_address", "", ""
        first, second = b[1:3]
        if first[0] == "const":
            offset, base = first[1], second
        elif second[0] == "const":
            offset, base = second[1], first
        else:
            return "bounded_xor_unverified", "", ""
        if a != base or not lane_base(base):
            return "bounded_xor_unverified", str(offset), lane_base(base) or "unknown_base"
        if offset in DPP_XOR:
            return "bounded_xor_direct_dpp", str(offset), lane_base(base)
        if offset in (4, 16, 32):
            return "bounded_xor_not_single_dpp", str(offset), lane_base(base)
        return "bounded_xor_other", str(offset), lane_base(base)
    if index[0] != "xor":
        return ("identity" if lane_base(index) else "unknown_address"), "", lane_base(index) or ""
    a, b = index[1:3]
    if a[0] == "const":
        offset, base = a[1], b
    elif b[0] == "const":
        offset, base = b[1], a
    else:
        return "xor_unverified", "", "unknown_base"
    lane = lane_base(base)
    if lane is None:
        return "xor_unverified", str(offset), "unknown_base"
    if offset in DPP_XOR:
        return "xor_direct_dpp", str(offset), lane
    if offset in (4, 16, 32):
        return "xor_not_single_dpp", str(offset), lane
    return "xor_other", str(offset), lane

def scan(path, sites, counts):
    inst = []
    function, block = "(unknown)", 0
    with path.open(encoding="utf-8", errors="replace") as stream:
        for line_number, line in enumerate(stream, 1):
            label = LABEL.match(line)
            if label:
                block += 1
                if not label[1].startswith(".L"):
                    function = label[1]
                continue
            item = parse(line, line_number, function, block)
            if item:
                inst.append(item)
    definitions = defaultdict(list)
    for i, item in enumerate(inst):
        if item["op"].startswith(("v_", "ds_")) and item["parts"] and REG.fullmatch(item["parts"][0]):
            definitions[(item["function"], item["parts"][0])].append(i)
    for i, item in enumerate(inst):
        if not item["op"].startswith("ds_bpermute_b32") or len(item["parts"]) < 3:
            continue
        dst, addr, src = item["parts"][:3]
        found, found_index = None, -1
        for j in range(i + 1, min(len(inst), i + 25)):
            nxt = inst[j]
            if nxt["block"] != item["block"]:
                break
            if nxt["op"].startswith("v_add_f32") and dst in nxt["parts"][1:3]:
                found, found_index = nxt, j
                break
            if defines(nxt, dst):
                break
        if found is None:
            continue
        expr, trace, trusted = expand(addr, inst, definitions, i, item["block"], item["function"])
        kind, mask, base = address_kind(expr)
        source_alive = not any(defines(nxt, src) for nxt in inst[i + 1:found_index])
        src_in_add = src in found["parts"][1:3]
        exec_changed = any(nxt["op"].startswith(("s_and_saveexec", "s_or_saveexec", "v_cmpx_"))
                           for nxt in inst[i + 1:found_index])
        scope = "same_block" if trusted else "cross_block_or_unknown"
        status = "unclassified"
        if kind == "bounded_xor_direct_dpp":
            status = "guard_must_be_verified"
        elif kind == "bounded_xor_not_single_dpp":
            status = "unsupported_single_dpp"
        elif kind == "xor_direct_dpp":
            status = ("candidate_verify_exec_and_rounding" if trusted and source_alive and src_in_add and not exec_changed
                      else "manual_review")
        elif kind == "xor_not_single_dpp":
            status = "unsupported_single_dpp"
        elif kind == "xor_unverified":
            status = "verify_lane_origin"
        row = dict(file=str(path), function=item["function"], block=item["block"],
                   permute_line=item["line"], add_line=found["line"], kind=kind, xor=mask, base=base,
                   scope=scope, status=status, address_reg=addr, source_reg=src, result_reg=dst,
                   expression=expression_text(expr), address_trace=" | ".join(trace[:18]),
                   permute=item["op"] + " " + item["args"], add=found["op"] + " " + found["args"])
        sites.append(row)
        counts[(str(path), item["function"], kind, mask, scope, status)] += 1


def write_table(path, columns, rows):
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path)
    parser.add_argument("--out", default="gfx906-address")
    args = parser.parse_args()
    files = sorted(args.path.rglob("*gfx906*.s")) if args.path.is_dir() else [args.path]
    if not files:
        parser.error("No gfx906 assembly files found")
    sites, counts = [], Counter()
    for file in files:
        scan(file, sites, counts)
    prefix = Path(args.out)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    sites.sort(key=lambda x: (x["status"] != "candidate_verify_exec_and_rounding", x["kind"] != "xor_direct_dpp",
                              x["file"], x["permute_line"]))
    columns = ["file", "function", "block", "permute_line", "add_line", "kind", "xor", "base", "scope",
               "status", "address_reg", "source_reg", "result_reg", "expression", "address_trace", "permute", "add"]
    write_table(f"{prefix}-sites.tsv", columns, sites)
    rows = [dict(file=k[0], function=k[1], kind=k[2], xor=k[3], scope=k[4], status=k[5], count=value)
            for k, value in counts.items()]
    rows.sort(key=lambda x: (-x["count"], x["file"], x["function"]))
    write_table(f"{prefix}-functions.tsv", ["file", "function", "kind", "xor", "scope", "status", "count"], rows)
    by_kind = Counter(row["kind"] for row in sites)
    by_status = Counter(row["status"] for row in sites)
    candidate_files = Counter(Path(row["file"]).name for row in sites
                              if row["status"] == "candidate_verify_exec_and_rounding")
    summary = ["# gfx906 bpermute address tracing", "", f"Assembly files: {len(files)}",
               f"Permute-to-add sites: {len(sites)}", "", "## Address patterns"]
    summary.extend(f"- {name}: {value}" for name, value in by_kind.most_common())
    summary.extend(["", "## Safety classification"])
    summary.extend(f"- {name}: {value}" for name, value in by_status.most_common())
    summary.extend(["", "## Candidate ISA files"])
    summary.extend(f"- {name}: {value}" for name, value in candidate_files.most_common(30))
    summary.extend(["", "Counts are static, not execution frequencies.",
                    "Cross-block tracing is speculative; manually verify EXEC, lane layout, floating point and source liveness."])
    Path(f"{prefix}-summary.md").write_text("\n".join(summary) + "\n", encoding="utf-8")
    print("\n".join(summary))


if __name__ == "__main__":
    main()
