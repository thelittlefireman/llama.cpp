#!/usr/bin/env python3
"""Scan all gfx906 HIP assembly for DPP-to-FMAC candidates.

Static ISA analysis only: candidates are not proof of equivalent semantics or a speedup.
"""

import argparse
import csv
import re
from collections import Counter, defaultdict
from pathlib import Path

INSTRUCTION = re.compile(r"^\s*([a-z][a-z0-9_]*)\s+(.+?)\s*$")
LABEL = re.compile(r"^([.$A-Za-z_][\w.$@]*):")
REGISTER = re.compile(r"\bv\d+\b")
REGISTER_RANGE = re.compile(r"v\[(\d+):(\d+)\]")
FMAC = re.compile(r"v_(?:fmac|mac)_f32(?:_e32|_e64)?$")
FMA = re.compile(r"v_fma_f32(?:_e32|_e64)?$")
DPP_MOVE = re.compile(r"v_mov_b32_dpp$")
BPERMUTE = re.compile(r"ds_bpermute_b32$")
DPP_FMAC = re.compile(r"v_(?:fmac|mac)_f32_dpp$")
WRITES_EXEC = re.compile(r"^(?:s_.*exec|v_cmpx_.*)$")


def registers(text):
    result = set(REGISTER.findall(text))
    for first, last in REGISTER_RANGE.findall(text):
        result.update(f"v{i}" for i in range(int(first), int(last) + 1))
    return result


def parse_instruction(line, number, function, block):
    code = line.split(";", 1)[0].split("//", 1)[0].strip()
    match = INSTRUCTION.match(code)
    if not match:
        return None
    op, args = match.groups()
    if not op.startswith(("v_", "ds_", "s_", "flat_", "global_", "buffer_", "image_")):
        return None
    parts = [part.strip() for part in args.split(",")]
    defs = registers(parts[0]) if op.startswith("v_") or op.startswith(("ds_read", "ds_bpermute", "ds_permute", "ds_swizzle")) else set()
    uses = registers(args[len(parts[0]):])
    if FMAC.fullmatch(op) or DPP_FMAC.fullmatch(op) or op.startswith("v_add_f32_dpp"):
        uses |= defs
    return dict(line=number, function=function, block=block, op=op, args=args, parts=parts, defs=defs, uses=uses)


def scan_block(instructions, file, stats, candidates):
    # A simple local def-use analysis: never infer a mapping across basic blocks.
    producers = {}
    uses = defaultdict(list)
    definitions = defaultdict(list)
    for index, ins in enumerate(instructions):
        for reg in ins["uses"]:
            uses[reg].append(index)
        for reg in ins["defs"]:
            definitions[reg].append(index)

    for index, ins in enumerate(instructions):
        key = (file, ins["function"])
        count = stats[key]
        op = ins["op"]
        if FMAC.fullmatch(op):
            count["fmac"] += 1
        if FMA.fullmatch(op):
            count["fma"] += 1
        if op.startswith("v_fma_mix_f32"):
            count["fma_mix"] += 1
        if op.startswith("v_pk_fma_f16"):
            count["pk_fma_f16"] += 1
        if op.startswith("v_dot"):
            count["dot"] += 1
        if DPP_FMAC.fullmatch(op):
            count["fmac_dpp"] += 1
        if op.startswith("v_add_f32_dpp"):
            count["add_dpp"] += 1
        if DPP_MOVE.fullmatch(op):
            count["mov_dpp"] += 1
        if BPERMUTE.fullmatch(op):
            count["bpermute"] += 1

        kind = "fmac" if FMAC.fullmatch(op) else "fma" if FMA.fullmatch(op) else "add" if op.startswith("v_add_f32") else ""
        if kind:
            srcs = ins["parts"][1:3] if kind in ("fmac", "add") else ins["parts"][1:4]
            for operand_idx, operand in enumerate(srcs):
                if not REGISTER.fullmatch(operand):
                    continue
                producer = producers.get(operand)
                if producer is None:
                    continue
                src_idx, old = producer
                if old["block"] != ins["block"]:
                    continue
                if ins["line"] - old["line"] > MAX_GAP:
                    continue
                source_type = "dpp" if DPP_MOVE.fullmatch(old["op"]) else "bpermute"
                if kind == "add":
                    category, reason = "bpermute_add", "not_fmac"
                    if source_type != "bpermute":
                        continue
                elif source_type == "bpermute":
                    category, reason = "bpermute_fmac", "dynamic_lane_mapping"
                else:
                    category = "direct_" + kind
                    reason = "review"
                    if kind == "fma" and (len(srcs) != 3 or ins["parts"][0] != srcs[2]):
                        reason = "non_accumulating_fma"
                    elif operand_idx == 1:
                        reason = "operand_swap_review"
                    elif operand_idx == 2:
                        reason = "accumulator_is_permuted"
                    if "row_mask:0xf" not in old["args"] or "bank_mask:0xf" not in old["args"]:
                        reason = "mask_review"
                    if not ("row_ror:" in old["args"] or "quad_perm:" in old["args"]):
                        reason = "bounds_review" if reason == "review" else reason
                    old_parts = old["parts"]
                    if len(old_parts) < 2 or not REGISTER.fullmatch(old_parts[1]):
                        reason = "source_operand_review"
                    elif any(old_parts[1] in item["defs"] for item in instructions[src_idx + 1:index]):
                        reason = "source_overwritten"
                    if operand in ins["defs"]:
                        reason = "accumulator_alias"
                    remaining_uses = [j for j in uses[operand] if src_idx < j and (not any(k in definitions[operand] for k in range(src_idx + 1, j)))]
                    if len(remaining_uses) > 1:
                        reason = "shared_temporary"
                    if any(WRITES_EXEC.match(item["op"]) for item in instructions[src_idx + 1:index]):
                        reason = "exec_mask_changed"
                count[category] += 1
                if category.startswith("direct_") and reason in ("review", "operand_swap_review"):
                    count["priority_candidates"] += 1
                candidates.append(dict(file=file, function=ins["function"], category=category,
                                       status=reason, producer_line=old["line"], consumer_line=ins["line"],
                                       gap=ins["line"] - old["line"], temporary=operand,
                                       producer=old["op"] + " " + old["args"], consumer=op + " " + ins["args"]))

        for register in ins["defs"]:
            producers.pop(register, None)
        if DPP_MOVE.fullmatch(op) and len(ins["parts"]) >= 2:
            producers[ins["parts"][0]] = (index, ins)
        elif BPERMUTE.fullmatch(op) and len(ins["parts"]) >= 3:
            producers[ins["parts"][0]] = (index, ins)


def scan_file(path, label, stats, candidates):
    instructions = []
    current_function = "(unknown)"
    block = 0
    with path.open(encoding="utf-8", errors="replace") as stream:
        for number, raw in enumerate(stream, 1):
            match = LABEL.match(raw)
            if match:
                if instructions:
                    scan_block(instructions, label, stats, candidates)
                    instructions.clear()
                if not match[1].startswith(".L"):
                    current_function = match[1]
                block += 1
                continue
            ins = parse_instruction(raw, number, current_function, block)
            if ins:
                instructions.append(ins)
    if instructions:
        scan_block(instructions, label, stats, candidates)


def write_tsv(path, columns, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", help="Generated gfx906 .s files or directories")
    parser.add_argument("--out", default="gfx906-fmac-dpp", help="Report filename prefix")
    parser.add_argument("--max-gap", type=int, default=48, help="Maximum ISA line distance")
    args = parser.parse_args()
    global MAX_GAP
    MAX_GAP = args.max_gap
    paths = sorted({file for name in args.paths for item in [Path(name)] for file in
                    (item.rglob("*gfx906*.s") if item.is_dir() else [item]) if file.is_file()})
    if not paths:
        parser.error("No gfx906 assembly found; check --save-temps and the HIP build.")
    root = Path.cwd()
    stats = defaultdict(Counter)
    candidates = []
    manifest = []
    for file in paths:
        try:
            label = str(file.resolve().relative_to(root))
        except ValueError:
            label = str(file)
        scan_file(file, label, stats, candidates)
        manifest.append(dict(file=label, bytes=file.stat().st_size))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    counts = ["priority_candidates", "direct_fmac", "direct_fma", "bpermute_fmac", "bpermute_add",
              "fmac", "fma", "fma_mix", "pk_fma_f16", "dot", "fmac_dpp", "add_dpp", "mov_dpp", "bpermute"]
    rows = [dict(file=file, function=function, **{name: values[name] for name in counts})
            for (file, function), values in stats.items()]
    rows.sort(key=lambda row: (-row["priority_candidates"], -row["direct_fmac"], -row["fmac"], row["file"], row["function"]))
    candidate_columns = ["file", "function", "category", "status", "producer_line", "consumer_line", "gap", "temporary", "producer", "consumer"]
    candidates.sort(key=lambda row: (not row["category"].startswith("direct_"), row["status"] not in ("review", "operand_swap_review"), row["file"], row["consumer_line"]))
    write_tsv(Path(str(out) + "-functions.tsv"), ["file", "function"] + counts, rows)
    write_tsv(Path(str(out) + "-candidates.tsv"), candidate_columns, candidates)
    write_tsv(Path(str(out) + "-assembly-files.tsv"), ["file", "bytes"], manifest)
    totals = Counter()
    for values in stats.values():
        totals.update(values)
    summary = ["# gfx906 FMA/DPP static ISA scan", "",
               f"- Assembly files: {len(paths)}", f"- Functions: {len(stats)}",
               "- Scope: all generated *gfx906*.s files in the build directory; rocBLAS internals excluded.",
               "- Counts are static occurrences, not kernel execution frequencies.",
               "- No candidate is safe to rewrite without checking lane masks, live ranges and numerical results.", ""]
    for name in counts:
        summary.append(f"- {name}: {totals[name]}")
    summary += ["", "## Top direct candidates", "",
                "| Function | Priority sites | All direct sites | FMAC instructions |",
                "| --- | ---: | ---: | ---: |"]
    for row in rows[:30]:
        if row["direct_fmac"] + row["direct_fma"] == 0:
            continue
        summary.append(f'| `{row["function"][:110]}` | {row["priority_candidates"]} | {row["direct_fmac"] + row["direct_fma"]} | {row["fmac"]} |')
    Path(str(out) + "-summary.md").write_text("\n".join(summary) + "\n", encoding="utf-8")
    print("\n".join(summary[:23]))
    print("TSV and Markdown reports:", out)


if __name__ == "__main__":
    main()
