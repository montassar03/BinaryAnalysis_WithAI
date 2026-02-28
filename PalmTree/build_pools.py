#!/usr/bin/env python3
import json
import re
from dataclasses import dataclass
from pathlib import Path
from collections import defaultdict
import numpy as np

# -------- parsing helpers --------

BIN_RE = re.compile(r"^(?P<prefix>[^_]+)_(?P<prog>.+)$")
# prefix example: x64-gcc-5-O2
# prog example: minigzip

def parse_binary_name(binary_name: str):
    """
    Returns (binary_base, build_id, prefix)
    build_id is derived from the prefix part.
    """
    m = BIN_RE.match(binary_name)
    if not m:
        # fallback: no underscore
        return binary_name, binary_name, binary_name

    prefix = m.group("prefix")   # x64-gcc-5-O2
    prog = m.group("prog")       # minigzip

    # build_id = everything after first '-' (drop arch-ish token)
    parts = prefix.split("-")
    if len(parts) >= 4:
        # x64, gcc, 5, O2
        build_id = "-".join(parts[1:])   # gcc-5-O2
    else:
        build_id = prefix

    return prog, build_id, prefix

def hex_to_int(x: str) -> int:
    return int(x, 16) if isinstance(x, str) and x.startswith("0x") else int(x)

# -------- data structures --------

@dataclass(frozen=True)
class FuncKey:
    project: str
    binary_name: str
    func_addr: str

@dataclass
class FunctionRec:
    fid: str
    project: str
    arch: str
    binary_name: str
    binary_base: str
    build_id: str
    function_name: str
    function_addr: str
    binary_path: str
    # concatenated tokens / sentence representation for embedding later
    tokens: str

# -------- main pipeline --------

def load_blocks(jsonl_path: Path):
    """Reads JSONL where each line is a basic block."""
    blocks = []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            blocks.append(json.loads(line))
    return blocks

def aggregate_functions(blocks):
    """
    Merge basic blocks into a function-level record.
    We sort blocks by block_addr and concatenate instructions.
    """
    grouped = defaultdict(list)
    for b in blocks:
        k = FuncKey(b["project_name"], b["binary_name"], b["function_addr"])
        grouped[k].append(b)

    funcs = []
    for k, blks in grouped.items():
        # Use metadata from first block (should be identical across blocks)
        b0 = blks[0]
        binary_base, build_id, _prefix = parse_binary_name(b0["binary_name"])

        # sort blocks by block_addr to get a stable sequence
        blks_sorted = sorted(blks, key=lambda x: hex_to_int(x["block_addr"]))

        # concatenate insns across blocks; keep simple separator
        insns = []
        for bb in blks_sorted:
            insns.extend(bb.get("insns", []))

        # simple "sentence": join instructions with " ; "
        # (This is NOT your CFG/DFG pair format; it is for embedding-per-function later.)
        tokens = " ; ".join(insns)

        fid = f"{b0['project_name']}|{b0['binary_name']}|{b0['function_addr']}"

        funcs.append(FunctionRec(
            fid=fid,
            project=b0["project_name"],
            arch=b0.get("arch", ""),
            binary_name=b0["binary_name"],
            binary_base=binary_base,
            build_id=build_id,
            function_name=b0.get("function_name", ""),
            function_addr=b0["function_addr"],
            binary_path=b0.get("binary_path", ""),
            tokens=tokens
        ))
    return funcs

def build_anchor_groups(funcs):
    """
    Group by (project, binary_base, function_name). Anchors must have >=2 builds.
    """
    by_key = defaultdict(list)
    for f in funcs:
        by_key[(f.project, f.binary_base, f.function_name)].append(f)
    return by_key

def build_pools(funcs, pool_size=1000, hard_neg=400, seed=0):
    """
    For each valid anchor, create a candidate pool:
      positives: same project+binary_base+function_name, different build_id
      hard negatives: same project+binary_base, different function_name
      easy negatives: different binary_base or different project
    """
    rng = np.random.default_rng(seed)
    by_anchor_key = build_anchor_groups(funcs)

    # Index helpers
    funcs_by_proj_bin = defaultdict(list)  # (project, binary_base) -> [funcs]
    for f in funcs:
        funcs_by_proj_bin[(f.project, f.binary_base)].append(f)

    all_funcs = funcs

    anchors_pools = []
    for (proj, binbase, fname), group in by_anchor_key.items():
        builds = set(g.build_id for g in group)
        if len(builds) < 2:
            continue  # no positives exist -> cannot be anchor group

        # choose anchors: you can choose all, or sample. Here: use all.
        for anchor in group:
            # positives: same group, different build
            positives = [g for g in group if g.build_id != anchor.build_id and g.fid != anchor.fid]
            if not positives:
                continue  # should not happen, but safe

            # hard negatives: same project+binary_base, different function_name
            hard_cands = [f for f in funcs_by_proj_bin[(proj, binbase)]
                          if f.function_name != fname]

            # easy negatives: everything else (different project or different binary_base)
            easy_cands = [f for f in all_funcs
                          if not (f.project == proj and f.binary_base == binbase)]

            # sample negatives to reach pool_size
            hard_n = min(hard_neg, len(hard_cands))
            hard_sample = rng.choice(hard_cands, size=hard_n, replace=False).tolist() if hard_n > 0 else []

            remaining = pool_size - len(positives) - len(hard_sample)
            easy_n = min(max(0, remaining), len(easy_cands))
            easy_sample = rng.choice(easy_cands, size=easy_n, replace=False).tolist() if easy_n > 0 else []

            pool = positives + hard_sample + easy_sample
            # drop any accidental duplicates / anchor itself
            seen = set()
            pool_fids = []
            for f in pool:
                if f.fid == anchor.fid:
                    continue
                if f.fid in seen:
                    continue
                seen.add(f.fid)
                pool_fids.append(f.fid)

            anchors_pools.append({
                "anchor_fid": anchor.fid,
                "anchor": {
                    "project": anchor.project,
                    "binary_base": anchor.binary_base,
                    "binary_name": anchor.binary_name,
                    "build_id": anchor.build_id,
                    "function_name": anchor.function_name,
                    "function_addr": anchor.function_addr,
                },
                "positives": [p.fid for p in positives],
                "candidates": pool_fids  # includes positives + negatives
            })

    return anchors_pools

def write_jsonl(path: Path, rows):
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--blocks_jsonl", required=True, help="Input JSONL (one basic block per line).")
    ap.add_argument("--out_dir", default="eval_build", help="Output directory.")
    ap.add_argument("--pool_size", type=int, default=1000)
    ap.add_argument("--hard_neg", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    blocks = load_blocks(Path(args.blocks_jsonl))
    funcs = aggregate_functions(blocks)

    # write function-level file (useful later for embedding)
    functions_out = []
    for f in funcs:
        functions_out.append({
            "fid": f.fid,
            "project": f.project,
            "arch": f.arch,
            "binary_name": f.binary_name,
            "binary_base": f.binary_base,
            "build_id": f.build_id,
            "function_name": f.function_name,
            "function_addr": f.function_addr,
            "binary_path": f.binary_path,
            "tokens": f.tokens
        })
    write_jsonl(out_dir / "functions.jsonl", functions_out)

    pools = build_pools(funcs, pool_size=args.pool_size, hard_neg=args.hard_neg, seed=args.seed)
    write_jsonl(out_dir / "anchors_pools.jsonl", pools)

    print(f"Wrote {len(funcs)} functions to {out_dir/'functions.jsonl'}")
    print(f"Wrote {len(pools)} anchor pools to {out_dir/'anchors_pools.jsonl'}")
    # quick sanity: how many anchors have >=1 positive
    print("Done.")

if __name__ == "__main__":
    main()

