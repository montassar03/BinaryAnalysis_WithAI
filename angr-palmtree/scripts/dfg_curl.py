#!/usr/bin/env python3
import os
import re
import gc
import random
import logging
import traceback
from typing import Dict, List, Optional, Tuple

import angr
import networkx as nx
from tqdm import tqdm
from angr.errors import SimMemoryError


# -----------------------------
# CONFIG: change per script
# -----------------------------
DATASET_NAME = "curl"  # clamav | curl | nmap | openssl | unrar

DATASET_ROOT = os.path.expanduser("~/Dataset-1")
OUT_ROOT = os.path.expanduser("~/projects/angr-palmtree/training/dfg")

WALK_LEN = 40
SKIP_IF_OUTPUT_EXISTS = True

# If a binary crashes angr (segfault), you can manually add it here to skip it.
MANUAL_SKIP = set([
    # "x64-....",
])

# Log file for progress (so you can see last processed binary even if segfault happens)
LOG_PATH = os.path.join(OUT_ROOT, f"{DATASET_NAME}.run.log")
CRASH_LIST_PATH = os.path.join(OUT_ROOT, f"{DATASET_NAME}.crashed.txt")

# Reduce log spam from angr/cle
logging.getLogger("angr").setLevel(logging.CRITICAL)
logging.getLogger("angr.analyses.reaching_definitions.engine_vex.SimEngineRDVEX").setLevel(logging.CRITICAL)
logging.getLogger("angr.knowledge_plugins.key_definitions.live_definitions").setLevel(logging.CRITICAL)
logging.getLogger("cle.loader").setLevel(logging.CRITICAL)


# -----------------------------
# Utilities
# -----------------------------

def log_line(msg: str) -> None:
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    with open(LOG_PATH, "a", encoding="utf-8") as fp:
        fp.write(msg.rstrip() + "\n")


def append_crash(name: str, path: str) -> None:
    with open(CRASH_LIST_PATH, "a", encoding="utf-8") as fp:
        fp.write(f"{name}\t{path}\n")


# -----------------------------
# Normalization
# -----------------------------

def parse_instruction(ins: str, symbol_map: Dict[int, str], string_map: Dict[int, str]) -> str:
    if not ins:
        return ""
    ins = re.sub(r"\s+", ", ", ins, 1)
    parts = ins.split(", ")
    opcode = parts[0].strip()
    operands = parts[1:] if len(parts) > 1 else []

    for i in range(len(operands)):
        tokens = re.split(r"([0-9A-Za-z]+)", operands[i])
        for j in range(len(tokens)):
            tok = tokens[j]
            if tok.startswith("0x") and len(tok) >= 6:
                try:
                    addr = int(tok, 16)
                except ValueError:
                    continue
                if addr in symbol_map:
                    tokens[j] = "symbol"
                elif addr in string_map:
                    tokens[j] = "string"
                else:
                    tokens[j] = "address"
        operands[i] = " ".join(t for t in tokens if t)

    return " ".join([opcode] + operands).strip()


def random_walk(g: nx.DiGraph, length: int, symbol_map: Dict[int, str], string_map: Dict[int, str]) -> List[List[str]]:
    sequences: List[List[str]] = []
    for n in g.nodes:
        if n == -1:
            continue
        text = g.nodes[n].get("text")
        if not text:
            continue

        seq = [parse_instruction(text, symbol_map, string_map)]
        cur = n
        steps = 0

        while steps < length:
            nbs = list(g.successors(cur))
            if not nbs:
                break
            cur = random.choice(nbs)
            nxt_text = g.nodes[cur].get("text")
            if not nxt_text:
                break
            seq.append(parse_instruction(nxt_text, symbol_map, string_map))
            steps += 1

        sequences.append(seq)
    return sequences


# -----------------------------
# angr helpers
# -----------------------------

def build_symbol_map(proj: angr.Project) -> Dict[int, str]:
    sm: Dict[int, str] = {}
    for sym in proj.loader.main_object.symbols:
        if sym.rebased_addr is not None and sym.name:
            sm[int(sym.rebased_addr)] = sym.name
    return sm


def build_string_map(proj: angr.Project, min_len: int = 4) -> Dict[int, str]:
    string_map: Dict[int, str] = {}

    def is_printable(b: int) -> bool:
        return 32 <= b <= 126

    for sec in proj.loader.main_object.sections:
        if not sec.is_readable:
            continue
        try:
            data = proj.loader.memory.load(sec.vaddr, sec.memsize)
        except Exception:
            continue

        base = int(sec.vaddr)
        i = 0
        n = len(data)

        while i < n:
            if is_printable(data[i]):
                start = i
                while i < n and is_printable(data[i]):
                    i += 1
                if i - start >= min_len:
                    raw = data[start:i].split(b"\x00", 1)[0]
                    try:
                        s = raw.decode("utf-8", errors="ignore")
                    except Exception:
                        s = ""
                    if s:
                        string_map[base + start] = s
            else:
                i += 1

    return string_map


def disasm_insn(proj: angr.Project, addr: int) -> Optional[str]:
    try:
        block = proj.factory.block(addr, size=16, opt_level=0)
        if not block.capstone or not block.capstone.insns:
            return None
        insn = block.capstone.insns[0]
        return f"{insn.mnemonic} {insn.op_str}".strip()
    except Exception:
        return None


# -----------------------------
# DFG construction: RD best-effort + fallback
# -----------------------------

def build_function_dfg(proj: angr.Project, func_addr: int) -> Optional[nx.DiGraph]:
    func = proj.kb.functions.get(func_addr, None)
    if func is None or func.is_plt or func.is_simprocedure:
        return None

    block_insns: Dict[int, List[int]] = {}
    ins_set = set()

    try:
        for block in func.blocks:
            ins_addrs: List[int] = []
            try:
                for insn in block.capstone.insns:
                    a = int(insn.address)
                    ins_addrs.append(a)
                    ins_set.add(a)
            except Exception:
                pass
            if ins_addrs:
                block_insns[int(block.addr)] = ins_addrs
    except Exception:
        return None

    if len(ins_set) < 2:
        return None

    G = nx.DiGraph()
    G.add_node(-1, text="entry_point")
    for a in ins_set:
        G.add_node(a, text=disasm_insn(proj, a))

    edges: set[Tuple[int, int]] = set()

    rd_ok = True
    try:
        rd = proj.analyses.ReachingDefinitions(
            subject=func,
            func_graph=func.graph,
            observe_all=True,
        )

        def obs_to_ins_addr(obs) -> Optional[int]:
            try:
                if hasattr(obs, "ins_addr") and obs.ins_addr is not None:
                    return int(obs.ins_addr)
            except Exception:
                pass
            if isinstance(obs, tuple):
                for item in reversed(obs):
                    if isinstance(item, int) and item in ins_set:
                        return int(item)
            return None

        for obs, state in rd.observed_results.items():
            use_addr = obs_to_ins_addr(obs)
            if use_addr is None or use_addr not in ins_set:
                continue

            defs_iter = []
            try:
                defs_container = getattr(state, "definitions", None)
                if isinstance(defs_container, dict):
                    defs_iter = defs_container.values()
                elif defs_container is not None:
                    defs_iter = list(defs_container)
            except Exception:
                defs_iter = []

            for d in defs_iter:
                try:
                    codeloc = getattr(d, "codeloc", None)
                    def_ins_addr = getattr(codeloc, "ins_addr", None) if codeloc is not None else None
                    if def_ins_addr is None:
                        continue
                    def_ins_addr = int(def_ins_addr)
                    if def_ins_addr in ins_set and def_ins_addr != use_addr:
                        edges.add((def_ins_addr, use_addr))
                except Exception:
                    continue

    except (SimMemoryError, Exception):
        rd_ok = False

    # fallback if RD failed or yielded nothing
    if (not rd_ok) or (len(edges) == 0):
        for _, ins_addrs in block_insns.items():
            for i in range(1, len(ins_addrs)):
                edges.add((ins_addrs[i - 1], ins_addrs[i]))

        try:
            for src, dst in func.graph.edges():
                src_addr = int(src.addr) if hasattr(src, "addr") else int(src)
                dst_addr = int(dst.addr) if hasattr(dst, "addr") else int(dst)
                if src_addr in block_insns and dst_addr in block_insns:
                    edges.add((block_insns[src_addr][-1], block_insns[dst_addr][0]))
        except Exception:
            pass

    if edges:
        G.add_edges_from(edges)

    for node in list(G.nodes):
        if node == -1:
            continue
        if G.in_degree(node) == 0:
            G.add_edge(-1, node)

    if len(G.nodes) <= 2:
        return None

    return G


# -----------------------------
# Processing
# -----------------------------

def is_x64_candidate(name: str) -> bool:
    return name.startswith("x64-")


def process_binary(bin_path: str, out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    if SKIP_IF_OUTPUT_EXISTS and os.path.exists(out_path) and os.path.getsize(out_path) > 0:
        return

    proj = angr.Project(bin_path, auto_load_libs=False)

    symbol_map = build_symbol_map(proj)
    string_map = build_string_map(proj)

    proj.analyses.CFGFast(normalize=True, data_references=True)

    with open(out_path, "w", encoding="utf-8") as w:
        for func_addr in proj.kb.functions.keys():
            try:
                G = build_function_dfg(proj, func_addr)
            except Exception:
                continue
            if G is None:
                continue

            sequences = random_walk(G, WALK_LEN, symbol_map, string_map)
            for s in sequences:
                if len(s) < 2:
                    continue
                for i in range(1, len(s)):
                    w.write(s[i - 1] + "\t" + s[i] + "\n")

    gc.collect()


def main():
    in_dir = os.path.join(DATASET_ROOT, DATASET_NAME)
    out_dir = os.path.join(OUT_ROOT, DATASET_NAME)

    if not os.path.isdir(in_dir):
        raise RuntimeError(f"Dataset folder not found: {in_dir}")

    os.makedirs(out_dir, exist_ok=True)

    binaries: List[Tuple[str, str]] = []
    for name in os.listdir(in_dir):
        if name in MANUAL_SKIP:
            continue
        p = os.path.join(in_dir, name)
        if os.path.isfile(p) and is_x64_candidate(name):
            binaries.append((name, p))

    binaries.sort(key=lambda x: x[0])

    print(f"[+] {DATASET_NAME}: {len(binaries)} x64 binaries")
    print(f"[+] Output dir: {out_dir}")
    log_line(f"=== START {DATASET_NAME} ===")
    log_line(f"Input: {in_dir}")
    log_line(f"Output: {out_dir}")
    log_line(f"Count: {len(binaries)}")

    for name, bp in tqdm(binaries, desc=f"DFG({DATASET_NAME})"):
        out_path = os.path.join(out_dir, f"{name}.dfg.txt")

        # Write progress marker BEFORE processing so segfault still leaves a clue
        log_line(f"BEGIN\t{name}\t{bp}")

        try:
            process_binary(bp, out_path)
        except Exception as e:
            # Python exceptions (not segfaults) land here
            log_line(f"FAIL\t{name}\t{bp}\t{repr(e)}")
            append_crash(name, bp)
            continue

        log_line(f"OK\t{name}\t{bp}")

    log_line(f"=== END {DATASET_NAME} ===")
    print("[+] Done.")


if __name__ == "__main__":
    main()
