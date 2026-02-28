import json
import angr

BIN = None
OUT = None

# Choose how strict you want the filter:
# - startswith: sym.name.startswith("Z3_")
# - contains:   "Z3_" in sym.name
USE_CONTAINS = True

MAX_FUNCS = 500  # increase if needed

def main():
    proj = angr.Project(BIN, auto_load_libs=False)

    cfg = proj.analyses.CFGFast(
        normalize=True,
        resolve_indirect_jumps=False,
        indirect_jump_resolvers=[],
    )

    data = {"binary": BIN, "arch": proj.arch.name, "functions": {}}

    kept = 0
    seen = 0

    # Iterate functions; keep those with matching loader symbol name
    for func in cfg.kb.functions.values():
        if getattr(func, "is_plt", False):
            continue

        seen += 1

        sym = proj.loader.find_symbol(func.addr)
        if sym is None or not getattr(sym, "name", None):
            continue

        name = sym.name
        ok = ("Z3_" in name) if USE_CONTAINS else name.startswith("Z3_")
        if not ok:
            continue

        blocks = {}
        for bb_addr in func.block_addrs_set:
            try:
                block = proj.factory.block(bb_addr)
            except Exception:
                continue

            insns = []
            for ci in block.capstone.insns:
                m = ci.insn.mnemonic
                op = ci.insn.op_str.strip()
                raw = f"{m} {op}" if op else m
                insns.append(raw)

            if insns:
                blocks[hex(bb_addr)] = insns

        if blocks:
            data["functions"][hex(func.addr)] = {
                "name": func.name,
                "sym_name": name,
                "blocks": blocks
            }
            kept += 1
            if kept >= MAX_FUNCS:
                break

    with open(OUT, "w") as f:
        json.dump(data, f)

    print(f"Scanned CFG funcs: {seen}")
    print(f"Kept Z3_ symbol funcs: {kept}")
    print(f"Wrote: {OUT}")

if __name__ == "__main__":
    main()
