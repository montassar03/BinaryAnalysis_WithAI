import json
import angr

BIN = "/home/ru94wid/Dataset-1/z3/x64-clang-5.0-O3_z3"
OUT = "/home/ru94wid/projects/angr-palmtree/evaluation/tokens/z3_x64-clang-5.0-O3_z3.capstone.json"

def main():
    proj = angr.Project(BIN, auto_load_libs=False)

    cfg = proj.analyses.CFGFast(
        normalize=True,
        resolve_indirect_jumps=False,
        indirect_jump_resolvers=[],
    )

    data = {"binary": BIN, "arch": proj.arch.name, "functions": {}}

    max_functions = 200
    fn_count = 0

    for func in cfg.kb.functions.values():
        if getattr(func, "is_plt", False):
            continue

        fn_entry = hex(func.addr)
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
            data["functions"][fn_entry] = {"name": func.name, "blocks": blocks}
            fn_count += 1
            if fn_count >= max_functions:
                break

    with open(OUT, "w") as f:
        json.dump(data, f)

    print(f"Wrote {fn_count} functions to {OUT}")

if __name__ == "__main__":
    main()
