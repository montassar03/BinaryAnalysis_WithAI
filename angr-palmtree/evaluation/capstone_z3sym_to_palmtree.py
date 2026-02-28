import json
import eval_utils as utils

def convert(inp, out):
    d = json.load(open(inp))
    symbol_map = {}
    string_map = {}

    outd = {"binary": d["binary"], "arch": d["arch"], "functions": {}}

    for faddr, fdata in d["functions"].items():
        blocks = {}
        for bb_addr, insns in fdata["blocks"].items():
            toks = [utils.parse_instruction(ins, symbol_map, string_map) for ins in insns]
            if toks:
                blocks[bb_addr] = toks

        if blocks:
            entry = {
                "name": fdata.get("name", ""),
                "sym_name": fdata.get("sym_name", ""),
                "blocks": blocks,
            }
            outd["functions"][faddr] = entry

    with open(out, "w") as f:
        json.dump(outd, f)

    print(f"Wrote PalmTree tokens to {out}")

def main():
    convert(
        "/home/ru94wid/projects/angr-palmtree/evaluation/tokens/z3A_Z3sym.capstone.json",
        "/home/ru94wid/projects/angr-palmtree/evaluation/tokens/z3A_Z3sym.palmtree.json",
    )
    convert(
        "/home/ru94wid/projects/angr-palmtree/evaluation/tokens/z3B_Z3sym.capstone.json",
        "/home/ru94wid/projects/angr-palmtree/evaluation/tokens/z3B_Z3sym.palmtree.json",
    )

if __name__ == "__main__":
    main()
