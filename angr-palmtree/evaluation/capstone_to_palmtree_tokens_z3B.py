import json
import eval_utils as utils

INP = "/home/ru94wid/projects/angr-palmtree/evaluation/tokens/z3_x64-clang-5.0-O3_z3.capstone.json"
OUT = "/home/ru94wid/projects/angr-palmtree/evaluation/tokens/z3_x64-clang-5.0-O3_z3.palmtree.json"

def main():
    d = json.load(open(INP))

    symbol_map = {}
    string_map = {}

    out = {"binary": d["binary"], "arch": d["arch"], "functions": {}}

    for faddr, fdata in d["functions"].items():
        blocks = {}
        for bb_addr, insns in fdata["blocks"].items():
            toks = [utils.parse_instruction(ins, symbol_map, string_map) for ins in insns]
            if toks:
                blocks[bb_addr] = toks

        if blocks:
            out["functions"][faddr] = {"name": fdata["name"], "blocks": blocks}

    with open(OUT, "w") as f:
        json.dump(out, f)

    print(f"Wrote PalmTree tokens to {OUT}")

if __name__ == "__main__":
    main()
