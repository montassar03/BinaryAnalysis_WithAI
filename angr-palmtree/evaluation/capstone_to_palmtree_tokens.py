import json
import eval_utils as utils

INP = "/home/ru94wid/projects/angr-palmtree/evaluation/tokens/z3_x64-gcc-7-O2_z3.capstone.json"
OUT = "/home/ru94wid/projects/angr-palmtree/evaluation/tokens/z3_x64-gcc-7-O2_z3.palmtree.json"

def main():
    d = json.load(open(INP))

    # For now, empty maps are fine; symbols improve quality but are optional
    symbol_map = {}
    string_map = {}

    out = {
        "binary": d["binary"],
        "arch": d["arch"],
        "functions": {}
    }

    for faddr, fdata in d["functions"].items():
        blocks = {}
        for bb_addr, insns in fdata["blocks"].items():
            toks = []
            for ins in insns:
                t = utils.parse_instruction(ins, symbol_map, string_map)
                toks.append(t)
            if toks:
                blocks[bb_addr] = toks

        if blocks:
            out["functions"][faddr] = {
                "name": fdata["name"],
                "blocks": blocks
            }

    with open(OUT, "w") as f:
        json.dump(out, f)

    print(f"Wrote PalmTree tokens to {OUT}")

if __name__ == "__main__":
    main()
