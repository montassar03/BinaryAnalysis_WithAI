import json
import angr

INP = "/home/ru94wid/projects/angr-palmtree/evaluation/tokens/z3_x64-gcc-7-O2_z3.capstone.json"
OUT = "/home/ru94wid/projects/angr-palmtree/evaluation/tokens/z3_x64-gcc-7-O2_z3.capstone.sym.json"

def main():
    d = json.load(open(INP))
    bin_path = d["binary"]

    proj = angr.Project(bin_path, auto_load_libs=False)

    added = 0
    total = 0

    for faddr, fdata in d["functions"].items():
        total += 1
        try:
            addr_int = int(faddr, 16)
        except Exception:
            continue

        sym = proj.loader.find_symbol(addr_int)
        if sym is not None and getattr(sym, "name", None):
            fdata["sym_name"] = sym.name
            added += 1

    with open(OUT, "w") as f:
        json.dump(d, f)

    print(f"Wrote annotated capstone JSON: {OUT}")
    print(f"Functions: {total}, sym_name added: {added}")

if __name__ == "__main__":
    main()
