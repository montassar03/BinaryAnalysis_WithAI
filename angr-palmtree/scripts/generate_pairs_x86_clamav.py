import os
import angr
from pathlib import Path


def get_function_instructions(proj, func):
    instrs = []
    for block in func.blocks:
        blk = proj.factory.block(block.addr)
        if blk.capstone is None:
            continue
        for insn in blk.capstone.insns:
            if insn.op_str:
                text = f"{insn.mnemonic} {insn.op_str}"
            else:
                text = insn.mnemonic
            text = text.strip()
            if text:
                instrs.append(text)
    return instrs


def write_pairs_for_binary(binary_path: Path, out_path: Path, max_funcs=None):
    print(f"[+] Processing binary: {binary_path}")
    proj = angr.Project(str(binary_path), auto_load_libs=False)
    cfg = proj.analyses.CFGFast()

    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w") as f:
        func_count = 0
        for func in cfg.kb.functions.values():
            func_count += 1
            if max_funcs is not None and func_count > max_funcs:
                break

            instrs = get_function_instructions(proj, func)
            for i in range(len(instrs) - 1):
                a = instrs[i]
                b = instrs[i + 1]
                # PalmTree-style: "<instr_a>\t<instr_b>\n"
                f.write(f"{a}\t{b}\n")


def main():
    dataset_root = Path("~/Dataset-1").expanduser().resolve()
    project = "clamav"
    proj_root = dataset_root / project

    if not proj_root.is_dir():
        raise SystemExit(f"Project folder not found: {proj_root}")

    repo_root = Path(__file__).resolve().parent.parent

    out_root = repo_root / "training" / project
    out_root.mkdir(parents=True, exist_ok=True)

    # Only use x64 binaries like x64-clang-..._clamscan, x64-gcc-..._sigtool, etc.
    binaries = []
    for path in proj_root.rglob("*"):
        if (
            path.is_file()
            and os.access(path, os.X_OK)
            and path.name.startswith("x64-")
        ):
            binaries.append(path)

    if not binaries:
        raise SystemExit(f"No x64 binaries found under {proj_root}")

    print(f"[+] Found {len(binaries)} x64 binaries for project '{project}'")

    max_funcs = None

    for bin_path in sorted(binaries):
        out_path = out_root / f"{bin_path.name}.pairs.txt"
        write_pairs_for_binary(bin_path, out_path, max_funcs=max_funcs)

    print(f"[+] Done. Pairs written under: {out_root}")


if __name__ == "__main__":
    main()
