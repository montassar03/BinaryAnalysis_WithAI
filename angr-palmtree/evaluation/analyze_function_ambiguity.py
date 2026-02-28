import json
from collections import Counter, defaultdict

A_PATH = "/home/ru94wid/projects/angr-palmtree/evaluation/tokens/z3A_Z3sym.palmtree.json"
B_PATH = "/home/ru94wid/projects/angr-palmtree/evaluation/tokens/z3B_Z3sym.palmtree.json"

MIN_TOTAL_INSNS = 10

def load_signatures(path):
    d = json.load(open(path))
    sigs = []
    for faddr, fdata in d["functions"].items():
        sym = fdata.get("sym_name")
        if not sym:
            continue
        ins = []
        for _, bb_ins in fdata["blocks"].items():
            ins.extend(bb_ins)
        if len(ins) < MIN_TOTAL_INSNS:
            continue

        # Signature options:
        # 1) exact token sequence (very strict)
        # sig = tuple(ins)

        # 2) multiset/bag of instructions (orderless, more collisions)
        # sig = tuple(sorted(ins))

        # 3) opcode-only (very lossy)
        # sig = tuple(i.split(' ')[0] for i in ins)

        # Use 1) but truncated to reduce huge funcs and focus on wrappers:
        sig = tuple(ins[:40])

        sigs.append((sym, faddr, sig, len(ins)))
    return sigs

def main():
    A = load_signatures(A_PATH)
    B = load_signatures(B_PATH)

    # Compute duplicate rates within each side
    A_cnt = Counter(sig for _,_,sig,_ in A)
    B_cnt = Counter(sig for _,_,sig,_ in B)

    def summary(label, items, cnt):
        total = len(items)
        dup = sum(1 for _,_,sig,_ in items if cnt[sig] > 1)
        print(f"{label}: total funcs={total}, funcs with duplicate signature={dup} ({dup/total:.3f})")
        top = cnt.most_common(10)
        print(f"{label}: top-10 most frequent signatures counts:", [c for _,c in top])

    summary("A", A, A_cnt)
    summary("B", B, B_cnt)

    # Cross-side: how many signatures are shared (and how many-to-many)
    A_map = defaultdict(list)
    for sym,faddr,sig,n in A:
        A_map[sig].append((sym,faddr,n))
    B_map = defaultdict(list)
    for sym,faddr,sig,n in B:
        B_map[sig].append((sym,faddr,n))

    shared = set(A_map.keys()) & set(B_map.keys())
    print(f"\nShared signatures A∩B: {len(shared)}")

    # Show a few ambiguous shared signatures
    amb = []
    for sig in shared:
        if len(A_map[sig]) > 1 or len(B_map[sig]) > 1:
            amb.append((len(A_map[sig]), len(B_map[sig]), sig))
    amb.sort(reverse=True)
    print(f"Ambiguous shared signatures (many-to-many): {len(amb)}")
    for a_ct,b_ct,sig in amb[:5]:
        print("\n--- example ambiguous signature ---")
        print("A matches:", a_ct, "B matches:", b_ct)
        print("A sample:", A_map[sig][:5])
        print("B sample:", B_map[sig][:5])
        print("signature first 8 insns:", list(sig[:8]))

