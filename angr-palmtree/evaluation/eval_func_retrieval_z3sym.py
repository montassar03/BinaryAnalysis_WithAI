import json
import numpy as np
import eval_utils as utils

A_PATH = "/home/ru94wid/projects/angr-palmtree/evaluation/tokens/z3A_Z3sym.palmtree.json"
B_PATH = "/home/ru94wid/projects/angr-palmtree/evaluation/tokens/z3B_Z3sym.palmtree.json"

MODEL_PATH = "/home/ru94wid/projects/PalmTree/cdfg_bert_1/transformerCfgdup.ep19"
VOCAB_PATH  = "/home/ru94wid/projects/PalmTree/cdfg_bert_1/vocab_cfgdup"

TOPK = 5
MIN_TOTAL_INSNS = 30         # keep, but we can tune later
BATCH_INSNS = 512

def load_funcs(path):
    d = json.load(open(path))
    funcs = []
    for faddr, fdata in d["functions"].items():
        sym = fdata.get("sym_name")
        if not sym:
            continue
        all_ins = []
        for _, insns in fdata["blocks"].items():
            all_ins.extend(insns)
        funcs.append((faddr, sym, all_ins))
    return funcs

def l2_norm(x):
    return x / (np.linalg.norm(x) + 1e-12)

def l2_normalize_rows(mat):
    n = np.linalg.norm(mat, axis=1, keepdims=True) + 1e-12
    return mat / n

def embed_function(palmtree, insns):
    # mean of instruction embeddings across whole function
    if not insns:
        return None
    chunks = []
    for i in range(0, len(insns), BATCH_INSNS):
        chunks.append(palmtree.encode(insns[i:i+BATCH_INSNS]))
    insn_emb = np.concatenate(chunks, axis=0)   # (N, D)
    return np.mean(insn_emb, axis=0).astype(np.float32)

def main():
    palmtree = utils.UsableTransformer(model_path=MODEL_PATH, vocab_path=VOCAB_PATH)

    A = load_funcs(A_PATH)
    B = load_funcs(B_PATH)

    A = [(fa, s, ins) for (fa, s, ins) in A if len(ins) >= MIN_TOTAL_INSNS]
    B = [(fb, s, ins) for (fb, s, ins) in B if len(ins) >= MIN_TOTAL_INSNS]

    A_syms = [s for _, s, _ in A]
    B_syms = [s for _, s, _ in B]

    inter = set(A_syms) & set(B_syms)

    print(f"Sym funcs kept (min_total_insns={MIN_TOTAL_INSNS}): A={len(A)} B={len(B)}")
    print(f"Intersecting sym_name count: {len(inter)}")

    if not inter:
        print("No overlap; lower MIN_TOTAL_INSNS and rerun.")
        return

    # Embed all B once
    B_vecs = []
    for _, _, ins in B:
        v = embed_function(palmtree, ins)
        if v is None:
            v = np.zeros((128,), dtype=np.float32)
        B_vecs.append(v)
    B_vecs = l2_normalize_rows(np.stack(B_vecs, axis=0))

    top1 = 0
    top5 = 0
    evaluated = 0

    for _, sym, ins in A:
        if sym not in inter:
            continue
        v = embed_function(palmtree, ins)
        if v is None:
            continue
        v = l2_norm(v)

        sims = (B_vecs @ v.reshape(-1, 1)).reshape(-1)
        order = np.argsort(-sims)
        top_syms = [B_syms[i] for i in order[:TOPK]]

        evaluated += 1
        if top_syms[0] == sym:
            top1 += 1
        if sym in top_syms:
            top5 += 1

    print(f"Evaluated queries: {evaluated}")
    print(f"Top-1 accuracy: {top1/evaluated:.4f} ({top1}/{evaluated})")
    print(f"Top-{TOPK} accuracy: {top5/evaluated:.4f} ({top5}/{evaluated})")

if __name__ == "__main__":
    main()
