import json
import math
import numpy as np
import eval_utils as utils
from collections import Counter

A_PATH = "/home/ru94wid/projects/angr-palmtree/evaluation/tokens/z3A_Z3sym.palmtree.json"
B_PATH = "/home/ru94wid/projects/angr-palmtree/evaluation/tokens/z3B_Z3sym.palmtree.json"

MODEL_PATH = "/home/ru94wid/projects/PalmTree/cdfg_bert_1/transformerCfgdup.ep19"
VOCAB_PATH  = "/home/ru94wid/projects/PalmTree/cdfg_bert_1/vocab_cfgdup"

TOPK = 5
MIN_TOTAL_INSNS = 10
BATCH_INSNS = 512

def load_funcs(path):
    d = json.load(open(path))
    funcs = []
    for faddr, fdata in d["functions"].items():
        sym = fdata.get("sym_name")
        if not sym:
            continue
        ins = []
        for _, bb_ins in fdata["blocks"].items():
            ins.extend(bb_ins)
        funcs.append((faddr, sym, ins))
    return funcs

def l2_norm(x):
    return x / (np.linalg.norm(x) + 1e-12)

def l2_normalize_rows(mat):
    n = np.linalg.norm(mat, axis=1, keepdims=True) + 1e-12
    return mat / n

def build_idf(corpus_ins_lists):
    # document frequency at "function" granularity
    df = Counter()
    N = len(corpus_ins_lists)
    for ins_list in corpus_ins_lists:
        for s in set(ins_list):
            df[s] += 1
    idf = {}
    for s, c in df.items():
        idf[s] = math.log((N + 1) / (c + 1)) + 1.0
    return idf

def embed_function_tfidf(palmtree, insns, idf):
    if not insns:
        return None
    # encode in chunks
    chunks = []
    for i in range(0, len(insns), BATCH_INSNS):
        chunks.append(palmtree.encode(insns[i:i+BATCH_INSNS]))
    E = np.concatenate(chunks, axis=0).astype(np.float32)  # (N, D)

    w = np.array([idf.get(s, 1.0) for s in insns], dtype=np.float32)  # (N,)
    w_sum = float(w.sum()) + 1e-12
    v = (E * w.reshape(-1, 1)).sum(axis=0) / w_sum
    return v

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

    # IDF built on combined corpora so weights are consistent
    idf = build_idf([ins for _, _, ins in A] + [ins for _, _, ins in B])

    # Embed B once
    B_vecs = []
    for _, _, ins in B:
        v = embed_function_tfidf(palmtree, ins, idf)
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
        v = embed_function_tfidf(palmtree, ins, idf)
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
