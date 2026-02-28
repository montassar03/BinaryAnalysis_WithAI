# eval_retrieval_NDCG.py
import json
from pathlib import Path
import numpy as np


def l2norm(E, eps=1e-12):
    n = np.linalg.norm(E, axis=1, keepdims=True)
    return E / np.maximum(n, eps)


def load_fids(fid_path: Path):
    with fid_path.open("r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def dcg_at_k(relevance: np.ndarray) -> float:
    """
    relevance: 1D array of 0/1 relevance values in ranked order (length K)
    DCG = sum(rel_i / log2(i+1)) with i starting at 1
    """
    if relevance.size == 0:
        return 0.0
    i = np.arange(1, relevance.size + 1, dtype=np.float64)
    return float(np.sum(relevance / np.log2(i + 1)))


def ndcg_from_ranked(ranked_fids, pos_set, k=None) -> float:
    """
    Compute NDCG (or NDCG@k if k is set) for binary relevance.
    - ranked_fids: list of candidate fids in ranked order
    - pos_set: set of positive fids
    - k: cutoff (None means use full ranked list)
    """
    if not pos_set:
        return 0.0

    if k is None:
        rel = np.fromiter((1.0 if fid in pos_set else 0.0 for fid in ranked_fids), dtype=np.float64)
        num_pos = len(pos_set)
    else:
        rel = np.fromiter((1.0 if fid in pos_set else 0.0 for fid in ranked_fids[:k]), dtype=np.float64)
        num_pos = min(len(pos_set), k)

    dcg = dcg_at_k(rel)
    if num_pos <= 0:
        return 0.0

    # Ideal ranking: all positives at the top
    ideal_rel = np.ones(num_pos, dtype=np.float64)
    idcg = dcg_at_k(ideal_rel)

    return 0.0 if idcg == 0.0 else float(dcg / idcg)


def main(emb_path: str, pools_path: str, out_path: str, ndcg_k: int = 10):
    emb_path = Path(emb_path)
    pools_path = Path(pools_path)
    out_path = Path(out_path)

    E = np.load(emb_path)
    fids = load_fids(emb_path.with_suffix(".fids.txt"))
    assert len(fids) == E.shape[0], "FID count != embedding rows"

    fid2i = {fid: i for i, fid in enumerate(fids)}
    E = l2norm(E)

    recall1 = []
    recall10 = []
    rr = []
    ndcg_full = []
    ndcg_at_k = []
    used = 0

    with pools_path.open("r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            a = obj["anchor_fid"]
            pos = set(obj["positives"])
            cand = obj["candidates"]

            if a not in fid2i:
                continue

            # keep only candidates present in embeddings
            cidx = []
            cfids = []
            for fid in cand:
                if fid in fid2i:
                    cidx.append(fid2i[fid])
                    cfids.append(fid)

            if not cidx or not pos:
                continue

            a_i = fid2i[a]
            scores = E[cidx] @ E[a_i]  # cosine via dot product
            order = np.argsort(-scores)
            ranked = [cfids[i] for i in order]

            # Recall@1 / @10
            recall1.append(1.0 if ranked and ranked[0] in pos else 0.0)
            top10 = ranked[:10]
            recall10.append(1.0 if any(x in pos for x in top10) else 0.0)

            # MRR
            first = 0
            for r, fid in enumerate(ranked, start=1):
                if fid in pos:
                    first = r
                    break
            rr.append(0.0 if first == 0 else 1.0 / first)

            # NDCG (full) + NDCG@k
            ndcg_full.append(ndcg_from_ranked(ranked, pos, k=None))
            ndcg_at_k.append(ndcg_from_ranked(ranked, pos, k=ndcg_k))

            used += 1

    result = {
        "model": emb_path.name,
        "num_anchors_evaluated": used,
        "Recall@1": float(np.mean(recall1)) if recall1 else 0.0,
        "Recall@10": float(np.mean(recall10)) if recall10 else 0.0,
        "MRR": float(np.mean(rr)) if rr else 0.0,
        "NDCG": float(np.mean(ndcg_full)) if ndcg_full else 0.0,
        f"NDCG@{ndcg_k}": float(np.mean(ndcg_at_k)) if ndcg_at_k else 0.0,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--emb", required=True)
    ap.add_argument("--pools", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ndcg_k", type=int, default=10, help="Cutoff K for NDCG@K (default: 10)")
    args = ap.parse_args()

    main(args.emb, args.pools, args.out, ndcg_k=args.ndcg_k)

