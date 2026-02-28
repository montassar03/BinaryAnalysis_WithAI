import json
from pathlib import Path
import numpy as np

def l2norm(E, eps=1e-12):
    n = np.linalg.norm(E, axis=1, keepdims=True)
    return E / np.maximum(n, eps)

def load_fids(fid_path: Path):
    with fid_path.open("r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]

def main(emb_path: str, pools_path: str, out_path: str):
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

            # Recall@1 / @10: hit if any positive appears in top K
            top1 = ranked[:1]
            top10 = ranked[:10]
            recall1.append(1.0 if any(x in pos for x in top1) else 0.0)
            recall10.append(1.0 if any(x in pos for x in top10) else 0.0)

            # MRR: 1/rank of first relevant
            first = 0
            for r, fid in enumerate(ranked, start=1):
                if fid in pos:
                    first = r
                    break
            rr.append(0.0 if first == 0 else 1.0 / first)
            used += 1

    result = {
        "model": emb_path.name,
        "num_anchors_evaluated": used,
        "Recall@1": float(np.mean(recall1)) if recall1 else 0.0,
        "Recall@10": float(np.mean(recall10)) if recall10 else 0.0,
        "MRR": float(np.mean(rr)) if rr else 0.0,
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
    args = ap.parse_args()
    main(args.emb, args.pools, args.out)
