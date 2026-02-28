import json
import os
from pathlib import Path
import numpy as np
import torch

from palmtree.dataset import vocab as vocab_mod
from palmtree.model.bert import BERT

# -------- config you will edit via CLI args --------
import argparse

def load_vocab(vocab_path: Path):
    return vocab_mod.WordVocab.load_vocab(str(vocab_path))

def load_checkpoint(model: torch.nn.Module, ckpt_path: Path):
    ckpt = torch.load(str(ckpt_path), map_location="cpu")

    # PalmTree saves the full BERT module (pickled)
    if isinstance(ckpt, torch.nn.Module):
        model.load_state_dict(ckpt.state_dict(), strict=False)
        return

    # fallback: plain state_dict
    if isinstance(ckpt, dict) and len(ckpt) > 0 and all(isinstance(v, torch.Tensor) for v in ckpt.values()):
        model.load_state_dict(ckpt, strict=False)
        return

    raise RuntimeError(f"Unrecognized checkpoint format in: {ckpt_path} (type={type(ckpt)})")


def tokenize_to_ids(vocab, text: str, seq_len: int):
    # your stored tokens are "insn ; insn ; ..."
    text = text.replace(";", " ")
    toks = text.split()

    ids = [vocab.sos_index] + [vocab.stoi.get(t, vocab.unk_index) for t in toks] + [vocab.eos_index]
    ids = ids[:seq_len]
    if len(ids) < seq_len:
        ids = ids + [vocab.pad_index] * (seq_len - len(ids))

    # segment label: all 1s (single sequence)
    seg = [1] * seq_len
    return ids, seg

def extract_cls(output):
    """
    Try to robustly get last hidden states and return CLS vector.
    """
    if isinstance(output, torch.Tensor):
        h = output
    elif isinstance(output, (list, tuple)) and len(output) > 0 and isinstance(output[0], torch.Tensor):
        h = output[0]
    elif isinstance(output, dict):
        # try common keys
        for k in ["hidden_states", "last_hidden", "x", "output"]:
            if k in output and isinstance(output[k], torch.Tensor):
                h = output[k]
                break
        else:
            # fallback: first tensor value
            h = next(v for v in output.values() if isinstance(v, torch.Tensor))
    else:
        raise RuntimeError(f"Unsupported model output type: {type(output)}")

    # Expect [B, T, H]
    if h.dim() == 2:
        # [B, H] already
        return h
    return h[:, 0, :]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--functions", required=True, help="evaluation/functions.jsonl")
    ap.add_argument("--vocab", required=True, help="Path to vocab file (e.g., FullTraining_All/vocab or FullTraining/vocab)")
    ap.add_argument("--ckpt", required=True, help="Checkpoint file (.pt/.pth/.tar)")
    ap.add_argument("--out", required=True, help="Output .npy path (embeddings)")
    ap.add_argument("--seq_len", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=512)
    args = ap.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    assert device.type == "cuda", "CUDA not available in this env."

    vocab = load_vocab(Path(args.vocab))

    # IMPORTANT: match your training hyperparams
    model = BERT(len(vocab), hidden=128, n_layers=12, attn_heads=8, dropout=0.0).to(device)
    model.eval()
    load_checkpoint(model, Path(args.ckpt))

    fids = []
    all_vecs = []

    with open(args.functions, "r", encoding="utf-8") as f:
        batch_ids, batch_seg, batch_fids = [], [], []
        for line in f:
            obj = json.loads(line)
            fid = obj["fid"]
            ids, seg = tokenize_to_ids(vocab, obj["tokens"], args.seq_len)

            batch_ids.append(ids)
            batch_seg.append(seg)
            batch_fids.append(fid)

            if len(batch_ids) >= args.batch_size:
                x = torch.tensor(batch_ids, dtype=torch.long, device=device)
                s = torch.tensor(batch_seg, dtype=torch.long, device=device)
                with torch.no_grad():
                    out = model(x, s)  # PalmTree BERT usually takes (x, segment)
                    cls = extract_cls(out).detach().cpu().numpy()
                all_vecs.append(cls)
                fids.extend(batch_fids)
                batch_ids, batch_seg, batch_fids = [], [], []

        # flush last batch
        if batch_ids:
            x = torch.tensor(batch_ids, dtype=torch.long, device=device)
            s = torch.tensor(batch_seg, dtype=torch.long, device=device)
            with torch.no_grad():
                out = model(x, s)
                cls = extract_cls(out).detach().cpu().numpy()
            all_vecs.append(cls)
            fids.extend(batch_fids)

    E = np.vstack(all_vecs)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    np.save(out_path, E)

    # save fid order alongside embeddings
    with open(out_path.with_suffix(".fids.txt"), "w", encoding="utf-8") as w:
        for fid in fids:
            w.write(fid + "\n")

    print("Saved embeddings:", out_path, "shape=", E.shape)
    print("Saved fids:", out_path.with_suffix(".fids.txt"))

if __name__ == "__main__":
    main()
