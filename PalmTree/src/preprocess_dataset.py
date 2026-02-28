import os
import random
import torch
import tqdm
from palmtree.dataset import WordVocab

# -----------------------------
# CONFIG
# -----------------------------
SEQ_LEN = 20
SHARD_SIZE = 200_000         # safe starting point
MAX_SAMPLES = 40_000_000     # cap if you want

VOCAB_PATH = "../FullTraining/vocab"
DFG_PATH   = "../data/training/FullTraining/dfg_train.txt"
CFG_PATH   = "../data/training/FullTraining/cfg_train.txt"
OUT_DIR    = "../preprocessed_shards"

# For reproducibility if desired
# random.seed(1234)

def random_word(sentence_tokens, vocab):
    # sentence_tokens: list[str]
    tokens = sentence_tokens[:]  # copy
    output_label = []
    for i, token in enumerate(tokens):
        prob = random.random()
        if prob < 0.15:
            prob /= 0.15
            if prob < 0.8:
                tokens[i] = vocab.mask_index
            elif prob < 0.9:
                tokens[i] = random.randrange(len(vocab))
            else:
                tokens[i] = vocab.stoi.get(token, vocab.unk_index)
            output_label.append(vocab.stoi.get(token, vocab.unk_index))
        else:
            tokens[i] = vocab.stoi.get(token, vocab.unk_index)
            output_label.append(0)
    return tokens, output_label

def build_sample(cfg1, cfg2, dfg1, dfg2, vocab):
    # cfg1/cfg2 are strings; dfg1/dfg2 are strings
    # Mimics __getitem__ logic but for already-chosen pairs

    d1_tokens = dfg1.split()
    d2_tokens = dfg2.split()

    d1_random, d1_label = random_word(d1_tokens, vocab)
    d2_random, d2_label = random_word(d2_tokens, vocab)

    d1 = [vocab.sos_index] + d1_random + [vocab.eos_index]
    d2 = d2_random + [vocab.eos_index]

    c1 = [vocab.sos_index] + [vocab.stoi.get(c, vocab.unk_index) for c in cfg1.split()] + [vocab.eos_index]
    c2 = [vocab.stoi.get(c, vocab.unk_index) for c in cfg2.split()] + [vocab.eos_index]

    d1_label = [vocab.pad_index] + d1_label + [vocab.pad_index]
    d2_label = d2_label + [vocab.pad_index]

    dfg_segment_label = ([1] * len(d1) + [2] * len(d2))[:SEQ_LEN]
    cfg_segment_label = ([1] * len(c1) + [2] * len(c2))[:SEQ_LEN]

    dfg_bert_input = (d1 + d2)[:SEQ_LEN]
    dfg_bert_label = (d1_label + d2_label)[:SEQ_LEN]
    cfg_bert_input = (c1 + c2)[:SEQ_LEN]

    # padding
    pad = vocab.pad_index
    if len(dfg_bert_input) < SEQ_LEN:
        padding = [pad] * (SEQ_LEN - len(dfg_bert_input))
        dfg_bert_input += padding
        dfg_bert_label += padding
        dfg_segment_label += padding

    if len(cfg_bert_input) < SEQ_LEN:
        cfg_padding = [pad] * (SEQ_LEN - len(cfg_bert_input))
        cfg_bert_input += cfg_padding
        cfg_segment_label += cfg_padding

    return {
        "dfg_bert_input": torch.tensor(dfg_bert_input, dtype=torch.long),
        "dfg_bert_label": torch.tensor(dfg_bert_label, dtype=torch.long),
        "dfg_segment_label": torch.tensor(dfg_segment_label, dtype=torch.long),
        "cfg_bert_input": torch.tensor(cfg_bert_input, dtype=torch.long),
        "cfg_segment_label": torch.tensor(cfg_segment_label, dtype=torch.long),
    }

def save_shard(buf, out_path):
    # buf: list[dict[str, Tensor(SEQ_LEN)]], add labels separately
    keys = buf[0].keys()
    stacked = {k: torch.stack([b[k] for b in buf], dim=0) for k in keys}
    torch.save(stacked, out_path)

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    vocab = WordVocab.load_vocab(VOCAB_PATH)

    # Open both files streaming
    with open(CFG_PATH, "r", encoding="utf-8") as cfg_f, open(DFG_PATH, "r", encoding="utf-8") as dfg_f:
        shard_buf = []
        shard_id = 0
        total = 0

        # We need random negative CFG line (get_random_line). In streaming mode,
        # we maintain a reservoir buffer of cfg second sentences.
        reservoir = []
        RESERVOIR_MAX = 200_000  # tune: more = better randomness, more RAM
        # This stores only strings, not full dataset. Manageable.

        pbar = tqdm.tqdm(desc="Preprocessing", unit="samples")

        for cfg_line, dfg_line in zip(cfg_f, dfg_f):
            if total >= MAX_SAMPLES:
                break

            cfg_parts = cfg_line.rstrip("\n").split("\t")
            dfg_parts = dfg_line.rstrip("\n").split("\t")
            if len(cfg_parts) < 2 or len(dfg_parts) < 2:
                continue

            c1, c2 = cfg_parts[0], cfg_parts[1]
            d1, d2 = dfg_parts[0], dfg_parts[1]

            # update reservoir with c2 for negatives
            reservoir.append(c2)
            if len(reservoir) > RESERVOIR_MAX:
                # simple FIFO drop; reservoir sampling can be added if desired
                reservoir.pop(0)

            # random_sent dice logic
            dice = random.random()
            if dice > 0.25:
                cfg1, cfg2, cfg_is_next = c1, c2, 1
                dfg1, dfg2, dfg_is_next = d1, d2, 1
            elif dice < 0.5:
                # negative cfg2 from reservoir if available
                neg = random.choice(reservoir) if reservoir else c2
                cfg1, cfg2, cfg_is_next = c1, neg, 0
                dfg1, dfg2, dfg_is_next = d1, d2, 1
            elif dice < 0.75:
                cfg1, cfg2, cfg_is_next = c1, c2, 1
                dfg1, dfg2, dfg_is_next = d2, d1, 0
            else:
                neg = random.choice(reservoir) if reservoir else c2
                cfg1, cfg2, cfg_is_next = c1, neg, 0
                dfg1, dfg2, dfg_is_next = d2, d1, 0

            sample = build_sample(cfg1, cfg2, dfg1, dfg2, vocab)
            sample["cfg_is_next"] = torch.tensor(cfg_is_next, dtype=torch.long)
            sample["dfg_is_next"] = torch.tensor(dfg_is_next, dtype=torch.long)

            shard_buf.append(sample)
            total += 1
            pbar.update(1)

            if len(shard_buf) >= SHARD_SIZE:
                out_path = os.path.join(OUT_DIR, f"shard_{shard_id:05d}.pt")
                save_shard(shard_buf, out_path)
                shard_buf.clear()
                shard_id += 1

        # final shard
        if shard_buf:
            out_path = os.path.join(OUT_DIR, f"shard_{shard_id:05d}.pt")
            save_shard(shard_buf, out_path)

        pbar.close()
        print(f"Done. Wrote {total:,} samples into {OUT_DIR}")

if __name__ == "__main__":
    main()

