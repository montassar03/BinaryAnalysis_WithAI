import os
import glob
import bisect
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.autograd import Variable
from config import *
import numpy as np
import palmtree
from palmtree import dataset
from palmtree import trainer
import pickle as pkl
from palmtree.model.bert import BERT


# -----------------------------
# GPU sanity
# -----------------------------
assert torch.cuda.is_available(), "CUDA is not available."
print("CUDA_VISIBLE_DEVICES =", os.environ.get("CUDA_VISIBLE_DEVICES"))
print("CUDA device count (visible) =", torch.cuda.device_count())
print("Using device 0 name =", torch.cuda.get_device_name(0))
print("Training on GPU:", torch.cuda.get_device_name(0))

print("palmtree package:", palmtree.__file__)

# -----------------------------
# paths
# -----------------------------
vocab_path = "FullTraining/vocab"
train_cfg_dataset = "data/training/FullTraining/cfg_train.txt"
train_dfg_dataset = "data/training/FullTraining/dfg_train.txt"
test_dataset = None
sent_dataset = "data/sentence.pkl"
output_path = "FullTraining/transformer"

# IMPORTANT: this is your preprocessed data location (shards)
SHARD_DIR = "preprocessed_shards"   # change if your directory name differs


# -----------------------------
# Build/load vocab (unchanged)
# -----------------------------
if not os.path.exists(vocab_path):
    print("Vocab not found -> building vocab from training corpora...")
    with open(train_cfg_dataset, "r", encoding="utf-8") as f1, open(train_dfg_dataset, "r", encoding="utf-8") as f2:
        vocab = dataset.WordVocab([f1, f2], max_size=13000, min_freq=1)
    print("VOCAB SIZE:", len(vocab))
    os.makedirs(os.path.dirname(vocab_path), exist_ok=True)
    vocab.save_vocab(vocab_path)

print("Loading Vocab", vocab_path)
vocab = dataset.WordVocab.load_vocab(vocab_path)
print("Vocab Size: ", len(vocab))


# -----------------------------
# Preprocessed shard dataset (minimal addition)
# -----------------------------
class PalmTreeShardDataset(Dataset):
    """
    Loads preprocessed shards saved as torch.save({key: Tensor[N, ...], ...})
    and returns per-sample dict[str, Tensor].
    Keeps only one shard in memory at a time (per worker).
    """
    def __init__(self, shard_dir: str):
        self.paths = sorted(glob.glob(os.path.join(shard_dir, "shard_*.pt")))
        if not self.paths:
            raise RuntimeError(f"No shards found in '{shard_dir}' (expected shard_*.pt).")

        # Fast init if sizes.json exists (optional)
        sizes_path = os.path.join(shard_dir, "sizes.json")
        if os.path.exists(sizes_path):
            with open(sizes_path, "r") as f:
                self.sizes = json.load(f)["sizes"]
            if len(self.sizes) != len(self.paths):
                raise RuntimeError("sizes.json count does not match shard file count.")
        else:
            self.sizes = []
            for p in self.paths:
                d = torch.load(p, map_location="cpu", weights_only=True)
                n = next(iter(d.values())).shape[0]
                self.sizes.append(n)

        self.prefix = [0]
        for n in self.sizes:
            self.prefix.append(self.prefix[-1] + n)

        self._cur_shard_idx = None
        self._cur_data = None

    def __len__(self):
        return self.prefix[-1]

    def _load_shard(self, shard_idx: int):
        self._cur_data = torch.load(self.paths[shard_idx], map_location="cpu", weights_only=True)
        self._cur_shard_idx = shard_idx

    def __getitem__(self, idx: int):
        shard_idx = bisect.bisect_right(self.prefix, idx) - 1
        if shard_idx != self._cur_shard_idx:
            self._load_shard(shard_idx)

        local = idx - self.prefix[shard_idx]
        return {k: v[local] for k, v in self._cur_data.items()}


def collate_dict(batch):
    # batch: list[dict[str, Tensor]]
    out = {}
    for k in batch[0].keys():
        out[k] = torch.stack([b[k] for b in batch], dim=0)
    return out


print("Loading preprocessed shard dataset from:", SHARD_DIR)
train_dataset = PalmTreeShardDataset(SHARD_DIR)

print("Creating Dataloader")
train_data_loader = DataLoader(
    train_dataset,
    batch_size=512,              # start higher; H100 can handle this easily at seq_len=20
    shuffle=False,               # IMPORTANT: avoids shard thrashing; see note below
    num_workers=12,               # raise workers to feed GPU
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=4,
    collate_fn=collate_dict
)

test_data_loader = None


# -----------------------------
# Model + trainer (unchanged)
# -----------------------------
print("Building BERT model")
bert = BERT(len(vocab), hidden=128, n_layers=12, attn_heads=8, dropout=0.0)

print("Creating BERT Trainer")
bert_trainer = trainer.BERTTrainer(
    bert,
    len(vocab),
    train_dataloader=train_data_loader,
    test_dataloader=test_data_loader,
    lr=1e-5,
    betas=(0.9, 0.999),
    weight_decay=0.0,
    with_cuda=True,
    cuda_devices=[0],
    log_freq=100
)

print("Training Start")
for epoch in range(10):
    bert_trainer.train(epoch)
    bert_trainer.save(epoch, output_path)
    print("max allocated MB:", torch.cuda.max_memory_allocated() / 1024 / 1024)
