import os
import json
import numpy as np

from config import *
from torch import nn
from scipy.ndimage.filters import gaussian_filter1d
from torch.autograd import Variable
import torch
import eval_utils as utils


palmtree = utils.UsableTransformer(
    model_path="/home/ru94wid/projects/PalmTree/cdfg_bert_1/transformerCfgdup.ep19",
    vocab_path="/home/ru94wid/projects/PalmTree/cdfg_bert_1/vocab_cfgdup"
)

INPUT_JSONL = "/home/ru94wid/projects/PalmTree/how2use/zlib_x64_blocks.jsonl"
OUTPUT_JSONL = "/home/ru94wid/projects/PalmTree/how2use/zlib_x64_insn_embeddings.jsonl"

os.makedirs(os.path.dirname(OUTPUT_JSONL), exist_ok=True)

print("INPUT :", INPUT_JSONL)
print("OUTPUT:", OUTPUT_JSONL)

written = 0
blocks = 0

with open(INPUT_JSONL, "r") as f, open(OUTPUT_JSONL, "w") as out:
    for i, line in enumerate(f):
        rec = json.loads(line)

        # Your working key
        text = rec.get("insns", [])
        if not text:
            continue

        # Encode: returns (N, 128) where N = len(text)
        embeddings = palmtree.encode(text)

        # Write one output record per instruction (instruction-level, as you requested)
        # Keep original metadata fields if present; safe if missing.
        for ins, vec in zip(text, embeddings):
            out_rec = {
                "project_name": rec.get("project_name"),
                "binary_name": rec.get("binary_name"),
                "target_name": rec.get("target_name"),
                "compiler": rec.get("compiler"),
                "compiler_ver": rec.get("compiler_ver"),
                "optimization_lvl": rec.get("optimization_lvl"),
                "function_name": rec.get("function_name"),
                "function_addr": rec.get("function_addr"),

                # keep whatever identifiers you have (safe if not present)
                "block_addr": rec.get("block_addr"),
                "bb_addr": rec.get("bb_addr"),

                "instruction": ins,
                "embedding": vec.tolist()
            }
            out.write(json.dumps(out_rec) + "\n")
            written += 1

        blocks += 1

        # Live progress every 200 blocks
        if blocks % 200 == 0:
            print(f"blocks={blocks} written_insns={written} last_block_insns={len(text)} last_emb_shape={tuple(embeddings.shape)}")

print("DONE")
print("blocks:", blocks)
print("written_insns:", written)
print("output bytes:", os.path.getsize(OUTPUT_JSONL))
