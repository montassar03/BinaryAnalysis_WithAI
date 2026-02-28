import json
import numpy as np
import eval_utils as utils

TOKENS_JSON = "/home/ru94wid/projects/angr-palmtree/evaluation/tokens/z3_x64-gcc-7-O2_z3.palmtree.json"

# Your trained model + vocab
MODEL_PATH = "/home/ru94wid/projects/PalmTree/cdfg_bert_1/transformerCfgdup.ep19"
VOCAB_PATH = "/home/ru94wid/projects/PalmTree/cdfg_bert_1/vocab_cfgdup"

def main():
    palmtree = utils.UsableTransformer(model_path=MODEL_PATH, vocab_path=VOCAB_PATH)

    d = json.load(open(TOKENS_JSON))
    fn = next(iter(d["functions"].values()))
    bb_addr, insn_tokens = next(iter(fn["blocks"].items()))

    print("Using BB:", bb_addr, "num_insns:", len(insn_tokens))
    emb = palmtree.encode(insn_tokens)  # shape: (N, D)
    print("Instruction-embedding tensor shape:", emb.shape)

    bb_emb = np.mean(emb, axis=0)       # shape: (D,)
    print("Basic-block embedding shape:", bb_emb.shape)
    print("First 8 dims:", bb_emb[:8])

if __name__ == "__main__":
    main()
