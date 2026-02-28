import os
from config import *
from torch import nn
from scipy.ndimage.filters import gaussian_filter1d
from torch.autograd import Variable
import torch
import numpy as np
import eval_utils as utils
import json

palmtree = utils.UsableTransformer(
    model_path="/home/ru94wid/projects/PalmTree/cdfg_bert_1/transformerCfgdup.ep19",
    vocab_path="/home/ru94wid/projects/PalmTree/cdfg_bert_1/vocab_cfgdup"
)
INPUT_JSONL = "/home/ru94wid/projects/PalmTree/how2use/zlib_x64_blocks.jsonl"
with open(INPUT_JSONL, "r") as f:
    for i, line in enumerate(f):
        rec = json.loads(line)

        # This replaces the hard-coded example list
        text = rec["insns"]   # List[str]

        embeddings = palmtree.encode(text)

        print("usable embedding of this basicblock:", embeddings)
        print("the shape of output tensor: ", embeddings.shape)

        # stop early just to test (remove later)
        if i == 5:
            break
# tokens has to be seperated by spaces.

#text = ["mov rbp rdi", 
 #       "mov ebx 0x1", 
  #      "mov rdx rbx", 
   #     "call memcpy", 
    #    "mov [ rcx + rbx ] 0x0", 
     #   "mov rcx rax", 
      #  "mov [ rax ] 0x2e"]

# it is better to make batches as large as possible.
#embeddings = palmtree.encode(text)
#print("usable embedding of this basicblock:", embeddings)
#print("the shape of output tensor: ", embeddings.shape)
