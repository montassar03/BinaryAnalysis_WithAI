# Submission_BinaryAnalysis_WithAI

---

# Binary Preprocessing

## Overview

Before training and evaluation, all raw binaries are converted into PalmTree-compatible instruction pair datasets. The goal of this preprocessing stage is to transform compiled binaries into structured instruction sequences that can be used for contextual representation learning.

The preprocessing pipeline is based on static disassembly using `angr`, followed by sequential instruction pairing.

---

## Dataset Structure

The raw dataset contains multiple binary projects compiled for different architectures and optimization levels. For the evaluation setup:

* 5 projects are used for training
* 2 projects are used for evaluation

The original dataset directory is structured as:

```
Dataset-1/
    clamav/
    curl/
    nmap/
    openssl/
    unrar/
    z3/
    zlib/
```

Each project folder contains multiple compiled binaries (e.g., different compilers and optimization levels).

---

## Preprocessing Pipeline

The preprocessing consists of the following steps:

### 1. Static Disassembly

Each binary is loaded using `angr`:

```python
proj = angr.Project(binary_path, auto_load_libs=False)
cfg = proj.analyses.CFGFast()
```

* `auto_load_libs=False` ensures that only the main binary is analyzed.
* `CFGFast()` constructs a fast static control-flow graph.

This extracts all discovered functions and basic blocks.

---

### 2. Instruction Extraction

For each function:

* Iterate over basic blocks
* Extract assembly instructions in textual form
* Normalize instruction formatting
* Preserve opcode and operand structure

Example raw instructions:

```
push ebx
sub esp, 8
call 0x8049d90
```

---

### 3. Instruction Pair Generation

PalmTree expects instruction pairs instead of single instructions.

Given a sequence:

```
push ebx
sub esp, 8
call 0x8049d90
```

We generate:

```
push ebx    sub esp 8
sub esp 8   call 0x8049d90
```

Each line follows the format:

```
<instruction_1><TAB><instruction_2>
```

This corresponds to a sliding window of size 1 over the instruction stream.

---

### 4. Output Format

For each binary:

* A `.pairs.txt` file is generated
* Stored under:

```
training/<project_name>/<binary_name>.pairs.txt
evaluation/<project_name>/<binary_name>.pairs.txt
```

Example:

```
training/clamav/x86-clang-3.5-O0_clamscan.pairs.txt
```

Each file contains:

```
push ebx    sub esp, 8
sub esp, 8  call 0x8049d90
call 0x8049d90  add ebx, 0x1d9a7
...
```

---

## Script Usage

For each project, preprocessing is executed via a dedicated script:

Example:

```bash
python angr-palmtre/scripts/generate_pairs_x86_clamav.py
python angr-palmtre/scripts/generate_pairs_x86_curl.py
python angr-palmtre/scripts/generate_pairs_x86_nmap.py
```

Each script:

* Iterates over binaries in the corresponding project folder
* Filters architecture (e.g., x86 only)
* Disassembles
* Generates instruction pairs
* Writes `.pairs.txt` output files

---

## Handling of Warnings

During disassembly, angr may emit warnings such as:

```
Symbol imported without a known size
Unsupported Dirty x86g_dirtyhelper_...
```

These do not prevent CFG construction and are expected for complex binaries (e.g., floating-point helpers, unresolved indirect jumps). The pipeline continues processing unaffected.

---

## Computational Characteristics

* Large binaries (e.g., ClamAV ~1.9GB) require significant processing time.
* CFG construction dominates preprocessing cost.
* The preprocessing phase is CPU-bound (no GPU required).
* Memory usage depends on binary size and CFG complexity.

---

## Resulting Training Corpus

The final output of preprocessing is a large corpus of instruction pairs used for:

* Vocabulary construction
* Pretraining (MLM, DUP, CWP objectives)
* Downstream retrieval evaluation

The resulting corpus contains tens of millions of instruction pairs.

---



## Shard Generation and Tensor Preprocessing

### Goal

Before training, I preprocess the raw CFG/DFG training corpora into a sharded tensor format stored in:

```
preprocessed_shards/
```

This step makes training scalable and avoids repeatedly parsing huge text files during training.

### Script

Preprocessing is done with:

```
src/preprocess_dataset.py
```

### Input

My raw training files live under:

```
data/training/FullTraining/
  cfg_train.txt
  dfg_train.txt
```

### Output

Preprocessing writes shards/tensors into:

```
preprocessed_shards/
```

### Command

From the project root:

```bash
cd ~/projects/PalmTree
conda activate palmtree-train
export PYTHONPATH=$PWD/src
```

Run preprocessing:

```bash
python src/preprocess_dataset.py \
  --cfg data/training/FullTraining/cfg_train.txt \
  --dfg data/training/FullTraining/dfg_train.txt \
  --out preprocessed_shards \
  2>&1 | tee preprocess_dataset.log
```

After it finishes, I verify that `preprocessed_shards/` is populated:

```bash
ls -lah preprocessed_shards | head
```

> Note: If your `preprocess_dataset.py` uses different argument names, run `python src/preprocess_dataset.py -h` and keep the same structure (cfg, dfg, out). The important part is: **this script produces `preprocessed_shards/`**.

---
## Training Variants

To run an ablation study, I created separate training variants by modifying the **trainer logic** (loss composition + logging) while keeping the rest of the pipeline unchanged (model architecture, data, optimizer, schedule).

### What I changed (high level)

PalmTree’s original trainer optimizes a combined objective:

* **MLM** (masked language modeling)
* **CWP** (CFG next-sentence-style objective)
* **DUP** (DFG next-sentence-style objective)

The original loss in `BERTTrainer.iteration()` is:

* `loss = dfg_next_loss + cfg_next_loss + mask_loss`

To create training variants, I **only changed which terms are included in `loss`**, and I commented out the corresponding logging entries to avoid confusion.

---

## Scripts I modified / created

### 1) `src/palmtree/trainer/pretrain.py` (original baseline)

This file contains the original `BERTTrainer` implementation and serves as the reference baseline (**MLM + CWP + DUP**).
I kept this file unchanged as the “full” model variant.

---

### 2) `src/palmtree/trainer/pretrain_MLM.py` (MLM only)

I created a new trainer script where I disabled both next-sentence objectives by commenting out their loss terms and keeping only the MLM loss.

**Change location:** `BERTTrainer.iteration()`
**Change type:** comment out CWP/DUP loss and keep `mask_loss` only.

* Original:

  * `loss = dfg_next_loss + cfg_next_loss + mask_loss`
* Variant:

  * `loss = mask_loss`

I also commented out the logging entries for CWP and DUP to match the actual training objective.

---

### 3) `src/palmtree/trainer/pretrain_MLM_CWP.py` (MLM + CWP)

I created a variant trainer where I trained with **MLM + CWP** and disabled DUP.

**Change location:** `BERTTrainer.iteration()`
**Change type:** comment out DUP loss term and keep MLM + CWP.

* Variant objective:

  * `loss = cfg_next_loss + mask_loss`

I commented the DUP logging line so that runtime logs reflect the enabled objectives.

---

### 4) `src/palmtree/trainer/pretrain_MLM_DUP.py` (MLM + DUP)

I created a variant trainer where I trained with **MLM + DUP** and disabled CWP.

**Change location:** `BERTTrainer.iteration()`
**Change type:** comment out CWP loss term and keep MLM + DUP.

* Variant objective:

  * `loss = dfg_next_loss + mask_loss`

I commented the CWP logging line so that runtime logs reflect the enabled objectives.

---

## How I selected which trainer is used

To switch between variants without changing imports throughout the code, I updated the trainer package export:

### `src/palmtree/trainer/__init__.py`

This file defines which `BERTTrainer` is imported when I call:

```python
from palmtree import trainer
trainer.BERTTrainer(...)
```

I changed the exported trainer depending on the variant I wanted to run, e.g.:

* Full baseline:

  * `from .pretrain import BERTTrainer`
* MLM-only:

  * `from .pretrain_MLM import BERTTrainer`
* MLM + CWP:

  * `from .pretrain_MLM_CWP import BERTTrainer`
* MLM + DUP:

  * `from .pretrain_MLM_DUP import BERTTrainer`

This allowed me to switch variants by changing a single line in `__init__.py`.

---

## Notes on implementation approach

* I did not remove code; I commented out the disabled loss terms as a reversible change.
* I kept the forward call unchanged (the model still returns all heads), but only the selected losses contribute to optimization.
* Each variant produces logs that match the enabled training tasks (i.e., no printing of disabled loss terms).
---


## Training

### Goal

I train PalmTree using the preprocessed shard/tensor dataset under `preprocessed_shards/`.
Each training objective saves checkpoints (`transformer.ep*`) into its own output folder:

* `FullTraining_All/`
* `FullTraining_MLM/`
* `FullTraining_MLM_CWP/`
* `FullTraining_MLM_DUP/`

### Script

Shard-based training is done with:

```
src/train_palmtree_SHARD.py
```

### Command (GPU)

From the project root:

```bash
cd ~/projects/PalmTree
conda activate palmtree-train
export PYTHONPATH=$PWD/src
export CUDA_VISIBLE_DEVICES=0
```

Run training (example for `FullTraining_All`):

```bash
python src/train_palmtree_SHARD.py \
  --shards preprocessed_shards \
  --output FullTraining_All \
  2>&1 | tee FullTraining_All/train.log
```

During training, I confirm GPU usage:

```bash
nvidia-smi
```

### Output

Training produces:

```
FullTraining_All/
  transformer.ep0
  ...
  transformer.ep9
  vocab
  train.log
```

---

## Training All Variants

To train all 4 objectives sequentially (same shards, different output dirs):

```bash
for MODEL in FullTraining_All FullTraining_MLM FullTraining_MLM_CWP FullTraining_MLM_DUP
do
  export CUDA_VISIBLE_DEVICES=0
  python src/train_palmtree_SHARD.py \
    --shards preprocessed_shards \
    --output ${MODEL} \
    2>&1 | tee ${MODEL}/train.log
done
```

---

## Notes

* I run preprocessing + training in the **`palmtree-train`** environment.
* `preprocessed_shards/` is the **shared preprocessed dataset** used by all training variants.
* `src/train_palmtree.py` exists, but my training pipeline uses `src/train_palmtree_SHARD.py` because it reads from `preprocessed_shards/`.


---

# Evaluation

## Overview

I evaluate PalmTree using a **function-level retrieval task**.
The objective is to measure how well the learned embeddings place semantically equivalent functions close to each other in vector space, while separating unrelated functions.

The evaluation follows an **anchor–candidate ranking protocol**.

---

## Function Extraction

All functions used in the evaluation are stored in:

```
evaluation/functions.jsonl
```

Each entry contains:

* `project_name`
* `binary_name`
* `arch`
* `function_addr`
* `function_name`
* Instruction blocks

Each function is assigned a unique **Function ID (FID)**.

---

## Anchor and Candidate Pool Construction

The retrieval evaluation is based on:

```
evaluation/anchors_pools.jsonl
```

This file is created using:

```
evaluation/build_pools.py
```

### Pool Construction Strategy

For each **anchor function**, I construct a candidate pool consisting of:

### Similar functions (positives)

Functions that:

* Share the same `function_name`
* Belong to the same project
* Originate from the same logical program
* Differ only in compilation settings (e.g., compiler version or optimization level)

### Dissimilar functions (negatives)

All other sampled functions that do not satisfy the above criteria.

Each anchor therefore has:

* One anchor function
* A set of positive functions
* A fixed-size candidate pool

This converts the problem into a **ranking task per anchor**.

---

## How to Build Functions and Pools

### Step 1 – Generate the function file

```
python evaluation/build_functions.py \
  --input ~/projects/PalmTree/how2use_test/zlib_x64_blocks.jsonl \
  --output evaluation/functions.jsonl
```

### Step 2 – Generate anchor pools

```
python evaluation/build_pools.py \
  --functions evaluation/functions.jsonl \
  --output evaluation/anchors_pools.jsonl \
  --pool_size 1000
```

### Notes

* `--pool_size` controls the number of candidates per anchor
* Larger pools make the evaluation more difficult
* The default pool size used in my experiments is 1000

---

## Embedding Extraction

For each trained model variant:

* `FullTraining_All`
* `FullTraining_MLM`
* `FullTraining_MLM_CWP`
* `FullTraining_MLM_DUP`

I extract embeddings using:

```
evaluation/embed_functions.py
```

Example:

```
python evaluation/embed_functions.py \
  --functions evaluation/functions.jsonl \
  --vocab FullTraining_All/vocab \
  --ckpt FullTraining_All/transformer.ep9 \
  --out evaluation/embeddings_FullTraining_All.npy
```

This produces:

* `embeddings_<MODEL>.npy`
* `embeddings_<MODEL>.fids.txt`

All embeddings are L2-normalized before similarity computation.

---

## Retrieval Evaluation

Evaluation is performed using:

```
evaluation/eval_retrieval.py
evaluation/eval_retrieval_NDCG.py
```

For each anchor function, I:

1. Compute cosine similarity between the anchor embedding and all candidate embeddings.
2. Rank candidates in descending similarity order.
3. Compute retrieval metrics based on the ranking.

---

## Evaluation Metrics

### Recall@1

Percentage of anchors for which at least one correct function is ranked first.

### Recall@10

Percentage of anchors for which at least one correct function appears within the top 10 ranked candidates.

This measures how often the model retrieves a correct match among the first 10 results.

### MRR (Mean Reciprocal Rank)

Measures how early the first correct match appears in the ranking.

Higher values indicate that correct matches are ranked closer to the top.

### NDCG

Measures overall ranking quality while considering the positions of all relevant functions in the pool.

Unlike Recall or MRR, NDCG rewards ranking multiple similar functions highly.

I report:

* NDCG (full ranking)
* NDCG@10

---

## Running the Evaluation

### Evaluate a Single Model

```
python evaluation/eval_retrieval_NDCG.py \
  --emb evaluation/embeddings_FullTraining_All.npy \
  --pools evaluation/anchors_pools.jsonl \
  --out evaluation/results_FullTraining_All_ndcg.json
```

### Evaluate All Variants at Once

```
for MODEL in FullTraining_All FullTraining_MLM FullTraining_MLM_CWP FullTraining_MLM_DUP
do
  python evaluation/eval_retrieval_NDCG.py \
    --emb evaluation/embeddings_${MODEL}.npy \
    --pools evaluation/anchors_pools.jsonl \
    --out evaluation/results_${MODEL}_ndcg.json
done
```

---

## Experimental Setting

* Total number of functions: 44,760
* Total evaluated anchors: 27,845
* Fixed-size candidate pools per anchor
* Intrinsic retrieval-based evaluation
* No downstream (extrinsic) task performed

---

## Interpretation

Higher metric values indicate better semantic representations:

* Higher Recall@1 → more exact top-ranked matches
* Higher MRR → correct matches appear earlier
* Higher NDCG → better overall ranking of all similar functions

This evaluation directly measures the quality of the learned embedding space.






------------------------------------------------------------------------------------------------------------------------------------------------------------------
# A clear structure for identifying relevant scripts, logs, checkpoints, and essential project components.

## Binary analysis and input generation.
Inside `angr-palmtree/training`, you can find the scripts used to generate the training pairs for PalmTree.
Since the generated .txt files containing the training pairs (derived from control-flow and data-flow information) are too large to upload to GitHub, I have not included them in this repository. They will be made available through an alternative source.

## Shard Generation and Tensor Preprocessing

- The preprocessing script for generating the tensor datasets is located in `PalmTree/src/` and is named `preprocess_dataset.py`. Since the resulting preprocessed tensors are too large to include in this repository, they will be provided through alternative means.

- The script PalmTree/src/train_palmtree_SHARD.py is the modified training script that uses the preprocessed tensor shards for training, instead of loading the generated .txt pair files directly.

## Training variants
All checkpoints are directly accessible under PalmTree/FullTraining_*, where you will find the corresponding vocabulary files and trained transformer models.


## Evaluation
The PalmTree/evaluation/ directory is organized to clearly separate source code, generated artifacts, logs, and final results to ensure reproducibility and maintainability.
- src/
 Contains all Python scripts used for embedding generation and retrieval evaluation (e.g., embedding extraction, ranking, NDCG computation).
- embeddings/
 Stores generated embedding vectors (.npy) and corresponding function ID mappings (.fids.txt). These files are model outputs and can be regenerated from the source scripts.
- logs/
 Contains execution logs for traceability and debugging.
   - logs/embed/ – Logs from embedding generation runs
   - logs/eval/ – Logs from retrieval and evaluation experiments
- results/
 Stores final evaluation outputs in JSON format (e.g., retrieval metrics, NDCG scores). These files represent the summarized experimental outcomes.
- metadata/
 Includes auxiliary data such as function descriptions and anchor pools used during evaluation.
