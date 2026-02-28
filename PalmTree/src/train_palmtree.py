import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.autograd import Variable
from config import *
import numpy as np
import palmtree
from palmtree import dataset
from palmtree import trainer
import pickle as pkl
from palmtree.model.bert import BERT


assert torch.cuda.is_available(), "CUDA is not available."
print("CUDA_VISIBLE_DEVICES =", __import__("os").environ.get("CUDA_VISIBLE_DEVICES"))
print("CUDA device count (visible) =", torch.cuda.device_count())
print("Using device 0 name =", torch.cuda.get_device_name(0))
print("Training on GPU:", torch.cuda.get_device_name(0))

print(palmtree.__file__)
vocab_path = "FullTraining/vocab"
train_cfg_dataset = "data/training/FullTraining/cfg_train.txt"
train_dfg_dataset = "data/training/FullTraining/dfg_train.txt"
test_dataset = None
sent_dataset = "data/sentence.pkl"
output_path = "FullTraining/transformer"

with open(train_cfg_dataset, "r", encoding="utf-8") as f1:
    with open(train_dfg_dataset, "r", encoding="utf-8") as f2:
        vocab = dataset.WordVocab([f1, f2], max_size=13000, min_freq=1)

print("VOCAB SIZE:", len(vocab))
vocab.save_vocab(vocab_path)


print("Loading Vocab", vocab_path)
vocab = dataset.WordVocab.load_vocab(vocab_path)
print("Vocab Size: ", len(vocab))
# print(vocab.itos)


print("Loading Train Dataset")
train_dataset = dataset.BERTDataset(train_cfg_dataset, train_dfg_dataset, vocab, seq_len=20,
                            corpus_lines=None, on_memory=True)

print("Loading Test Dataset", test_dataset)
test_dataset = bert_pytorch.dataset.BERTDataset(test_dataset, test_dataset, vocab, seq_len=20, on_memory=True) \
    if test_dataset is not None else None

print("Creating Dataloader")
train_data_loader = DataLoader(train_dataset, batch_size=64, num_workers=2)



test_data_loader = DataLoader(test_dataset, batch_size=64, num_workers=2) \
    if test_dataset is not None else None

print("Building BERT model")
bert = bert_pytorch.BERT(len(vocab), hidden=128, n_layers=12, attn_heads=8, dropout=0.0)
print("Creating BERT Trainer")
trainer = trainer.BERTTrainer(bert, len(vocab), train_dataloader=train_data_loader, test_dataloader=test_data_loader,
                        lr=1e-5, betas=(0.9, 0.999), weight_decay=0.0,
                        with_cuda=True, cuda_devices=[0], log_freq=100)


print("Training Start")
for epoch in range(10):
    trainer.train(epoch)
    trainer.save(epoch, output_path)
#    if test_data_loader is not None:
#        trainer.test(epoch)
