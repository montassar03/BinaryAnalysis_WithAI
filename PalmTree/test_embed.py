import torch

# Path to your trained BERT model (from the log: EP:19 Model Saved on: cdfg_bert_1/transformer.ep19)
MODEL_PATH = "cdfg_bert_1/transformer.ep19"

# From the training log: VOCAB SIZE: 13005
VOCAB_SIZE = 13005

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

print("Loading model from:", MODEL_PATH)
bert = torch.load(MODEL_PATH, map_location=device)
bert.to(device)
bert.eval()

# Build a fake batch: 2 sequences, each length 20, with token IDs in [0, VOCAB_SIZE)
batch_size = 2
seq_len = 20

input_ids = torch.randint(
    low=0,
    high=VOCAB_SIZE,
    size=(batch_size, seq_len),
    dtype=torch.long,
    device=device,
)

# Segment ids: all zeros (only one segment)
segment_ids = torch.zeros(
    (batch_size, seq_len),
    dtype=torch.long,
    device=device,
)

with torch.no_grad():
    # BERT-pytorch BERT returns hidden states: [batch, seq_len, hidden_dim]
    hidden = bert(input_ids, segment_ids)

print("Hidden shape:", hidden.shape)

# Show a small slice of the first embedding vector
first_vec = hidden[0, 0, :10].detach().cpu().numpy()
print("First [CLS-like] embedding (first 10 dims):")
print(first_vec)
