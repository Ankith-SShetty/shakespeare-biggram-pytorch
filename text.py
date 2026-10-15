import torch
import torch.nn as nn
import torch.optim as optim
import requests
import torch.nn.functional as F
from torchgen.api.types import longT

with open("shakespeare.txt", "r", encoding="utf-8") as f:
    text = f.read()

print("Dataset length:", len(text))
print(text[:500])

chars = sorted(list(set(text)))

vocab_size = len(chars)

char_to_int = {ch: i for i, ch in enumerate(chars)}
int_to_char = {i: ch for i, ch in enumerate(chars)}

encoded_text = torch.tensor(
    [char_to_int[ch] for ch in text],
    dtype=torch.long
)

print("Vocabulary size:", vocab_size)

sequence_length = 100

X = []
Y = []

for i in range(len(encoded_text) - sequence_length):
    X.append(encoded_text[i:i + sequence_length])
    Y.append(encoded_text[i + 1:i + sequence_length + 1])

X = torch.stack(X)
Y = torch.stack(Y)

print("X shape:", X.shape)
print("Y shape:", Y.shape)

split_index = int(0.9 * len(encoded_text))

train_data = encoded_text[:split_index]
val_data = encoded_text[split_index:]

print("Training characters:", len(train_data))
print("Validation characters:", len(val_data))

block_size = 100
batch_size = 32

def get_batch(split):
    if split == "train":
        data = train_data
    else:
        data = val_data

    ix = torch.randint(
        0,
        len(data) - block_size,
        (batch_size,)
    )

    x = torch.stack(
        [data[i: i + block_size] for i in ix]
    )

    y = torch.stack(
        [data[i + 1: i +block_size + 1] for i in ix]
    )

    return x, y

xb, yb = get_batch("train")

print(xb, xb.shape)
print(yb, yb.shape)

class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size,vocab_size)

    def forward(self, idx, target = None):
        logits = self.token_embedding_table(idx)
        if target is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            target = target.view(B * T)
            loss = F.cross_entropy(logits, target)

        return logits, loss

    def generate(self, idx, max_new_token):
        for i in range(max_new_token):
            logits, loss = self(idx)
            logits = logits[:, -1, :]
            probs = F.softmax(logits,dim=-1)

            idx_next = torch.multinomial(probs,num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)

        return idx

model = BigramLanguageModel(vocab_size)
optimizer = torch.optim.Adam(model.parameters(),
                                     lr=0.01)

for step in range(1000):
    xb, yb = get_batch("train")
    logits, loss = model(xb, yb)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % 100 == 0:
        print(f"Step = {step} Loss = {loss.item():.2f}")

logits, loss = model(xb, yb)

context = torch.zeros((1,1), dtype= torch.long)

with torch.no_grad():
    generated_ids = model.generate(context, max_new_token=500)

generated_text = ''.join(
    int_to_char [i] for i in generated_ids[0].tolist()
)

print("\nGenerated Shakespeare Text:\n")
print(generated_text)










































