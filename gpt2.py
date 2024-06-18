from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
import math  # 添加math库以使用sqrt函数
import tiktoken
encoder = tiktoken.get_encoding("gpt2")

from model import GPT

if __name__ == '__main__':

    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'

    num_return_sequences = 1
    max_length = 30

    model = GPT()
    model.load_state_dict(torch.load('gpt2.pth'))
    model.to(device)

    tokens = encoder.encode("First Citizen: \
Before we proceed any further, hear me speak.")
    tokens = torch.tensor(tokens, dtype=torch.long)
    tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
    x = tokens.to(device)

    torch.manual_seed(42)
    torch.mps.manual_seed(42)
    model.eval()
    logits, loss = model(x)

    while x.size(1) < max_length:
        with torch.no_grad():
            logits, loss = model(x)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
            ix = torch.multinomial(topk_probs, 1)
            xcol = torch.gather(topk_indices, -1, ix)
            x = torch.cat((x, xcol), dim=-1)

    for i in range(num_return_sequences):
        tokens = x[i, :max_length].tolist()
        decoded = encoder.decode(tokens)
        print(">", decoded)