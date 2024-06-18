from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
import math  # 添加math库以使用sqrt函数
from model import GPTConfig
from model import GPT 
from model import DataLoader
import tiktoken
encoder = tiktoken.get_encoding("gpt2")
        

def train(model, dataloader):
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    i = 0
    while not dataloader.no_next():
        i = i + 1
        x, y = dataloader.next_batch()
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits, loss = model(x, y)
        loss.backward()
        optimizer.step()
        print(f"step {i}, loss: {loss.item()}")


if __name__ == "__main__":
    num_return_sequences = 5
    max_length = 30

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = GPT(config=GPTConfig())
    model.to(device)

    trainloader = DataLoader(B = 4, T = 32)
    train(model, trainloader)

    torch.save(model.state_dict(), 'gpt2.pth')
