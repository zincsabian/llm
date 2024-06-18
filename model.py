from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
import math  # 添加math库以使用sqrt函数
import tiktoken
encoder = tiktoken.get_encoding("gpt2")


# 定义模型配置数据类
@dataclass
class GPTConfig:
    """配置类，用于存储模型的超参数"""

    block_size: int = 1024  # 序列最大长度
    vocab_size: int = 50257  # 词汇表大小
    n_layer: int = 12  # Transformer的层数
    n_head: int = 12  # 注意力头数
    n_embd: int = 768  # 嵌入维度


# 定义带因果遮罩的自注意力层
class CausalSelfAttention(nn.Module):
    """实现带因果遮罩的多头自注意力机制"""

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0  # 确保嵌入维度能被头数整除
        self.n_embd = config.n_embd
        self.n_head = config.n_head
        self.block_size = config.block_size

        self.c_attn = nn.Linear(self.n_embd, 3 * self.n_embd)  # 用于生成Q、K、V的线性层
        self.c_proj = nn.Linear(self.n_embd, self.n_embd)  # 注意力输出到嵌入维度的映射

        self.register_buffer(
            "bias",
            torch.tril(torch.ones(self.block_size, self.block_size)).view(
                1, 1, config.block_size, config.block_size
            ),
        )  # 最大为T*T的mask矩阵

    def forward(self, x):
        (
            B,
            T,
            C,
        ) = x.size()  # batch_size, sequence_length, embedding_dimensionality (n_embd)
        qkv = self.c_attn(x)  # (B, T, 3C)
        q, k, v = qkv.split(self.n_embd, dim=2)  # 3 * (B, T, C)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, T, nh, hs) --> (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, T, nh, hs) --> (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, T, nh, hs) --> (B, nh, T, hs)

        att = (q @ k.transpose(-2, -1)) * (
            1 / math.sqrt(k.size(-1))
        )  # (B, nh, T, hs) @ (B, nh, hs, T)
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)  # (B, nh, T, T)
        y = att @ v  # (B, nh, T, C / nh)

        y = (
            y.transpose(1, 2).contiguous().view(B, T, C)
        )  # (B, T, nh, C / nh) -> (B, T, C)
        y = self.c_proj(y)  # multi-head feature assemble
        return y


# 定义多层感知机(MLP)
class MLP(nn.Module):
    """前馈神经网络，用于Transformer的非线性变换"""

    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)  # 第一层线性变换
        self.gelu = nn.GELU(approximate="tanh")  # GELU激活函数
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)  # 第二层线性变换，回到原始嵌入维度

    def forward(self, x):
        x = self.c_fc(x)  # 第一层变换
        x = self.gelu(x)  # 激活
        x = self.c_proj(x)  # 第二层变换
        return x


# 定义Transformer块
class Block(nn.Module):
    """Transformer的基本构建块，包含自注意力和前馈网络"""

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)  # 层归一化1
        self.attn = CausalSelfAttention(config)  # 自注意力层
        self.ln_2 = nn.LayerNorm(config.n_embd)  # 层归一化2
        self.mlp = MLP(config)  # 前馈网络

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))  # 自注意力 + 残差连接
        x = x + self.mlp(self.ln_2(x))  # 前馈网络 + 残差连接
        return x


# 定义完整的GPT模型
class GPT(nn.Module):
    """GPT模型，包含词嵌入、位置嵌入、Transformer层和预测头"""

    def __init__(self, config=GPTConfig()):
        super().__init__()
        self.config = config

        # 定义模型组件
        self.transformer = nn.ModuleDict(
            {
                "wte": nn.Embedding(config.vocab_size, config.n_embd),  # 词嵌入
                "wpe": nn.Embedding(config.block_size, config.n_embd),  # 位置嵌入
                "h": nn.ModuleList(
                    [Block(config) for _ in range(config.n_layer)]
                ),  # 多个Transformer块
                "ln_f": nn.LayerNorm(config.n_embd),  # 最终的层归一化
            }
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)  # 预测头

    @classmethod
    def from_pretrain(cls, model_type):
        """Loads pretrained GPT-2 model weights from huggingface"""
        assert model_type in {"gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"}
        from transformers import GPT2LMHeadModel

        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            "gpt2": dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            "gpt2-medium": dict(n_layer=24, n_head=16, n_embd=1024),  # 350M params
            "gpt2-large": dict(n_layer=36, n_head=20, n_embd=1280),  # 774M params
            "gpt2-xl": dict(n_layer=48, n_head=25, n_embd=1600),  # 1558M params
        }[model_type]

        config_args["vocab_size"] = 50257  # always 50257 for GPT model checkpoints
        config_args["block_size"] = 1024  # always 1024 for GPT model checkpoints

        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [
            k for k in sd_keys if not k.endswith(".attn.bias")
        ]  # discard this mask / buffer, not a param
        # 实际上应该啥都没有去掉
        # print(sd_keys)

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [
            k for k in sd_keys_hf if not k.endswith(".attn.masked_bias")
        ]  # ignore these, just a buffer
        sd_keys_hf = [
            k for k in sd_keys_hf if not k.endswith(".attn.bias")
        ]  # same, just the mask (buffer)
        # print(sd_keys_hf)
        # 实际上应该啥都没有去掉
        transposed = [
            "attn.c_attn.weight",
            "attn.c_proj.weight",
            "mlp.c_fc.weight",
            "mlp.c_proj.weight",
        ]

        assert len(sd_keys_hf) == len(
            sd_keys
        ), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model


    def forward(self, idx, targets=None):
        device = idx.device
        B, T = idx.size()  # 批次大小，序列长度

        assert (
            T <= self.config.block_size
        ), f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"

        # 词嵌入和位置嵌入
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        tok_emb = self.transformer["wte"](idx)  # 词嵌入
        pos_emb = self.transformer["wpe"](pos)  # 位置嵌入
        x = tok_emb + pos_emb  # 结合词与位置嵌入
        for block in self.transformer["h"]:
            x = block(x)
        x = self.transformer["ln_f"](x)  # layernorm

        # 预测头，计算logits
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss


class DataLoader:
    def __init__(self, B, T):
        self.B = B              # batch_size
        self.T = T              # tokens_size

        with open('input.txt', 'r') as file:
            text = file.read()
        tokens = encoder.encode(text)
        self.tokens = torch.tensor(tokens)
        print(f"loaded {len(self.tokens)} tokens")
        print(f"1 epoch = {len(self.tokens) // (B * T)} batches")

        self.current_posisition = 0

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_posisition : self.current_posisition + B*T + 1]
        x = (buf[:-1]).view(B, T)
        y = (buf[1:]).view(B, T)
        self.current_posisition += B*T
        if self.no_next():
            self.current_posisition = 0
        return x, y
    
    def no_next(self):
        B, T = self.B, self.T
        # print(self.current_posisition, B, T, len(self.tokens))
        return self.current_posisition + (B*T) > len(self.tokens)