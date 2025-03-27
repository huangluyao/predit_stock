import math
import torch
import torch.nn as nn
from dataclasses import dataclass
from torch.nn import functional as F


class LayerNorm(nn.Module):

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                        .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        if self.flash:
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
        else:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        y = self.resid_dropout(self.c_proj(y))
        return y


class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


@dataclass
class GPTConfig:
    block_size: int = 720                   # 从3年前的股价进行推理
    vocab_size: int = 10001
    n_layer: int = 6                        # 6个层
    n_head: int = 6                         # 6个头的注意力机制
    n_embd: int = 384
    dropout: float = 0.2
    bias: bool = True
    min_days: int = 30


class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.vol_e = nn.Linear(1, config.n_embd, bias=False)
        self.proj = nn.Conv2d(config.n_embd, config.n_embd, kernel_size=(1, 5))

        self.wpe = nn.Embedding(config.block_size, config.n_embd)
        self.drop=nn.Dropout(config.dropout)
        self.hide=nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_f=LayerNorm(config.n_embd, bias=config.bias)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size * 4, bias=False)
        self.vol_head = nn.Linear(config.n_embd, 1, bias=False)

        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x_values, x_vol_rates, y_values=None, y_vol_rates=None):
        device = x_values.device
        b, t, c = x_values.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device) # shape (t)

        # 获取对应的价格编码和位置编码
        tok_emb = self.wte(x_values) 
        vol_emb = self.vol_e(x_vol_rates).unsqueeze(2)
        x_data = torch.cat([tok_emb, vol_emb], dim=2)
        batch_emb = []
        for data in x_data:
            data = data.permute([0, 2, 1]).unsqueeze(2)
            emb = self.proj(data)
            batch_emb.append(emb.squeeze())
        x_emb = torch.stack(batch_emb, dim=0)
        pos_emb = self.wpe(pos).unsqueeze(0)

        x = self.drop(x_emb + pos_emb)
        for block in self.hide:
            x = block(x)
        x = self.ln_f(x)

        if y_values is not None:
            logits = self.lm_head(x)
            logits = torch.split(logits, self.config.vocab_size, dim=-1)
            targets = torch.split(y_values,1, dim=-1)
            total_loss = 0
            for res, tag in zip(logits, targets):
                # 至少需要min_days的信息再做预测
                res = res[:, self.config.min_days:].contiguous()
                tag = tag[:, self.config.min_days:].squeeze().contiguous().long()
                loss = F.cross_entropy(res.view(-1, res.size(-1)), tag.view(-1), ignore_index=-1)
                total_loss += loss
            
            vol_prob = self.vol_head(x)
            vol_loss = F.l1_loss(vol_prob, y_vol_rates)

            total_loss += vol_loss
            return total_loss
        else:
            logits = self.lm_head(x[:, [-1], :])
            logits = torch.split(logits, self.config.vocab_size, dim=-1)
            return logits

    @torch.no_grad()
    def generate(self, x_values, x_vol_rates, max_new_tokens=30, temperature=1.0, top_k=None, use_max=True):
        """
        用于预测股价
        """
        for _ in range(max_new_tokens):

            # 控制好最大长度
            block_size = self.config.block_size
            x_values = x_values if x_values.shape[1] <= block_size else x_values[:, -block_size:]
            x_vol_rates = x_vol_rates if x_values.shape[1] <= block_size else x_vol_rates[:, -block_size:]
            # 预测股价
            logits, vol_prob = self(x_values, x_vol_rates)

            next_values = []
            max_prob_list = []
            for res in logits:
                # 从top_k的情况下，随机选择一个结果
                if top_k is not None:
                    v, _ = torch.topk(res, min(top_k, res.size(-1)))
                    logits[logits < v[:, [-1]]] = -float('Inf')
                probs = F.softmax(res, dim=-1)
                if use_max:
                    max_prob, idx_next = torch.max(probs, dim=-1, keepdim=True)
                    max_prob_list.append(max_prob.item())
                else:
                    # 温度参数平滑结果
                    logits = [res[:, -1, :] / temperature for res in logits]
                    # 采样分布
                    idx_next = torch.multinomial(probs, num_samples=1)
                next_values.append(idx_next)
            next_values = torch.stack(next_values)
            next_values = next_values.permute(1, 2, 0)
            x_values = torch.cat((x_values, next_values), dim=1)
            x_vol_rates = torch.cat([x_vol_rates, vol_prob], dim=1)

        return x_values, x_vol_rates, max_prob_list
