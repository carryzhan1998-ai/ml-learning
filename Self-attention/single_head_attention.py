import torch
import torch.nn as nn

class SelfAttention(nn.Module):
    def __init__(self, c, d_k):
        super().__init__()
        self.d_k = d_k
        self.Wq = nn.Linear(c, d_k, bias=False)   # C → d_k
        self.Wk = nn.Linear(c, d_k, bias=False)
        self.Wv = nn.Linear(c, d_k, bias=False)

    def forward(self, x):
        """x: (B, T, C)  → 返回 out (B, T, d_k) 和 attn (B, T, T)"""
        B, T, C = x.shape

        Q = self.Wq(x)      # (B, T, d_k)
        K = self.Wk(x)      # (B, T, d_k)
        V = self.Wv(x)      # (B, T, d_k)

        scores = Q @ K.transpose(-2,-1)  # TODO 1：Q 和 K 的"相似度"，形状要凑成 (B, T, T)
        scores = scores / (self.d_k ** 0.5)    # TODO 2：为什么要除以 sqrt(d_k)？
        attn   = torch.softmax(scores, dim=-1)    # TODO 3：把每行的分数变成概率
        out    = attn @ V    # TODO 4：用权重对 V 做加权求和

        return out, attn


# ===== 验证 =====
x = torch.randn(2, 5, 8)          # 2 个句子、每句 5 个词、每个词 8 维
attn_layer = SelfAttention(c=8, d_k=16)
out, attn = attn_layer(x)

print('out :', out.shape)          # 应该是 (2, 5, 16)
print('attn:', attn.shape)         # 应该是 (2, 5, 5)
print('每行和:', attn.sum(-1))     # 应该全是 1.0