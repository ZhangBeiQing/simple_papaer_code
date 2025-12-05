import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class AttentionConfig:
    """简单的配置类，用于定义模型维度"""
    def __init__(self, n_embd=768, n_head=12, seq_len=1024):
        self.n_embd = n_embd        # 模型隐藏层维度 (d_model)
        self.n_head = n_head        # 注意力头数
        self.seq_len = seq_len      # 序列长度
        assert n_embd % n_head == 0
        self.head_dim = n_embd // n_head

# ==========================================
# 1. 传统经典注意力机制 (Standard Attention)
# ==========================================
class StandardAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_head = config.n_head
        self.head_dim = config.head_dim
        
        # Q, K, V 投影层
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=False)
        # 输出投影层 (W_o)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)

    def forward(self, x):
        B, T, C = x.size() # Batch, Seq_Len, Channel(d_model)

        # 1. QKV 投影并切分多头
        # 输出形状: (B, T, 3 * C) -> (B, T, 3, n_head, head_dim)
        qkv = self.c_attn(x).view(B, T, 3, self.n_head, self.head_dim)
        
        # 调整维度顺序为 (B, n_head, T, head_dim) 以便并行计算
        q, k, v = qkv.permute(2, 0, 3, 1, 4) 

        # 2. 缩放点积注意力 (SDPA)
        # 计算 Attention Scores: (B, H, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        
        # 使用因果掩码 (Causal Mask) - 这里为了简单省略 mask 细节，假设是自回归
        att = F.softmax(att, dim=-1)
        
        # 聚合 Values
        # y 形状: (B, H, T, head_dim)
        y = att @ v 

        # 3. 拼接多头结果
        # 形状变为 (B, T, C)
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # 4. 最终输出投影 W_o
        y = self.c_proj(y)
        
        return y

# ==========================================
# 2. 论文提出的门控注意力 (Gated Attention)
# 论文位置: Section 2.2, 公式 (5)
# 最佳变体: SDPA Output Gating (G1)
# ==========================================
class GatedAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_head = config.n_head
        self.head_dim = config.head_dim
        
        # Q, K, V 投影层
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=False)
        
        # --- 论文核心创新点 ---
        # 门控投影层 (W_theta)
        # 输入是 X (d_model)，输出也是 (d_model) 用于逐元素相乘
        # 这对应论文中的 "Head-Specific" (因为每个维度独立学习) 和 "Element-wise"
        self.gate_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        # --------------------

        # 输出投影层 (W_o)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)

    def forward(self, x):
        B, T, C = x.size()

        # 1. QKV 投影 (同标准版)
        qkv = self.c_attn(x).view(B, T, 3, self.n_head, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)

        # 2. SDPA (同标准版)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = F.softmax(att, dim=-1)
        y = att @ v # 此时 y 是 SDPA 的输出

        # 3. 拼接多头结果
        # y 形状: (B, T, C)
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # --- 论文核心操作: Gating ---
        # 公式: Y' = Y ⊙ σ(X W_theta)
        # 计算门控分数
        gate_score = self.gate_proj(x)  # Linear 投影
        gate_score = torch.sigmoid(gate_score) # Sigmoid 激活 (论文强调 Sigmoid 优于 SiLU)
        
        # 应用门控 (逐元素乘法)
        y = y * gate_score
        # ---------------------------

        # 4. 最终输出投影 W_o
        # 注意：门控是在 SDPA 之后，W_o 之前应用的
        y = self.c_proj(y)
        
        return y

# ==========================================
# 测试代码
# ==========================================
if __name__ == "__main__":
    # 配置
    config = AttentionConfig(n_embd=128, n_head=4, seq_len=10)
    
    # 创建假输入 (Batch=2, Seq=10, Dim=128)
    x = torch.randn(2, 10, 128)
    
    # 1. 运行标准 Attention
    std_attn = StandardAttention(config)
    out_std = std_attn(x)
    print(f"Standard Attention Output Shape: {out_std.shape}")
    
    # 2. 运行 Gated Attention
    gated_attn = GatedAttention(config)
    out_gated = gated_attn(x)
    print(f"Gated Attention Output Shape:    {out_gated.shape}")
    
    # 打印参数量对比
    std_params = sum(p.numel() for p in std_attn.parameters())
    gated_params = sum(p.numel() for p in gated_attn.parameters())
    
    print(f"\nStandard Params: {std_params}")
    print(f"Gated Params:    {gated_params}")
    print(f"Difference:      +{gated_params - std_params} (来自 Gate Linear 层)")
    print(f"Percentage Incr: +{(gated_params - std_params)/std_params * 100:.2f}%")