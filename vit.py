import torch
import torch.nn as nn
from typing import Tuple

# --------------------------------------------------------------------------------
# 1. 底层核心组件 (Core Building Blocks)
# --------------------------------------------------------------------------------

class MultiHeadAttention(nn.Module):
    """
    标准的多头自注意力机制实现。
    完全遵循 "Attention is All You Need" 论文的设计。
    """
    def __init__(self, embed_dim: int, num_heads: int, dropout_rate: float = 0.1):
        """
        初始化多头自注意力层。

        参数:
        - embed_dim (int): 输入和输出的维度 D。
        - num_heads (int): 注意力头的数量。
        - dropout_rate (float): 应用于注意力权重和输出的dropout比率。
        """
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(f"嵌入维度 ({embed_dim}) 必须能被注意力头数量 ({num_heads}) 整除。")

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # 使用一个大的全连接层同时计算 Q, K, V，比用三个独立的层更高效。
        self.qkv_proj = nn.Linear(embed_dim, embed_dim * 3)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        self.attn_dropout = nn.Dropout(dropout_rate)
        self.out_dropout = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播。

        参数:
        - x (torch.Tensor): 输入张量，形状为 (B, N, D)，其中
                           B = 批次大小 (Batch Size)
                           N = 序列长度 (Number of Patches + 1)
                           D = 嵌入维度 (Embedding Dimension)
        返回:
        - torch.Tensor: 输出张量，形状与输入相同 (B, N, D)。
        """
        B, N, D = x.shape

        # 1. 线性投射并分离成 Q, K, V
        # (B, N, D) -> (B, N, 3 * D)
        qkv = self.qkv_proj(x)
        
        # (B, N, 3 * D) -> (B, N, 3, num_heads, head_dim)
        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim)
        
        # (B, N, 3, num_heads, head_dim) -> (3, B, num_heads, N, head_dim)
        # 将 q, k, v 分离
        q, k, v = qkv.permute(2, 0, 3, 1, 4)

        # 2. 计算注意力权重 (Scaled Dot-Product Attention)
        # (B, num_heads, N, head_dim) @ (B, num_heads, head_dim, N) -> (B, num_heads, N, N)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_weights = nn.functional.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        # 3. 加权求和
        # (B, num_heads, N, N) @ (B, num_heads, N, head_dim) -> (B, num_heads, N, head_dim)
        weighted_avg = torch.matmul(attn_weights, v)
        
        # 4. 合并多头并进行最终投射
        # (B, num_heads, N, head_dim) -> (B, N, num_heads * head_dim) -> (B, N, D)
        weighted_avg = weighted_avg.transpose(1, 2).reshape(B, N, self.embed_dim)
        
        # (B, N, D) -> (B, N, D)
        output = self.out_proj(weighted_avg)
        output = self.out_dropout(output)

        return output

class MLP(nn.Module):
    """
    多层感知机（或前馈网络），Transformer Block中的另一个核心组件。
    """
    def __init__(self, embed_dim: int, mlp_dim: int, dropout_rate: float = 0.1):
        """
        初始化MLP。

        参数:
        - embed_dim (int): 输入和输出的维度 D。
        - mlp_dim (int): MLP隐藏层的维度，通常是 embed_dim 的4倍。
        - dropout_rate (float): dropout比率。
        """
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, mlp_dim)
        self.gelu = nn.GELU()  # ViT论文中使用GELU激活函数
        self.fc2 = nn.Linear(mlp_dim, embed_dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x

# --------------------------------------------------------------------------------
# 2. Transformer 编码器层 (Transformer Encoder Block)
# --------------------------------------------------------------------------------

class TransformerEncoderBlock(nn.Module):
    """
    一个标准的Transformer编码器层。
    结构: LayerNorm -> MultiHeadAttention -> Residual -> LayerNorm -> MLP -> Residual
    """
    def __init__(self, embed_dim: int, num_heads: int, mlp_dim: int, dropout_rate: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads, dropout_rate)
        
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim, mlp_dim, dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 自注意力部分
        residual = x
        x = self.norm1(x)
        x = self.attn(x)
        x = x + residual  # 残差连接

        # MLP部分
        residual = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = x + residual  # 残差连接
        
        return x

# --------------------------------------------------------------------------------
# 3. 完整的 Vision Transformer (ViT) 模型
# --------------------------------------------------------------------------------

class VisionTransformer(nn.Module):
    """
    Vision Transformer 模型的完整实现。
    """
    def __init__(
        self,
        img_size: Tuple[int, int] = (224, 224),
        patch_size: Tuple[int, int] = (16, 16),
        in_channels: int = 3,
        num_classes: int = 1000,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_dim: int = 3072,
        dropout_rate: float = 0.1,
    ):
        """
        初始化Vision Transformer模型。

        参数:
        - img_size (Tuple[int, int]): 输入图像尺寸 (H, W)。
        - patch_size (Tuple[int, int]): 每个图像块的尺寸 (PH, PW)。
        - in_channels (int): 输入图像的通道数。
        - num_classes (int): 最终分类任务的类别数。
        - embed_dim (int): 贯穿模型的嵌入维度 D。
        - depth (int): Transformer编码器层的数量 L。
        - num_heads (int): 多头注意力中的头数。
        - mlp_dim (int): MLP隐藏层的维度。
        - dropout_rate (float): dropout比率。
        """
        super().__init__()

        # --- 1. 参数校验和计算 ---
        H, W = img_size
        PH, PW = patch_size
        if H % PH != 0 or W % PW != 0:
            raise ValueError("图像尺寸必须能被patch尺寸整除。")
        
        self.num_patches = (H // PH) * (W // PW)
        patch_dim = in_channels * PH * PW

        # --- 2. 图像块嵌入 (Patch Embedding) ---
        # 这是一个非常优雅的实现技巧：使用一个卷积层来完成图像分块和线性投射。
        # Kernel size 和 stride 都等于 patch size。
        self.patch_embed = nn.Conv2d(
            in_channels, embed_dim, kernel_size=patch_size, stride=patch_size
        )

        # --- 3. [CLS] Token 和位置嵌入 ---
        # [CLS] token，用于最终的分类任务。
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # 位置嵌入，包含[CLS] token的位置。
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))
        self.pos_dropout = nn.Dropout(dropout_rate)

        # --- 4. Transformer 编码器 ---
        # 使用 ModuleList 来存储所有的编码器层。
        self.encoder_blocks = nn.ModuleList(
            [
                TransformerEncoderBlock(embed_dim, num_heads, mlp_dim, dropout_rate)
                for _ in range(depth)
            ]
        )

        # --- 5. 分类头 ---
        self.norm = nn.LayerNorm(embed_dim) # 在送入分类头前进行最后一次归一化
        self.head = nn.Linear(embed_dim, num_classes)
        
        # 初始化权重 (可选但推荐)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # 使用截断正态分布初始化，这在很多Transformer实现中很常见
            nn.init.trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播。

        参数:
        - x (torch.Tensor): 输入图像张量，形状为 (B, C, H, W)。
        
        返回:
        - torch.Tensor: 分类logits，形状为 (B, num_classes)。
        """
        B = x.shape[0]

        # 1. 图像分块与嵌入
        # (B, C, H, W) -> (B, D, H/P, W/P)
        x = self.patch_embed(x)
        # (B, D, H/P, W/P) -> (B, D, N_patches)
        x = x.flatten(2)
        # (B, D, N_patches) -> (B, N_patches, D)
        x = x.transpose(1, 2)

        # 2. 添加 [CLS] Token
        # (1, 1, D) -> (B, 1, D)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        # (B, N_patches, D) -> (B, N_patches + 1, D)
        x = torch.cat((cls_tokens, x), dim=1)

        # 3. 添加位置嵌入
        x = x + self.pos_embed
        x = self.pos_dropout(x)

        # 4. 通过 Transformer 编码器
        for block in self.encoder_blocks:
            x = block(x)

        # 5. 提取 [CLS] Token 并进行分类
        # 取出 [CLS] token 的输出
        cls_output = x[:, 0]
        cls_output = self.norm(cls_output)
        
        # 送入分类头
        logits = self.head(cls_output)

        return logits

# --------------------------------------------------------------------------------
# 4. 示例：如何使用模型
# --------------------------------------------------------------------------------

if __name__ == '__main__':
    # --- 模型配置 (ViT-Base/16) ---
    config = {
        "img_size": (224, 224),
        "patch_size": (16, 16),
        "in_channels": 3,
        "num_classes": 1000,       # ImageNet-1k
        "embed_dim": 768,
        "depth": 12,
        "num_heads": 12,
        "mlp_dim": 3072,
        "dropout_rate": 0.1,
    }

    # --- 实例化模型 ---
    print("正在实例化 ViT 模型...")
    vit_model = VisionTransformer(**config)
    print("模型实例化成功！")

    # --- 创建一个虚拟输入张量 ---
    # (批次大小, 通道数, 高, 宽)
    dummy_input = torch.randn(8, 3, 224, 224)
    print(f"\n创建虚拟输入张量，形状为: {dummy_input.shape}")

    # --- 前向传播 ---
    print("正在进行前向传播...")
    output_logits = vit_model(dummy_input)
    print("前向传播完成！")

    # --- 检查输出 ---
    print(f"输出Logits的形状: {output_logits.shape}") # 应该为 (8, 1000)
    assert output_logits.shape == (8, config["num_classes"])

    # --- 计算模型参数量 ---
    num_params = sum(p.numel() for p in vit_model.parameters() if p.requires_grad)
    print(f"\n模型总参数量: {num_params / 1e6:.2f} M") # ViT-Base 大约 86M