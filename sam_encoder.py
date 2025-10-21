import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Type

# --------------------------------------------------------------------------
# 1. 基础模块 (Patch Embedding, MLP)
# --------------------------------------------------------------------------

class PatchEmbed(nn.Module):
    """
    将图像转换为补丁嵌入 (Patch Embeddings)。
    通过一个卷积层高效实现。
    """
    def __init__(self, kernel_size=(16, 16), stride=(16, 16), padding=(0, 0), in_chans=3, embed_dim=768):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=kernel_size, stride=stride, padding=padding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 输入 x: (B, C, H, W) -> (B, 3, 1024, 1024)
        x = self.proj(x)
        # 卷积后 x: (B, embed_dim, H', W') -> (B, 768, 64, 64)
        return x

class MLPBlock(nn.Module):
    """一个标准的多层感知机模块"""
    def __init__(self, embedding_dim: int, mlp_dim: int, act: Type[nn.Module] = nn.GELU):
        super().__init__()
        self.lin1 = nn.Linear(embedding_dim, mlp_dim)
        self.lin2 = nn.Linear(mlp_dim, embedding_dim)
        self.act = act()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin2(self.act(self.lin1(x)))

# --------------------------------------------------------------------------
# 2. 核心模块 (Windowed Multi-head Self-Attention)
# --------------------------------------------------------------------------

class Attention(nn.Module):
    """
    标准的多头自注意力模块。
    在 SAM 中，它既可以用于全局注意力，也可以在窗口内使用。
    """
    def __init__(self, dim, num_heads=12, qkv_bias=True, proj_drop=0.0):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        # qkv(): -> (B, N, 3 * C)
        # reshape: -> (B, N, 3, num_heads, C // num_heads)
        # permute: -> (3, B, num_heads, N, C // num_heads)
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

def window_partition(x: torch.Tensor, window_size: int) -> torch.Tensor:
    """
    将特征图划分为多个窗口。
    Args:
        x (torch.Tensor): 输入张量 (B, H, W, C)
        window_size (int): 窗口大小
    Returns:
        torch.Tensor: 划分后的窗口 (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows

def window_unpartition(windows: torch.Tensor, window_size: int, H: int, W: int) -> torch.Tensor:
    """
    将窗口合并回原始特征图。
    Args:
        windows (torch.Tensor): 划分后的窗口 (num_windows*B, window_size, window_size, C)
        window_size (int): 窗口大小
        H (int): 原始高度
        W (int): 原始宽度
    Returns:
        torch.Tensor: 合并后的张量 (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    """
    窗口化多头自注意力模块。
    """
    def __init__(self, dim, num_heads, window_size, qkv_bias=True, proj_drop=0.0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.attention = Attention(dim, num_heads, qkv_bias, proj_drop)
        
    def forward(self, x):
        B, H, W, C = x.shape

        # 1. 划分窗口
        x_windows = window_partition(x, self.window_size) # (num_windows*B, win_size, win_size, C)
        
        # 2. 将窗口展平以进行注意力计算
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C) # (num_windows*B, win_size*win_size, C)

        # 3. 在窗口内计算注意力
        attn_windows = self.attention(x_windows) # (num_windows*B, win_size*win_size, C)

        # 4. 恢复窗口形状
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        
        # 5. 合并窗口
        x = window_unpartition(attn_windows, self.window_size, H, W) # (B, H, W, C)
        
        return x

# --------------------------------------------------------------------------
# 3. 组装 Transformer Block
# --------------------------------------------------------------------------

class TransformerBlock(nn.Module):
    """
    一个Transformer Block，包含注意力模块和MLP。
    通过 use_global_attn 控制使用全局注意力还是窗口化注意力。
    """
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        act_layer: Type[nn.Module] = nn.GELU,
        norm_layer: Type[nn.Module] = nn.LayerNorm,
        window_size: int = 14,
        use_global_attn: bool = False,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.use_global_attn = use_global_attn

        if use_global_attn:
            self.attn = Attention(dim, num_heads, qkv_bias)
        else:
            self.attn = WindowAttention(dim, num_heads, window_size, qkv_bias)

        self.norm2 = norm_layer(dim)
        self.mlp = MLPBlock(embedding_dim=dim, mlp_dim=int(dim * mlp_ratio), act=act_layer)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 输入 x: (B, H, W, C)
        shortcut = x
        x = self.norm1(x)

        if self.use_global_attn:
            # 全局注意力需要展平的输入
            B, H, W, C = x.shape
            x = x.view(B, H * W, C)
            x = self.attn(x)
            x = x.view(B, H, W, C)
        else:
            # 窗口化注意力直接使用网格输入
            x = self.attn(x)

        x = shortcut + x
        x = x + self.mlp(self.norm2(x))
        return x

# --------------------------------------------------------------------------
# 4. 最终的 SAM 图像编码器
# --------------------------------------------------------------------------

class SamImageEncoderViT(nn.Module):
    def __init__(
        self,
        img_size: int = 1024,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768, # ViT-Base 维度
        depth: int = 12, # ViT-Base 层数
        num_heads: int = 12, # ViT-Base 头数
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        norm_layer: Type[nn.Module] = nn.LayerNorm,
        act_layer: Type[nn.Module] = nn.GELU,
        window_size: int = 14,
        global_attn_indexes: tuple = (2, 5, 8, 11), # 在哪些层使用全局注意力
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim

        # 1. 补丁嵌入层
        self.patch_embed = PatchEmbed(
            kernel_size=(patch_size, patch_size),
            stride=(patch_size, patch_size),
            in_chans=in_chans,
            embed_dim=embed_dim,
        )
        
        # 2. 位置编码 (绝对位置编码)
        self.num_patches = (img_size // patch_size) ** 2
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=.02)

        # 3. Transformer Blocks
        self.blocks = nn.ModuleList()
        for i in range(depth):
            block = TransformerBlock(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                norm_layer=norm_layer,
                act_layer=act_layer,
                window_size=window_size,
                use_global_attn=i in global_attn_indexes,
            )
            self.blocks.append(block)
            
        # 4. Neck (SAM 输出前会经过一个简单的 'neck' 结构)
        # 简化版，实际SAM更复杂
        self.neck = nn.Sequential(
            nn.Conv2d(embed_dim, 256, kernel_size=1, bias=False),
            norm_layer(256),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            norm_layer(256),
        )


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Step 1: Patch Embedding
        x = self.patch_embed(x) # (B, 768, 64, 64)
        x = x.flatten(2).transpose(1, 2) # (B, 4096, 768)

        # Step 2: Add Positional Embedding
        x = x + self.pos_embed

        # Step 3: Go through Transformer Blocks
        # 需要在 grid 和 sequence 格式之间转换
        B, N, C = x.shape
        H = W = int(N ** 0.5)
        x = x.view(B, H, W, C)
        
        for blk in self.blocks:
            x = blk(x)
        
        # (B, H, W, C) -> (B, 64, 64, 768)
        
        # Step 4: Neck
        # Permute for Conv2d: (B, H, W, C) -> (B, C, H, W)
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.neck(x)
        
        # Final Output: (B, 256, 64, 64)
        return x


# --------------------------------------------------------------------------
# 5. 示例用法
# --------------------------------------------------------------------------
if __name__ == '__main__':
    # 模拟一张高分辨率图片
    dummy_image = torch.randn(1, 3, 1024, 1024)

    # 实例化 SAM-base 编码器
    # 使用 ViT-B 的标准配置
    sam_encoder = SamImageEncoderViT(
        embed_dim=768,
        depth=12,
        num_heads=12,
        global_attn_indexes=(2, 5, 8, 11) # 与真实 SAM ViT-B 配置一致
    )

    # 运行模型
    with torch.no_grad():
        image_embedding = sam_encoder(dummy_image)

    print(f"输入图像尺寸: {dummy_image.shape}")
    print(f"输出图像嵌入尺寸: {image_embedding.shape}")
    # 预期输出: torch.Size([1, 256, 64, 64])