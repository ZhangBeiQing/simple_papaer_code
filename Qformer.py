import torch
import torch.nn as nn
import math

class QFormerBlock(nn.Module):
    """
    Q-Former的一个基本构建块。
    每个块包含一个自注意力层、一个交叉注意力层和一个前馈网络。
    """
    def __init__(self, hidden_dim, num_heads, ffn_dim):
        super().__init__()
        
        # 1. 查询向量之间的自注意力层
        #    - 允许可学习的查询向量之间进行信息交互
        #    - 目的是协调它们将要从视觉特征中提取什么样的信息
        self.self_attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(hidden_dim)
        
        # 2. 查询向量与视觉特征之间的交叉注意力层
        #    - 这是Q-Former的核心，是查询向量“审问”视觉特征的地方
        #    - 查询向量是'query'，视觉特征是'key'和'value'
        self.cross_attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden_dim)
        
        # 3. 标准的前馈网络 (Feed-Forward Network)
        #    - 用于对融合了视觉信息的查询向量进行非线性变换和深度处理
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ffn_dim),
            nn.GELU(),
            nn.Linear(ffn_dim, hidden_dim)
        )
        self.norm3 = nn.LayerNorm(hidden_dim)

    def forward(self, queries, visual_features):
        """
        前向传播过程。

        Args:
            queries (torch.Tensor): 查询张量，形状为 [Batch, NumQueries, HiddenDim]
            visual_features (torch.Tensor): 视觉特征张量，形状为 [Batch, NumPatches, HiddenDim]
            
        Returns:
            torch.Tensor: 更新后的查询张量，形状为 [Batch, NumQueries, HiddenDim]
        """
        # --- 第一步: 自注意力 (Self-Attention) ---
        # 查询向量关注自身，进行信息内部整合
        # 残差连接: 将输入与输出相加，防止梯度消失
        attn_output, _ = self.self_attention(query=self.norm1(queries), key=self.norm1(queries), value=self.norm1(queries))
        queries = queries + attn_output
        
        # --- 第二步: 交叉注意力 (Cross-Attention) ---
        # 查询向量作为query，去关注(attend to)视觉特征(key和value)
        # 这是信息从视觉模态流向查询向量的关键步骤
        attn_output, _ = self.cross_attention(query=self.norm2(queries), key=visual_features, value=visual_features)
        queries = queries + attn_output
        
        # --- 第三步: 前馈网络 (FFN) ---
        # 对经过注意力融合后的查询向量进行深度处理
        ffn_output = self.ffn(self.norm3(queries))
        queries = queries + ffn_output
        
        return queries


class QFormer(nn.Module):
    """
    完整的Q-Former模型。
    包含可学习的查询向量和多个堆叠的QFormerBlock。
    """
    def __init__(self, num_queries=32, hidden_dim=768, num_heads=12, num_layers=6, ffn_dim=3072):
        super().__init__()
        
        # 这是Q-Former的灵魂：一组可学习的查询向量。
        # 它们是模型的参数，在训练中被学习，独立于任何输入图像。
        # 它们的形状是 [1, NumQueries, HiddenDim]，'1'是为了方便后续的批处理广播。
        self.learned_queries = nn.Parameter(torch.randn(1, num_queries, hidden_dim))
        
        # 将多个QFormerBlock堆叠起来，形成一个深度网络。
        # 每一层都会对查询向量进行更精细的提炼。
        self.layers = nn.ModuleList([
            QFormerBlock(hidden_dim, num_heads, ffn_dim) for _ in range(num_layers)
        ])
        
        # 最终的层归一化
        self.final_norm = nn.LayerNorm(hidden_dim)

    def forward(self, visual_features):
        """
        Q-Former的前向传播。

        Args:
            visual_features (torch.Tensor): 来自视觉编码器（如ViT）的输出特征。
                                           形状: [Batch, NumPatches, HiddenDim]
        
        Returns:
            torch.Tensor: 经过视觉信息“填充”后的查询向量，作为LLM的输入前缀。
                          形状: [Batch, NumQueries, HiddenDim]
        """
        # 获取输入视觉特征的batch_size
        batch_size = visual_features.shape[0]
        
        # 将可学习的查询向量广播到与输入相同的批次大小
        # 这样，每一批的图像都有自己独立的查询过程
        # .expand() 是一个高效的操作，不会实际复制数据
        queries = self.learned_queries.expand(batch_size, -1, -1)
        
        # 依次通过每一个QFormerBlock
        for layer in self.layers:
            queries = layer(queries, visual_features)
            
        # 返回经过所有层处理和最终归一化后的查询向量
        return self.final_norm(queries)


if __name__ == '__main__':
    # --- 模拟参数 ---
    BATCH_SIZE = 4
    NUM_PATCHES = 256  # 模拟ViT输出的patch数量 (e.g., 16x16 patches for a 224x224 image)
    HIDDEN_DIM = 768   # 模拟ViT-Base的特征维度
    
    # --- 模拟输入 ---
    # 假设这是从一个冻结的ViT模型中得到的视觉特征
    # 形状：[批大小, Patch数量, 特征维度]
    dummy_visual_features = torch.randn(BATCH_SIZE, NUM_PATCHES, HIDDEN_DIM)
    
    print(f"输入视觉特征的形状: {dummy_visual_features.shape}\n")
    
    # --- 初始化并使用Q-Former ---
    # 我们希望将256个视觉特征压缩成32个代表性的特征
    q_former = QFormer(
        num_queries=32,      # 我们希望得到的输出序列长度
        hidden_dim=HIDDEN_DIM, # 必须与视觉特征的维度匹配
        num_heads=12,        # Transformer的头数
        num_layers=6         # Q-Former的层数
    )
    
    # --- 执行前向传播 ---
    output_queries = q_former(dummy_visual_features)
    
    # --- 打印结果 ---
    print("--- Q-Former ---")
    print(f"可学习查询 (Learned Queries) 的基础形状: {q_former.learned_queries.shape}")
    print(f"输出查询特征的形状: {output_queries.shape}\n")
    
    print(">>> 流程总结 <<<")
    print(f"Q-Former成功地将 {NUM_PATCHES} 个高维视觉特征，")
    print(f"通过 {q_former.learned_queries.shape[1]} 个可学习的'问题'（查询向量），")
    print(f"压缩并提炼成了一个固定长度 {output_queries.shape[1]} 的序列。")
    print("这个输出序列随后可以被投影到LLM的词嵌入空间，作为视觉信息的前缀输入给LLM。")