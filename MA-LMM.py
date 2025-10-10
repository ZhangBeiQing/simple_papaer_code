import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional

# --- 1. Q-Former 基础构建块 ---
class QFormerBlock(nn.Module):
    """
    Q-Former的一个基本构建块。
    包含自注意力、交叉注意力和前馈网络。
    """
    def __init__(self, hidden_dim, num_heads, ffn_dim):
        super().__init__()
        # 自注意力层：查询向量内部交互
        self.self_attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(hidden_dim)
        
        # 交叉注意力层：查询向量“审问”视觉特征
        self.cross_attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden_dim)
        
        # 前馈网络
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ffn_dim),
            nn.GELU(),
            nn.Linear(ffn_dim, hidden_dim)
        )
        self.norm3 = nn.LayerNorm(hidden_dim)

    def forward(self, queries: torch.Tensor, 
                  self_attn_kv: torch.Tensor, 
                  cross_attn_kv: torch.Tensor) -> torch.Tensor:
        """
        前向传播。
        Args:
            queries (torch.Tensor): 当前时间步的查询 [B, N_q, C]
            self_attn_kv (torch.Tensor): 自注意力的Key/Value，包含历史查询 [B, L_q, C]
            cross_attn_kv (torch.Tensor): 交叉注意力的Key/Value，包含历史视觉特征 [B, L_v, C]
        Returns:
            torch.Tensor: 更新后的查询 [B, N_q, C]
        """
        # 自注意力 (残差连接)
        attn_output, _ = self.self_attention(query=self.norm1(queries), key=self_attn_kv, value=self_attn_kv)
        queries = queries + attn_output
        
        # 交叉注意力 (残差连接)
        attn_output, _ = self.cross_attention(query=self.norm2(queries), key=cross_attn_kv, value=cross_attn_kv)
        queries = queries + attn_output
        
        # 前馈网络 (残差连接)
        ffn_output = self.ffn(self.norm3(queries))
        queries = queries + ffn_output
        
        return queries


# --- 2. 记忆库压缩器 ---
class MemoryBankCompressor(nn.Module):
    """
    实现记忆库压缩（MBC）逻辑的模块。
    通过融合时间上最相似的相邻帧来维持记忆库大小。
    """
    def __init__(self):
        super().__init__()

    @torch.no_grad() # 压缩过程不参与梯度计算
    def forward(self, memory_bank: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        对给定的记忆库（视觉或查询）执行一次压缩。
        Args:
            memory_bank (List[torch.Tensor]): 包含特征张量的列表，每个张量形状为 [B, N, C]。
        Returns:
            List[torch.Tensor]: 压缩后的记忆库列表。
        """
        if len(memory_bank) <= 1:
            return memory_bank

        # 将列表中的张量堆叠成一个大张量以便高效计算
        # 形状: [B, L, N, C], L是记忆库长度
        bank_tensor = torch.stack(memory_bank, dim=1)
        B, L, N, C = bank_tensor.shape

        # 提取相邻帧对
        adjacent_a = bank_tensor[:, :-1] # [B, L-1, N, C]
        adjacent_b = bank_tensor[:, 1:]  # [B, L-1, N, C]

        # 计算相邻帧特征之间的余弦相似度
        # 我们在每个patch/query位置上独立计算相似度，然后求平均
        similarities = F.cosine_similarity(adjacent_a, adjacent_b, dim=-1) # [B, L-1, N]
        avg_similarities = similarities.mean(dim=-1) # [B, L-1]

        # 找到每个batch中最相似的相邻帧对的索引
        # best_indices 的形状是 [B]
        best_indices = torch.argmax(avg_similarities, dim=1)

        # 创建一个新的、压缩后的记忆库列表
        new_memory_bank = []
        for i in range(B): # 逐个样本处理
            k = best_indices[i].item()
            
            # 融合最相似的帧
            merged_feature = (memory_bank[k][i] + memory_bank[k+1][i]) / 2.0
            
            # 构建新的记忆库
            batch_item_list = []
            for j in range(len(memory_bank)):
                if j < k:
                    batch_item_list.append(memory_bank[j][i])
                elif j == k:
                    batch_item_list.append(merged_feature)
                elif j > k + 1:
                    batch_item_list.append(memory_bank[j][i])
            
            # 将单个样本的压缩后列表堆叠回张量
            new_memory_bank.append(torch.stack(batch_item_list, dim=0))

        # 将batch中所有样本的结果转置并重新整理为列表
        # 结果是一个长度为 L-1 的列表，每个元素是 [B, N, C]
        final_bank = []
        for j in range(len(new_memory_bank[0])):
            final_bank.append(torch.stack([new_memory_bank[i][j] for i in range(B)], dim=0))
            
        return final_bank


# --- 3. 核心：带有记忆库的Q-Former ---
class MA_QFormer(nn.Module):
    """
    Memory-Augmented Q-Former (MA-QFormer)
    模拟在线处理视频帧，并维护视觉和查询记忆库。
    """
    def __init__(self, max_memory_len: int, num_queries=32, hidden_dim=768, num_heads=12, num_layers=6, ffn_dim=3072):
        super().__init__()
        
        self.max_memory_len = max_memory_len
        self.num_queries = num_queries
        self.hidden_dim = hidden_dim
        
        # 可学习的查询向量
        self.learned_queries = nn.Parameter(torch.randn(1, num_queries, hidden_dim))
        
        # 堆叠的Q-Former块
        self.layers = nn.ModuleList([
            QFormerBlock(hidden_dim, num_heads, ffn_dim) for _ in range(num_layers)
        ])
        
        # 记忆库压缩器
        self.compressor = MemoryBankCompressor()
        
        # 运行时状态：记忆库
        self.visual_memory_bank: List[torch.Tensor] = []
        self.query_memory_bank: List[torch.Tensor] = []

    def reset_memory(self):
        """清空记忆库，用于处理新视频。"""
        self.visual_memory_bank = []
        self.query_memory_bank = []

    def _prepare_attention_inputs(self, current_visual_feature: torch.Tensor):
        """根据当前状态和记忆库准备注意力的输入。"""
        B, P, C = current_visual_feature.shape

        # 1. 准备交叉注意力的 Key/Value (视觉历史)
        # 将当前视觉特征与视觉记忆库拼接
        full_visual_history = self.visual_memory_bank + [current_visual_feature]
        # 拼接形状: ([B, P, C], [B, P, C]...) -> [B, L*P, C]
        cross_attn_kv = torch.cat(full_visual_history, dim=1)

        # 2. 准备自注意力的 Key/Value (查询历史)
        queries = self.learned_queries.expand(B, -1, -1)
        if not self.query_memory_bank:
            # 第一帧，没有历史，K/V就是查询自身
            self_attn_kv = queries
        else:
            # 拼接历史查询和当前查询
            history_queries = torch.cat(self.query_memory_bank, dim=1) # [B, (L-1)*N_q, C]
            self_attn_kv = torch.cat([history_queries, queries], dim=1) # [B, L*N_q, C]
            
        return queries, self_attn_kv, cross_attn_kv

    def forward(self, current_visual_feature: torch.Tensor) -> torch.Tensor:
        """
        处理单帧视频特征（在线模式）。
        Args:
            current_visual_feature (torch.Tensor): 当前帧的视觉特征 [B, NumPatches, C]
        Returns:
            torch.Tensor: 当前时间步的Q-Former输出 [B, NumQueries, C]
        """
        B, P, C = current_visual_feature.shape
        
        # 准备注意力输入
        queries, self_attn_kv, cross_attn_kv = self._prepare_attention_inputs(current_visual_feature)
        
        # 依次通过Q-Former层
        for layer in self.layers:
            queries = layer(queries, self_attn_kv, cross_attn_kv)

        # 得到当前时间步的输出
        current_query_output = queries
        
        # --- 更新记忆库 ---
        self.visual_memory_bank.append(current_visual_feature)
        self.query_memory_bank.append(current_query_output.detach()) # 存储时不带梯度

        # --- 检查并执行压缩 ---
        if len(self.visual_memory_bank) > self.max_memory_len:
            self.visual_memory_bank = self.compressor(self.visual_memory_bank)
            self.query_memory_bank = self.compressor(self.query_memory_bank)
            
        return current_query_output

# --- 4. 运行示例 ---
if __name__ == '__main__':
    # --- 模型和数据参数 ---
    BATCH_SIZE = 2
    NUM_PATCHES = 64      # 简化的patch数量
    HIDDEN_DIM = 512      # 简化的特征维度
    MAX_MEMORY = 10       # 记忆库最大长度，超过则压缩
    VIDEO_LENGTH = 20     # 模拟视频的总帧数

    # --- 初始化模型 ---
    ma_qformer = MA_QFormer(
        max_memory_len=MAX_MEMORY,
        num_queries=32,
        hidden_dim=HIDDEN_DIM,
        num_heads=8,
        num_layers=4 # 简化层数
    )
    
    print("--- 模型初始化 ---")
    print(f"最大记忆长度 (M): {ma_qformer.max_memory_len}")
    print(f"视频总帧数: {VIDEO_LENGTH}\n")

    # --- 模拟在线处理视频流 ---
    ma_qformer.reset_memory() # 开始前清空记忆
    
    # 模拟一个视频的特征流
    video_stream = [torch.randn(BATCH_SIZE, NUM_PATCHES, HIDDEN_DIM) for _ in range(VIDEO_LENGTH)]

    for t, frame_feature in enumerate(video_stream):
        print(f"--- 正在处理第 {t+1}/{VIDEO_LENGTH} 帧 ---")
        
        # 获取当前记忆库状态 (处理前)
        vmb_len_before = len(ma_qformer.visual_memory_bank)
        qmb_len_before = len(ma_qformer.query_memory_bank)
        
        # 模型处理当前帧
        output_z_t = ma_qformer(frame_feature)
        
        # 获取当前记忆库状态 (处理后)
        vmb_len_after = len(ma_qformer.visual_memory_bank)
        qmb_len_after = len(ma_qformer.query_memory_bank)
        
        print(f"输入视觉特征形状: {frame_feature.shape}")
        print(f"输出查询形状 (z_{t+1}): {output_z_t.shape}")
        print(f"视觉记忆库长度: {vmb_len_before} -> {vmb_len_after}")
        print(f"查询记忆库长度: {qmb_len_before} -> {qmb_len_after}")
        
        if vmb_len_before == MAX_MEMORY and vmb_len_after == MAX_MEMORY:
            print(">>> 触发记忆压缩! 长度从 {MAX_MEMORY+1} 压缩回 {MAX_MEMORY}。")
        print("-" * 20)

    print("\n--- 视频处理完成 ---")
    print(f"最终视觉记忆库长度: {len(ma_qformer.visual_memory_bank)}")
    print(f"最终查询记忆库长度: {len(ma_qformer.query_memory_bank)}")
    print("模型已准备好使用最后的输出或完整的记忆库进行下游任务（如VQA、Captioning）。")