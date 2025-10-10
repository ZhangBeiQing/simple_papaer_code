import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# 超参数
T = 1000  # 时间步数
beta_start = 0.0001  # 初始beta
beta_end = 0.02  # 结束beta
data_dim = 784  # 数据维度（28x28图像展平后的维度）

# 1. 噪声调度器
betas = torch.linspace(beta_start, beta_end, T)  # 线性调度
alphas = 1.0 - betas
alphas_cumprod = torch.cumprod(alphas, dim=0)  # 累积乘积 alpha_t
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)

# 2. 前向过程：加噪函数
def forward_diffusion(x0, t, noise):
    """
    x0: 初始数据 [batch_size, data_dim]
    t: 时间步 [batch_size]
    noise: 随机噪声 [batch_size, data_dim]
    返回: 加噪后的 xt
    """
    sqrt_alpha_t = sqrt_alphas_cumprod[t].view(-1, 1)  # [batch_size, 1]
    sqrt_one_minus_alpha_t = sqrt_one_minus_alphas_cumprod[t].view(-1, 1)
    xt = sqrt_alpha_t * x0 + sqrt_one_minus_alpha_t * noise
    return xt

# 3. 神经网络：噪声预测器
class NoisePredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim, time_dim):
        super(NoisePredictor, self).__init__()
        self.time_embedding = nn.Embedding(T, time_dim)  # 时间步嵌入
        self.net = nn.Sequential(
            nn.Linear(input_dim + time_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, xt, t):
        t_embed = self.time_embedding(t)  # [batch_size, time_dim]
        input = torch.cat([xt, t_embed], dim=-1)  # 拼接 xt 和时间嵌入
        return self.net(input)

# 4. 训练模型
def train_diffusion(model, data, epochs=1000, batch_size=32):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    for epoch in range(epochs):
        # 随机采样 batch 和时间步
        indices = torch.randint(0, len(data), (batch_size,))
        x0 = data[indices]  # [batch_size, data_dim]
        t = torch.randint(0, T, (batch_size,))  # [batch_size]
        noise = torch.randn_like(x0)  # [batch_size, data_dim]
        
        # 前向加噪
        xt = forward_diffusion(x0, t, noise)
        
        # 预测噪声
        pred_noise = model(xt, t)
        
        # 损失：预测噪声与真实噪声的均方误差
        loss = F.mse_loss(pred_noise, noise)
        
        # 优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

# 5. 生成样本
def sample(model, n_samples):
    xt = torch.randn(n_samples, data_dim)  # 从纯噪声开始
    for t in reversed(range(T)):
        t_tensor = torch.full((n_samples,), t, dtype=torch.long)
        pred_noise = model(xt, t_tensor)
        
        # 逆向一步
        alpha_t = alphas[t]
        beta_t = betas[t]
        noise = torch.randn_like(xt) if t > 0 else 0  # 最后一步不加噪声
        xt = (1 / torch.sqrt(alpha_t)) * (xt - (beta_t / torch.sqrt(1 - alphas_cumprod[t])) * pred_noise) + torch.sqrt(beta_t) * noise
    
    return xt

# 示例运行
if __name__ == "__main__":
    # 生成伪图像数据（模拟28x28的图像）
    # 每个样本是一个784维的向量（28*28=784），代表一张展平的图像
    n_samples = 1000
    
    # 生成一些简单的模式图像数据
    # 模式1：中心亮的图像
    center_pattern = torch.zeros(n_samples // 2, 28, 28)
    center_pattern[:, 10:18, 10:18] = 1.0  # 中心8x8区域为亮
    center_pattern = center_pattern.view(n_samples // 2, -1)  # 展平为784维
    
    # 模式2：边缘亮的图像  
    edge_pattern = torch.zeros(n_samples // 2, 28, 28)
    edge_pattern[:, :3, :] = 1.0   # 上边缘
    edge_pattern[:, -3:, :] = 1.0  # 下边缘
    edge_pattern[:, :, :3] = 1.0   # 左边缘
    edge_pattern[:, :, -3:] = 1.0  # 右边缘
    edge_pattern = edge_pattern.view(n_samples // 2, -1)  # 展平为784维
    
    # 合并两种模式（不添加额外噪声！）
    data = torch.cat([center_pattern, edge_pattern], dim=0)
    # 注意：这里不添加噪声！训练时会通过forward_diffusion动态加噪
    data = torch.clamp(data, 0, 1)  # 限制在[0,1]范围内
    
    # 随机打乱数据
    indices = torch.randperm(n_samples)
    data = data[indices]
    
    print(f"Generated image data shape: {data.shape}")  # 应该是 [1000, 784]
    print(f"Data range: [{data.min():.3f}, {data.max():.3f}]")
    print(f"Data mean: {data.mean():.3f}")
    
    # 初始化模型
    model = NoisePredictor(input_dim=data_dim, hidden_dim=512, time_dim=64)  # 增大网络容量
    
    # 训练
    train_diffusion(model, data, epochs=2000)  # 增加训练轮数
    
    # 生成样本
    samples = sample(model, n_samples=10)
    samples = torch.clamp(samples, 0, 1)  # 确保生成的样本在合理范围内
    
    print("Generated samples shape:", samples.shape)  # 应该是 [10, 784]
    print("Generated samples range:", f"[{samples.min():.3f}, {samples.max():.3f}]")
    
    # 将第一个生成的样本重塑为28x28图像并打印部分内容
    first_sample = samples[0].view(28, 28)
    print("First generated sample (28x28 image, showing top-left 8x8 corner):")
    print(first_sample[:8, :8])