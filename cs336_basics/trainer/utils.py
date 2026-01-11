import torch
import math
import numpy as np

def cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    计算交叉熵损失。
    logits: (batch_size, ..., vocab_size)
    targets: (batch_size, ...) 包含类别索引
    """
    # 1. 记录原始形状并将 logits 展平为 (N, vocab_size)，targets 展平为 (N,)
    # 其中 N 是所有 batch 维度的乘积
    vocab_size = logits.size(-1)
    logits_flat = logits.reshape(-1, vocab_size)
    targets_flat = targets.reshape(-1)

    # 2. 数值稳定性：减去每一行的最大值
    # keepdim=True 方便广播
    logits_max, _ = torch.max(logits_flat, dim=-1, keepdim=True)
    logits_stable = logits_flat - logits_max

    # 3. 计算 Log-Sum-Exp
    # log(sum(exp(x)))
    log_sum_exp = torch.log(torch.sum(torch.exp(logits_stable), dim=-1))
    
    # 4. 提取正确类别的 logits (o_yi)
    # 使用 gather 提取对应 target 索引处的 logit 值
    # targets_flat.unsqueeze(1) 变成 (N, 1)
    correct_logits = logits_stable.gather(dim=1, index=targets_flat.unsqueeze(1)).squeeze(1)

    # 5. 计算损失： - (o_yi - log_sum_exp) = log_sum_exp - o_yi
    # 注意：logits_max 已经抵消了，所以最终公式如下：
    loss = log_sum_exp - correct_logits
    
    # 6. 返回全局平均值
    return loss.mean()


def get_lr_cosine_schedule(t, alpha_max, alpha_min, Tw, Tc):
    # 第一阶段：Warm-up (线性增加)
    if t < Tw:
        return (t / Tw) * alpha_max
    
    # 第二阶段：Cosine Annealing (余弦退火)
    if Tw <= t <= Tc:
        # 计算进度比例
        progress = (t - Tw) / (Tc - Tw)
        # 计算余弦系数 (从 1 到 0)
        multiplier = 0.5 * (1.0 + math.cos(progress * math.pi))
        # 在 max 和 min 之间插值
        return alpha_min + multiplier * (alpha_max - alpha_min)
    
    # 第三阶段：Post-annealing (保持最小值)
    return alpha_min



def gradient_clipping(parameters, max_norm: float):
    # 1. 收集所有具有梯度的参数
    grads = [p.grad for p in parameters if p.grad is not None]
    
    # 2. 计算全局 L2 范数
    # sum(g^2) for all elements in all tensors
    total_norm = torch.sqrt(sum(torch.sum(g ** 2) for g in grads))
    
    # 3. 如果超过阈值，进行缩放
    if total_norm > max_norm:
        # 结果应略小于 max_norm (加了 epsilon)
        clip_coef = max_norm / (total_norm + 1e-6)
        for g in grads:
            g.mul_(clip_coef)


def get_batch(data, batch_size, context_length, device):
    # 1. 随机生成起始点索引 (注意范围不能让 Y 越界)
    # data 的长度为 n，我们需要取到 i + context_length 的位置
    ix = torch.randint(0, len(data) - context_length, (batch_size,))
    
    # 2. 提取 X 和 Y
    x_list = [torch.from_numpy((data[i : i + context_length]).astype(np.int64)) for i in ix]
    y_list = [torch.from_numpy((data[i + 1 : i + 1 + context_length]).astype(np.int64)) for i in ix]
    
    # 3. 堆叠并移动到设备
    x = torch.stack(x_list).to(device)
    y = torch.stack(y_list).to(device)
    return x, y