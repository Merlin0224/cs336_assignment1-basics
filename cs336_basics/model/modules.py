import torch 
import torch.nn as nn 
import math
import torch.nn.functional as F
from einops import rearrange


class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, device=None, dtype=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # 文档要求存储为 W (形状: out_features x in_features)
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features, device=device, dtype=dtype)
        )
        self._init_weight()
    
    def _init_weight(self):
        # 按照文档 3.4.1 的公式计算标准差
        std = math.sqrt(2.0 / (self.in_features + self.out_features))
        # 截断正态分布：均值0, 标准差std, 截断范围为 [-3sigma, 3sigma]
        nn.init.trunc_normal_(self.weight, mean=0.0, std=std, a=-3.0*std, b=3.0*std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: 形状为 (..., in_features) 的张量
        返回: 形状为 (..., out_features) 的张量
        """
        return torch.einsum("...i, oi -> ...o", x, self.weight)
    

class Embedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, device=None, dtype=None):
        """
        num_embeddings: 词表大小 (vocab_size)
        embedding_dim: 嵌入向量维度 (d_model)
        """
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        
        # 1. 创建嵌入矩阵参数
        # 形状为 (num_embeddings, embedding_dim)
        self.weight = nn.Parameter(
            torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype)
        )
        
        # 2. 按照文档 3.4.1 的要求初始化
        self._init_weight()

    def _init_weight(self):
        # 均值 0, 标准差 1, 截断在 [-3, 3]
        nn.init.trunc_normal_(self.weight, mean=0.0, std=1.0, a=-3.0, b=3.0)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        token_ids: 形状为 (batch_size, sequence_length) 的 LongTensor
        返回: 形状为 (batch_size, sequence_length, embedding_dim) 的张量
        """
        # 检查输入类型（必须是长整型以便索引）
        if not isinstance(token_ids, torch.LongTensor) and token_ids.dtype != torch.long:
            token_ids = token_ids.long()
            
        # 3. 索引逻辑
        return self.weight[token_ids]


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.eps = eps
        self.d_model = d_model
        # 可学习的参数g，初始化为 1
        self.weight = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))
        self._init_weight()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. 记录原始类型并向上转型为 float32
        orig_dtype = x.dtype
        x = x.to(torch.float32)
        
        # 2. 计算 RMS (均方根)
        # x.pow(2).mean(-1, keepdim=True) 计算沿最后一个维度的平均平方值
        # 然后加上 eps 并开方
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        
        # 3. 归一化
        x_norm = x / rms
        
        # 4. 转回原始类型并乘以增益 w
        out = torch.einsum("...d, d -> ...d", x_norm, self.weight).to(orig_dtype)
        return out

    def _init_weight(self):
        nn.init.trunc_normal_(self.weight, mean=0.0, std=1.0, a=-3.0, b=3.0)


class SwiGLU(nn.Module):
    def __init__(self, d_model: int,d_ff: int, device=None, dtype=None):
        super().__init__()
        # 1. 计算 d_ff
        if d_ff is None: # 如果不传，则按公式计算
            d_ff = int(8 * d_model / 3)
            d_ff = 64 * ((d_ff + 64 - 1) // 64)
        
        # 2. 定义三个线性层（使用你之前的自定义 Linear）
        # W1 和 W3 的输出维度是 d_ff
        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w3 = Linear(d_model, d_ff, device=device, dtype=dtype)
        # W2 的输入维度是 d_ff，输出回 d_model
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 按照公式：W2( SiLU(W1(x)) * W3(x) )
        
        # 第一路：经过 W1 后接 SiLU 激活
        gate = F.silu(self.w1(x))
        
        # 第二路：经过 W3
        value = self.w3(x)
        
        # 元素级乘法并投影回 d_model
        return self.w2(gate * value)
    
class RotaryPositionalEmbedding(nn.Module):
    def __init__(self,theta: float,d_k:int,max_seq_len:int,device=None):
        super().__init__()
        self.d_k = d_k

         # 1. 预计算 inv_freq (逆频率)
        # 根据公式: theta_{i,k} = i / (Theta ** ((2k-2)/d))
        # 这里 k 从 1 到 d/2，所以指数部分是 0, 2, 4, ..., d-2
        # 计算: 1.0 / (Theta ** (torch.arange(0, d_k, 2) / d_k))
        powers = torch.arange(0, d_k, 2, device=device).float() / d_k
        inv_freq = 1.0 / (theta ** powers)
        
        # 2. 预计算所有可能的角度 (i * inv_freq)
        # t 形状: (max_seq_len,)
        t = torch.arange(max_seq_len, device=device).float()
        
        # freqs 形状: (max_seq_len, d_k/2)
        # 外积运算计算所有位置的 theta
        freqs = torch.einsum("i, j -> ij", t, inv_freq)
        
        # 3. 预计算并缓存 cos 和 sin
        # 使用 register_buffer，这样它们会随模型移动到 GPU，但不会被视为训练参数
        self.register_buffer("cos_cached", freqs.cos(), persistent=False) # (max_seq_len, d_k/2)
        self.register_buffer("sin_cached", freqs.sin(), persistent=False) # (max_seq_len, d_k/2)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        """
        x: (..., seq_len, d_k) - 任意数量的 batch 维度
        token_positions: (..., seq_len) - 对应 x 中每个 token 的绝对位置
        """
        # 1. 获取对应的 cos 和 sin 值
        # 使用 token_positions 进行索引，形状变为 (..., seq_len, d_k/2)
        cos = self.cos_cached[token_positions]
        sin = self.sin_cached[token_positions]
        
        # 2. 将 x 拆分为两部分进行旋转
        # 按照文档公式，两两配对进行旋转：(x0, x1), (x2, x3) ...
        # 通过切片提取偶数位和奇数位
        x_even = x[..., 0::2]  # (..., seq_len, d_k/2)
        x_odd = x[..., 1::2]   # (..., seq_len, d_k/2)
        
        # 3. 应用旋转变换 (见复数旋转公式或 2D 旋转矩阵)
        # x_even_new = x_even * cos - x_odd * sin
        # x_odd_new = x_even * sin + x_odd * cos
        x_rotated_even = x_even * cos - x_odd * sin
        x_rotated_odd = x_even * sin + x_odd * cos
        
        # 4. 合并回去
        # 这种方式可以将两个 (..., d_k/2) 的张量交替合并回 (..., d_k)
        # 结果应与 x 形状完全一致
        out = torch.empty_like(x)
        out[..., 0::2] = x_rotated_even
        out[..., 1::2] = x_rotated_odd
        
        return out


def softmax(x : torch.Tensor, dim : int) -> torch.Tensor:
    """
    softmax 的 Docstring
    
    :param x: 输入张量
    :param dim: 需要进行 Softmax 的维度
    :return: 归一化后的概率分布
    """
    x = x - torch.max(x, dim=dim, keepdim=True).values
    x = torch.exp(x)
    return x / torch.sum(x, dim=dim, keepdim=True)



def scaled_dot_product_attention(
    Q: torch.Tensor, 
    K: torch.Tensor, 
    V: torch.Tensor, 
    mask: torch.Tensor = None
) -> torch.Tensor:
    """
    实现缩放点积注意力。
    
    Q: (batch_size, ..., seq_len_q, d_k)
    K: (batch_size, ..., seq_len_k, d_k)
    V: (batch_size, ..., seq_len_k, d_v)
    mask: (seq_len_q, seq_len_k) 或者是带 batch 维度的布尔张量
    """
    # 1. 获取 d_k (Query/Key 的维度)
    d_k = Q.size(-1)
    
    # 2. 计算点积分数并缩放: QK^T / sqrt(d_k)
    # 使用 transpose(-2, -1) 将 K 的最后两个维度转置
    # matmul 会自动处理前面的所有 batch 维度
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
    
    # 3. 应用掩码 (Masking)
    if mask is not None:
        # 文档规定: False 表示不参加注意力计算
        # 使用 masked_fill 将 False 的位置替换为极小值（负无穷）
        # 注意: 如果 mask 是 (seq_len, seq_len)，它会自动广播到 batch 维度
        scores = scores.masked_fill(mask == False, float("-inf"))
    
    # 4. 在最后一个维度（Key 的维度）应用 Softmax
    # 使用之前实现的自定义 softmax 函数
    attn_weights = softmax(scores, dim=-1)
    
    # 5. 加权求和: AttentionWeights * V
    # 结果形状: (batch_size, ..., seq_len_q, d_v)
    output = torch.matmul(attn_weights, V)
    
    return output


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, 
                 d_k: int = None, d_v: int = None, 
                 rope: nn.Module = None, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads

        self.d_k = d_k if d_k is not None else d_model // num_heads
        self.d_v = d_v if d_v is not None else d_model // num_heads
        self.rope = rope 

        self.q_proj = Linear(d_model, self.num_heads * self.d_k, device=device, dtype=dtype)
        self.k_proj = Linear(d_model, self.num_heads * self.d_k, device=device, dtype=dtype)
        self.v_proj = Linear(d_model, self.num_heads * self.d_v, device=device, dtype=dtype)
        self.o_proj = Linear(self.num_heads * self.d_v, d_model, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor = None) -> torch.Tensor:
        b, s, _ = x.shape
        h = self.num_heads

        q = rearrange(self.q_proj(x), "b s (h dk) -> b h s dk", h=h)
        k = rearrange(self.k_proj(x), "b s (h dk) -> b h s dk", h=h)
        v = rearrange(self.v_proj(x), "b s (h dv) -> b h s dv", h=h)

        # 只有在传入了 rope 且有位置信息时才应用
        if self.rope is not None and token_positions is not None:
            q = self.rope(q, token_positions)
            k = self.rope(k, token_positions)

        # 因果掩码
        causal_mask = torch.tril(torch.ones((s, s), device=x.device, dtype=torch.bool))
        
        # 计算注意力
        out = scaled_dot_product_attention(q, k, v, mask=causal_mask)
        out = rearrange(out, "b h s dv -> b s (h dv)")
        return self.o_proj(out)
    

