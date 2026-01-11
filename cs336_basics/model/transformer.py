import torch 
import torch.nn as nn 
from .modules import RMSNorm, MultiHeadSelfAttention, SwiGLU, Embedding, Linear, RotaryPositionalEmbedding


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, 
                 rope: nn.Module, device=None, dtype=None):
        super().__init__()
        
        # 1. 第一子层相关组件
        self.attn_norm = RMSNorm(d_model=d_model)
        self.attn = MultiHeadSelfAttention(
            d_model=d_model, 
            num_heads=num_heads, 
            rope=rope, 
            device=device, 
            dtype=dtype
        )
        
        # 2. 第二子层相关组件
        self.ffn_norm = RMSNorm(d_model=d_model)
        self.ffn = SwiGLU(
            d_model=d_model, 
            d_ff=d_ff,
            device=device, 
            dtype=dtype
        )

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, seq_len, d_model)
        token_positions: (batch, seq_len) 用于 RoPE
        """
        # --- 第一子层：Attention ---
        # Pre-norm 路径
        h = self.attn_norm(x)
        h = self.attn(h, token_positions)
        # 残差连接
        x = x + h
        
        # --- 第二子层：FFN ---
        # Pre-norm 路径
        h = self.ffn_norm(x)
        h = self.ffn(h)
        # 残差连接
        x = x + h
        
        return x
    


class TransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        num_layers: int,
        d_model: int,
        num_heads: int,
        d_ff: int,
        theta: float = 10000.0,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.context_length = context_length
        
        # 1. Token 嵌入层
        self.token_embeddings = Embedding(
            vocab_size, d_model, device=device, dtype=dtype
        )
        
        # 2. 初始化唯一的 RoPE 模块（所有层共享同一个缓存）
        # 每个 head 的维度是 d_model // num_heads
        d_k = d_model // num_heads
        self.rope = RotaryPositionalEmbedding(
            theta=theta, d_k=d_k, max_seq_len=context_length, device=device
        )
        
        # 3. 堆叠 num_layers 个 Transformer 块
        # 使用 nn.ModuleList 确保 PyTorch 能正确追踪这些层
        self.layers = nn.ModuleList([
            TransformerBlock(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                rope=self.rope,
                device=device,
                dtype=dtype,
            )
            for _ in range(num_layers)
        ])
        
        # 4. 最终归一化层 (Final Norm)
        self.ln_f = RMSNorm(d_model, device=device, dtype=dtype)
        
        # 5. 输出投影层 (LM Head)
        # 将维度从 d_model 映射到 vocab_size
        self.output_layer = Linear(d_model, vocab_size, device=device, dtype=dtype)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        token_ids: (batch_size, seq_len)
        返回: (batch_size, seq_len, vocab_size) 的 Logits
        """
        b, s = token_ids.shape
        # 确保序列长度不超过 context_length
        assert s <= self.context_length, f"Cannot forward sequence of length {s}, max is {self.context_length}"

        # 1. 准备位置信息 (用于 RoPE)
        # 生成 [0, 1, 2, ..., s-1] 并扩展到 batch 维度
        token_positions = torch.arange(s, device=token_ids.device).expand(b, s)
        
        # 2. Embedding
        x = self.token_embeddings(token_ids)
        
        # 3. 逐层通过 Transformer Blocks
        for layer in self.layers:
            x = layer(x, token_positions)
            
        # 4. 最终归一化
        x = self.ln_f(x)
        
        # 5. 生成 Logits
        logits = self.output_layer(x)
        
        return logits

    @torch.no_grad()
    def generate(self, prompt_ids, max_new_tokens, temperature=1.0, top_p=0.9, eos_token_id=None):
        """
        self: 类的实例引用
        prompt_ids: 输入的起始 token IDs (batch_size, seq_len)
        max_new_tokens: 最大生成长度
        eos_token_id: 终止符的 ID (需从外部传入，如 256)
        """
        self.eval()
        curr_ids = prompt_ids
    
        for _ in range(max_new_tokens):
            # 1. 裁剪输入长度，不能超过模型的最大上下文长度
            # 使用 self.context_length 访问类属性
            input_ids = curr_ids[:, -self.context_length:]
        
            # 2. 前向传播拿到最后一个位置的 logits
            # 调用 self(input_ids) 相当于调用 self.forward
            logits = self(input_ids)[:, -1, :] # 形状 (batch_size, vocab_size)
        
            # 3. 应用温度缩放
            logits = logits / max(temperature, 1e-5)
        
            # 4. 应用 Top-p 过滤
            if top_p < 1.0:
                # 这里使用 torch.softmax 也可以，或者使用你自定义的 softmax 函数
                probs = torch.softmax(logits, dim=-1)
                sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
            
                # 逻辑修正：计算掩码
                sorted_indices_to_remove = cumulative_probs > top_p
                # 保证至少保留一个最可能的 token (防止 top_p 太小导致全被删掉)
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
            
                # 将掩码映射回原始 indices 顺序
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = float('-inf')

            # --- 注意：以下逻辑必须缩进在 if top_p 之外，但在 for 循环之内 ---
            
            # 5. 采样
            probs = torch.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
        
            # 6. 拼接新生成的 token
            curr_ids = torch.cat((curr_ids, next_id), dim=1)
        
            # 7. 停止条件检查：如果生成了终止符则退出循环
            if eos_token_id is not None and next_id.item() == eos_token_id:
                break
            
        return curr_ids