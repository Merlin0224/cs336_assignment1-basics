import os
import time
import math
import numpy as np
import torch
from tqdm import tqdm

from cs336_basics.model.transformer import TransformerLM
from cs336_basics.trainer.AdamW import AdamW
from cs336_basics.trainer.utils import get_lr_cosine_schedule
from cs336_basics.trainer.utils import cross_entropy
from cs336_basics.trainer.utils import gradient_clipping
from cs336_basics.trainer.check_point import save_checkpoint, load_checkpoint

# 限制 PyTorch 显存管理的扩张
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
# --- 1. 配置超参数 (参考文档 Page 41) ---
# 模型配置
vocab_size = 10000
context_length = 256
num_layers = 4
num_heads = 16
d_model = 512
d_ff = 1344
theta = 10000.0

# 训练配置
batch_size = 16  
max_iters = 10000 # 步数 = 总Token / (batch_size * context_length)
eval_interval = 500
save_interval = 1000
log_interval = 10

# 优化器配置
learning_rate = 6e-4
warmup_iters = 500
lr_decay_iters = 10000 # 通常等于 max_iters
min_lr = 6e-5 # learning_rate / 10
weight_decay = 0.1
beta1, beta2 = 0.9, 0.95

device = 'cuda' if torch.cuda.is_available() else 'cpu'
if torch.backends.mps.is_available(): device = 'mps'
print(f"Using device: {device}")

# --- 2. 加载数据 (使用 memmap) ---
train_data = np.memmap('data/processed/train.bin', dtype=np.uint16, mode='r')
val_data = np.memmap('data/processed/val.bin', dtype=np.uint16, mode='r')

def get_batch(data):
    ix = torch.randint(0, len(data) - context_length - 1, (batch_size,))
    x = torch.stack([torch.from_numpy(data[i:i+context_length].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i+1:i+1+context_length].astype(np.int64)) for i in ix])
    return x.to(device), y.to(device)

@torch.no_grad()
def estimate_loss(model):
    model.eval()
    losses = []
    # 在验证集上随机采样 20 个 batch 算平均值
    for _ in range(20):
        X, Y = get_batch(val_data)
        logits = model(X)
        loss = cross_entropy(logits, Y)
        losses.append(loss.item())
    model.train()
    torch.cuda.empty_cache() # 强制释放评估时的显存碎片
    return np.mean(losses)

# --- 3. 初始化模型、优化器 ---
model = TransformerLM(
    vocab_size=vocab_size, context_length=context_length, num_layers=num_layers,
    d_model=d_model, num_heads=num_heads, d_ff=d_ff, theta=theta, device=device
).to(device)

optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay, betas=(beta1, beta2))
scaler = torch.amp.GradScaler('cuda') # 初始化缩放器
# --- 4. 训练循环 ---
start_time = time.time()
best_val_loss = float('inf')

accumulation_steps = 8  # 16 * 8 = 128 (模拟文档建议的有效 batch size)

for it in range(max_iters):
    # 更新学习率
    lr = get_lr_cosine_schedule(it, learning_rate, min_lr, warmup_iters, lr_decay_iters)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    
    optimizer.zero_grad()
    # 模拟大的 batch size
    for _ in range(accumulation_steps):
        X, Y = get_batch(train_data)
        
        # 混合精度开启
        with torch.amp.autocast('cuda'):
            logits = model(X)
            # 注意：loss 需要除以累加步数
            loss = cross_entropy(logits, Y) / accumulation_steps
        
        scaler.scale(loss).backward()


    # 累加完成后，先 unscale 梯度，再进行裁剪
    scaler.unscale_(optimizer) 
    gradient_clipping(model.parameters(), max_norm=1.0)
    
    # 使用 scaler 进行 step 和 update
    scaler.step(optimizer)
    scaler.update()

    # 日志记录
    if it % log_interval == 0:
        elapsed = time.time() - start_time
        print(f"Iter {it}: loss {loss.item():.4f}, lr {lr:.2e}, time {elapsed:.2f}s")

    # 评估与保存
    if it % eval_interval == 0:
        val_loss = estimate_loss(model)
        val_perplexity = math.exp(val_loss)
        print(f">>> EVAL: iter {it}, val_loss {val_loss:.4f}, perplexity {val_perplexity:.2f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(model, optimizer, it, "models/best_model_AMP_GA.pt")

print("训练结束！")