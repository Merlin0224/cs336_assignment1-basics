import torch
from cs336_basics.model.transformer import TransformerLM
from cs336_basics.tokenizer import Tokenizer

# 1. 配置（确保与训练时一致）
device = 'cuda'
vocab_size = 10000
context_length = 256
num_layers = 4
num_heads = 16
d_model = 512
d_ff = 1344

# 2. 加载模型
model = TransformerLM(
    vocab_size=vocab_size, context_length=context_length, num_layers=num_layers,
    d_model=d_model, num_heads=num_heads, d_ff=d_ff, device=device
).to(device)

# 加载你保存的最佳检查点
checkpoint = torch.load("models/best_model.pt", map_location=device)
model.load_state_dict(checkpoint['model'])
model.eval()

# 3. 加载分词器
tokenizer = Tokenizer.from_files(
    "models/tokenizer_vocab.json", 
    "models/tokenizer_merges.txt", 
    ["<|endoftext|>"]
)
eos_id = 256 # 或者是你分配的 ID

# 4. 开始生成故事
prompt = "Once upon a time, a little girl named Lily found a magic"
input_ids = torch.tensor([tokenizer.encode(prompt)], device=device)

print(f"--- Prompt: {prompt} ---")

# 尝试不同的采样参数
# temperature < 1.0 会更保守，> 1.0 会更具创意
output_ids = model.generate(
    input_ids, 
    max_new_tokens=150, 
    temperature=0.8, 
    top_p=0.95, 
    eos_token_id=eos_id
)

# 解码并打印
generated_text = tokenizer.decode(output_ids[0].tolist())
print("\n--- Generated Story ---\n")
print(generated_text)