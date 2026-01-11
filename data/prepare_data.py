import numpy as np
import os
from cs336_basics.tokenizer import Tokenizer # 替换为你自己的路径

def tokenize_and_save(input_path, output_path, tokenizer):
    print(f"正在编码 {input_path}...")
    
    # 记录总 token 数
    token_count = 0
    
    # 使用之前实现的迭代编码器，避免内存爆炸
    # 假设你的 Tokenizer 有 encode_iterable 方法
    with open(input_path, 'r', encoding='utf-8') as f:
        # 我们一次读入一大块，或者逐行读入
        ids = []
        for token_id in tokenizer.encode_iterable(f):
            ids.append(token_id)
            token_count += 1
            if token_count % 1000000 == 0:
                print(f"已处理 {token_count // 1000000}M tokens...")
                
    # 转换为 uint16 并保存
    ids_array = np.array(ids, dtype=np.uint16)
    ids_array.tofile(output_path)
    print(f"成功！{output_path} 已保存。总 Token 数: {token_count}")

# 运行示例
if __name__ == "__main__":
    # 加载你训练好的分词器
    tokenizer = Tokenizer.from_files(
        vocab_filepath="models/tokenizer_vocab.json",
        merges_filepath="models/tokenizer_merges.txt",
        special_tokens=["<|endoftext|>"]
    )
    
    os.makedirs("data/processed", exist_ok=True)
    tokenize_and_save("data/TinyStories_train.txt", "data/processed/train.bin", tokenizer)
    tokenize_and_save("data/TinyStories_val.txt", "data/processed/val.bin", tokenizer)