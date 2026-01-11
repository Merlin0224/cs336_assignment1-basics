import os
import json
from cs336_basics.tokenizer import train_bpe, Tokenizer

def main():
    # 配置路径
    input_path = "data/TinyStories_train.txt"
    vocab_output = "models/tokenizer_vocab.json"
    merges_output = "models/tokenizer_merges.txt"
    os.makedirs("models", exist_ok=True)

    print("开始在 TinyStories 上训练 BPE 分词器...")
    print("目标词表大小: 10,000")
    
    # 按照文档要求，包含特殊 Token <|endoftext|>
    special_tokens = ["<|endoftext|>"]
    
    # 调用你之前实现的训练函数
    # 注意：如果训练太慢，请确保你的 train_bpe 使用了字典计数优化
    vocab, merges = train_bpe(
        input_path=input_path,
        vocab_size=10000,
        special_tokens=special_tokens
    )

    # 保存结果（使用之前讨论过的保存逻辑）
    # 保存词表
    serializable_vocab = {k: v.hex() for k, v in vocab.items()}
    with open(vocab_output, "w", encoding="utf-8") as f:
        json.dump(serializable_vocab, f, indent=4)
    
    # 保存合并规则
    with open(merges_output, "w", encoding="utf-8") as f:
        for p1, p2 in merges:
            f.write(f"{p1.hex()} {p2.hex()}\n")

    print(f"训练完成！分词器模型已保存至 models/ 文件夹。")

if __name__ == "__main__":
    main()