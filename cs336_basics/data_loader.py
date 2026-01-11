import numpy as np
from cs336_basics.tokenizer import Tokenizer 

def prepare_data(input_file, vocab_file, merges_file, output_file, special_tokens=["<|endoftext|>"]):
    # 1. 加载分词器
    tokenizer = Tokenizer.from_files(
        vocab_filepath=vocab_file,
        merges_filepath=merges_file,
        special_tokens=special_tokens
    )
    
    print(f"正在处理 {input_file}...")
    
    # 2. 读取并编码文本
    all_ids = []
    with open(input_file, 'r', encoding='utf-8') as f:
        # 如果文件非常大，建议分块处理
        # 这里的 f 本身就是一个字符串迭代器（按行读取）
        for token_id in tokenizer.encode_iterable(f):
            all_ids.append(token_id)
            
    # 3. 转换为 uint16 numpy 数组 (文档 Page 12 要求)
    ids_array = np.array(all_ids, dtype=np.uint16)
    
    # 4. 保存为二进制文件
    # 使用 tofile 直接保存原始字节，或者使用 np.save
    ids_array.tofile(output_file)
    
    print(f"处理完成！Token 总数: {len(ids_array)}")
    print(f"文件已保存至: {output_file}")

# 示例调用
if __name__ == "__main__":
    # 分别处理训练集和验证集
    from datasets import load_dataset
    ds = load_dataset("roneneldan/TinyStories")
    prepare_data(
        input_file="data/TinyStories_train.txt", 
        vocab_file="models/tokenizer_vocab.json",
        merges_file="models/tokenizer_merges.txt",
        output_file="data/train.bin"
    )
    prepare_data(
        input_file="data/TinyStories_val.txt", 
        vocab_file="models/tokenizer_vocab.json",
        merges_file="models/tokenizer_merges.txt",
        output_file="data/val.bin"
    )