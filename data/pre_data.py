import os
import numpy as np
from tqdm import tqdm
from cs336_basics.tokenizer import Tokenizer 

def process_data(input_path, output_path, tokenizer):
    print(f"正在处理: {input_path}")
    
    # 统计总行数以便显示进度条 (可选)
    line_count = sum(1 for _ in open(input_path, 'r', encoding='utf-8'))
    
    ids = []
    # 使用 open 配合 encode_iterable。f 本身就是一个行迭代器
    with open(input_path, 'r', encoding='utf-8') as f:
        # 逐行处理可以保持较低的内存占用
        # encode_iterable 会内部调用 encode
        for token_id in tqdm(tokenizer.encode_iterable(f), desc="Tokenizing"):
            ids.append(token_id)
            
    # 转换为 numpy 数组并指定为 uint16 (文档 Page 12 要求)
    ids_ndarray = np.array(ids, dtype=np.uint16)
    
    # 使用 tofile 保存为原始二进制格式，这比 np.save 更通用，
    # 且方便后面用 np.memmap 以 'r' 模式读取
    ids_ndarray.tofile(output_path)
    
    print(f"处理完成！")
    print(f"Token 总数: {len(ids_ndarray):,}")
    print(f"二进制文件大小: {os.path.getsize(output_path) / (1024*1024):.2f} MB")

def main():
    # 1. 加载训练好的分词器
    tokenizer = Tokenizer.from_files(
        vocab_filepath="models/tokenizer_vocab.json",
        merges_filepath="models/tokenizer_merges.txt",
        special_tokens=["<|endoftext|>"]
    )
    
    # 2. 处理训练集和验证集
    os.makedirs("data/processed", exist_ok=True)
    
    process_data("data/TinyStories_train.txt", "data/processed/train.bin", tokenizer)
    process_data("data/TinyStories_val.txt", "data/processed/val.bin", tokenizer)

if __name__ == "__main__":
    main()