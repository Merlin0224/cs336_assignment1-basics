import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from datasets import load_dataset
from tqdm import tqdm

def export_to_txt(split, output_file):
    print(f"正在下载并转换 {split} 分片...")
    dataset = load_dataset("roneneldan/TinyStories", split=split)
    
    with open(output_file, "w", encoding="utf-8") as f:
        for item in tqdm(dataset):
            # 提取故事内容
            story = item['text'].strip()
            # 按照作业要求，每个故事后面添加结束符
            # 这样 BPE 训练和数据加载时能正确识别边界
            f.write(story + "<|endoftext|>")

if __name__ == "__main__":
    export_to_txt("train", "data/TinyStories_train.txt")
    export_to_txt("validation", "data/TinyStories_val.txt")