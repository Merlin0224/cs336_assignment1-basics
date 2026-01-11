import json
import os
import regex as re
from collections import Counter, defaultdict
from typing import Iterable, Iterator
import multiprocessing
from functools import partial

GPT2_PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

class Tokenizer:
    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens=None):
        self.vocab = vocab
        self.byte_to_id = {v: k for k, v in vocab.items()}
        self.merges_dict = {pair: i for i, pair in enumerate(merges)}
        self.special_tokens = special_tokens or []
        
        # 1. 重要：将特殊 Token 添加到词表中（如果不存在）
        # 文档要求：appending them to the vocabulary if they aren't already there (Page 11)
        current_max_id = max(vocab.keys()) if vocab else -1
        for st in self.special_tokens:
            st_bytes = st.encode("utf-8")
            if st_bytes not in self.byte_to_id:
                current_max_id += 1
                self.vocab[current_max_id] = st_bytes
                self.byte_to_id[st_bytes] = current_max_id

        # 2. 优化：按长度从长到短排序特殊 Token，解决重叠匹配优先级问题
        if self.special_tokens:
            sorted_specials = sorted(self.special_tokens, key=len, reverse=True)
            self.special_pattern = re.compile("|".join(re.escape(st) for st in sorted_specials))
        else:
            self.special_pattern = None

    def train(self, input_path: str, vocab_size: int, special_tokens: list[str] = None):
        pass

    def encode(self, text: str) -> list[int]:
        if not text:
            return []
        if not self.special_pattern:
            return self._encode_chunk(text)
        
        ids = []
        last_pos = 0
        # 使用 finditer 配合排序后的正则，确保优先匹配最长的特殊 token
        for match in self.special_pattern.finditer(text):
            # 处理特殊 token 之间的普通文本
            ids.extend(self._encode_chunk(text[last_pos:match.start()]))
            # 添加特殊 token 的 ID
            special_bytes = match.group().encode("utf-8")
            ids.append(self.byte_to_id[special_bytes])
            last_pos = match.end()
        # 处理剩余文本
        ids.extend(self._encode_chunk(text[last_pos:]))
        return ids

    def _encode_chunk(self, text: str) -> list[int]:
        """修正初始 ID 映射的高效编码逻辑"""
        if not text:
            return []
        
        words = re.findall(GPT2_PAT, text)
        all_token_ids = []
        
        for word in words:
            # --- 修正点：根据词表映射初始字节 ID，而不是直接使用字节值 ---
            word_bytes = word.encode("utf-8")
            ids = [self.byte_to_id[bytes([b])] for b in word_bytes]
            
            # 高效合并算法
            while len(ids) >= 2:
                best_pair = None
                min_rank = float('inf')
                
                # 找出当前序列中合并优先级最高（rank最小）的 pair
                for i in range(len(ids) - 1):
                    pair = (self.vocab[ids[i]], self.vocab[ids[i+1]])
                    rank = self.merges_dict.get(pair)
                    if rank is not None and rank < min_rank:
                        min_rank = rank
                        best_pair = (ids[i], ids[i+1])
                
                if best_pair is None:
                    break
                
                # 执行替换
                new_ids = []
                i = 0
                p1, p2 = best_pair
                merged_id = self.byte_to_id[self.vocab[p1] + self.vocab[p2]]
                while i < len(ids):
                    if i < len(ids) - 1 and ids[i] == p1 and ids[i+1] == p2:
                        new_ids.append(merged_id)
                        i += 2
                    else:
                        new_ids.append(ids[i])
                        i += 1
                ids = new_ids
            
            all_token_ids.extend(ids)
        return all_token_ids

    def decode(self, ids: list[int]) -> str:
        # 使用 errors='replace' 处理非法的字节序列
        byte_data = b"".join(self.vocab[i] for i in ids if i in self.vocab)
        return byte_data.decode("utf-8", errors="replace")

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text in iterable:
            yield from self.encode(text)

    
    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        """
        符合 Problem (tokenizer) 的类方法 (Page 11-12)
        """
        # 1. 加载词表 (Vocab)
        with open(vocab_filepath, "r", encoding="utf-8") as f:
            raw_vocab = json.load(f)
        
        # 关键转换：JSON 键是字符串 -> 转回 int；值是十六进制字符串 -> 转回 bytes
        vocab = {int(k): bytes.fromhex(v) for k, v in raw_vocab.items()}

        # 2. 加载合并规则 (Merges)
        merges = []
        if os.path.exists(merges_filepath):
            with open(merges_filepath, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    # 将十六进制字符串转回 bytes 元组
                    parts = line.split(" ")
                    if len(parts) == 2:
                        merges.append((bytes.fromhex(parts[0]), bytes.fromhex(parts[1])))
        
        return cls(vocab, merges, special_tokens=special_tokens)
    
    def save_to_files(self, vocab_filepath, merges_filepath):
        # 保存词表：bytes 转为 hex 字符串
        serializable_vocab = {k: v.hex() for k, v in self.vocab.items()}
        with open(vocab_filepath, "w", encoding="utf-8") as f:
            json.dump(serializable_vocab, f, indent=4)
        
        # 保存合并规则：每个 bytes 元组存为一行 hex
        with open(merges_filepath, "w", encoding="utf-8") as f:
            for p1, p2 in self.merges:
                f.write(f"{p1.hex()} {p2.hex()}\n")
    
    



# def train_bpe(input_path: str, vocab_size: int, special_tokens: list[str]):
#     # 1. 初始化词表：0-255 固定为基础字节 (符合 encode 默认行为)
#     vocab = {i: bytes([i]) for i in range(256)}
    
#     # 2. 特殊 Token 从 256 开始编号
#     current_id = 256
#     for st in special_tokens:
#         vocab[current_id] = st.encode("utf-8")
#         current_id += 1
    
#     # 3. 读取文本
#     with open(input_path, "r", encoding="utf-8") as f:
#         text = f.read()

#     # 4. 隔离特殊 Token (Page 8)
#     # 只有非特殊 Token 的文本块参与 BPE 训练
#     if special_tokens:
#         # 使用 | 组合所有特殊 token，并用括号捕获以便 split 后保留它们（虽然训练时不直接用它们）
#         # 但更简单的方法是直接 split，只对剩下的 segment 进行正则表达式分词
#         pattern = "|".join(re.escape(st) for st in special_tokens)
#         segments = re.split(pattern, text)
#     else:
#         segments = [text]

#     # 5. 预分词计数
#     word_counts = Counter()
#     for segment in segments:
#         if not segment: continue
#         # 使用 GPT-2 正则表达式 (Page 6)
#         for word in re.findall(GPT2_PAT, segment):
#             # 字节直接转为 ID 序列 (0-255)
#             word_counts[tuple(word.encode("utf-8"))] += 1

#     merges = []
#     # 目标合并次数 = 最终词表大小 - 当前已有大小
#     num_merges = vocab_size - len(vocab)
    
#     for _ in range(num_merges):
#         pair_counts = defaultdict(int)
#         for word_tuple, count in word_counts.items():
#             for i in range(len(word_tuple) - 1):
#                 pair = (word_tuple[i], word_tuple[i+1])
#                 pair_counts[pair] += count
        
#         if not pair_counts:
#             break
            
#         # 6. Tie-breaking: 频率相同时，选字节内容字典序最大的 (Page 7)
#         # 注意：这里比较的是两个 ID 对应的字节块拼接后的内容，或元组级字节比较
#         best_pair = max(
#             pair_counts.items(), 
#             key=lambda x: (x[1], vocab[x[0][0]], vocab[x[0][1]])
#         )[0]
        
#         p1, p2 = best_pair
#         new_token_bytes = vocab[p1] + vocab[p2]
#         vocab[current_id] = new_token_bytes
#         # merges 存储 bytes 元组
#         merges.append((vocab[p1], vocab[p2]))
        
#         # 7. 更新 word_counts (只在包含 p1 的单词中进行替换以优化速度)
#         new_word_counts = Counter()
#         for word_tuple, count in word_counts.items():
#             if p1 not in word_tuple:
#                 new_word_counts[word_tuple] = count
#                 continue
                
#             new_word_tuple = []
#             i = 0
#             while i < len(word_tuple):
#                 if i < len(word_tuple) - 1 and word_tuple[i] == p1 and word_tuple[i+1] == p2:
#                     new_word_tuple.append(current_id)
#                     i += 2
#                 else:
#                     new_word_tuple.append(word_tuple[i])
#                     i += 1
#             new_word_counts[tuple(new_word_tuple)] = count
        
#         word_counts = new_word_counts
#         current_id += 1
        
#     return vocab, merges
def _process_chunk(text_chunk):
    """
    进程池中的工人函数：处理一个文本块并返回词频统计
    """
    counts = Counter()
    # 使用 finditer 节省内存，避免产生巨大的列表
    for match in re.finditer(GPT2_PAT, text_chunk):
        word = match.group()
        # 将单词转为字节元组
        counts[tuple(word.encode("utf-8"))] += 1
    return counts

def train_bpe(input_path: str, vocab_size: int, special_tokens: list[str]):
    # 1. 初始化词表 (0-255 为基础字节)
    vocab = {i: bytes([i]) for i in range(256)}
    current_id = 256
    # 特殊 Token 紧随其后
    for st in special_tokens:
        vocab[current_id] = st.encode("utf-8")
        current_id += 1

    # 2. 读取并分块处理 (Page 8 & 10)
    print(f"正在读取文件并进行并行预分词...")
    with open(input_path, "r", encoding="utf-8") as f:
        full_text = f.read()

    # 按照 <|endoftext|> 切分文档，确保不会跨文档合并
    # 如果 special_tokens 有多个，取第一个作为分隔符（通常是 TinyStories 的需求）
    sep = special_tokens[0] if special_tokens else None
    if sep and sep in full_text:
        # 我们保留分隔符，但在统计时不合并它
        documents = full_text.split(sep)
    else:
        documents = [full_text]

    # 使用进程池并行统计词频
    num_workers = max(1, multiprocessing.cpu_count() - 1)
    # 将文档列表分成 num_workers 个块
    chunk_size = (len(documents) + num_workers - 1) // num_workers
    doc_chunks = [sep.join(documents[i:i + chunk_size]) for i in range(0, len(documents), chunk_size)]

    with multiprocessing.Pool(processes=num_workers) as pool:
        # 并行执行 _process_chunk
        chunk_counters = pool.map(_process_chunk, doc_chunks)

    # 聚合所有进程的结果
    word_counts = Counter()
    for c in chunk_counters:
        word_counts.update(c)
    
    # 释放内存
    del full_text
    del documents
    del doc_chunks

    # 3. 迭代合并 (这部分在 Python 中无法并行，但 word_counts 已经大幅缩小了数据规模)
    merges = []
    num_merges = vocab_size - len(vocab)
    
    # 预先提取 ID 到字节的映射用于决胜逻辑
    # 优化：迭代过程中维护 pair_counts
    pair_counts = defaultdict(int)
    for word_tuple, count in word_counts.items():
        for i in range(len(word_tuple) - 1):
            pair_counts[(word_tuple[i], word_tuple[i+1])] += count

    print(f"开始 BPE 合并迭代...")
    for _ in range(num_merges):
        if not pair_counts:
            break
            
        # Tie-breaking: 频率高者优先；频率相同时，字节内容字典序大者优先 (Page 7)
        # 这里的 vocab[p[0]] 是 bytes 类型
        best_pair = max(
            pair_counts.items(), 
            key=lambda x: (x[1], vocab[x[0][0]], vocab[x[0][1]])
        )[0]
        
        p1, p2 = best_pair
        new_token_bytes = vocab[p1] + vocab[p2]
        vocab[current_id] = new_token_bytes
        merges.append((vocab[p1], vocab[p2]))
        
        # --- 高效更新逻辑 (Page 8 Optimizing the merging step) ---
        new_word_counts = Counter()
        for word_tuple, count in word_counts.items():
            if p1 not in word_tuple:
                new_word_counts[word_tuple] = count
                continue
            
            # 执行合并
            new_word = []
            i = 0
            while i < len(word_tuple):
                if i < len(word_tuple) - 1 and word_tuple[i] == p1 and word_tuple[i+1] == p2:
                    # 在更新 word_counts 的同时，更新 pair_counts 是一项进阶优化
                    # 这里为了代码清晰，采用每轮重算 pair_counts 的折中方案
                    # 如果还是慢，可以尝试在合并时原位加减 pair_counts 的计数
                    new_word.append(current_id)
                    i += 2
                else:
                    new_word.append(word_tuple[i])
                    i += 1
            new_word_counts[tuple(new_word)] = count
        
        word_counts = new_word_counts
        
        # 重新统计 pair_counts (如果 num_merges 很大，这步建议改为增量更新)
        pair_counts = defaultdict(int)
        for word_tuple, count in word_counts.items():
            for i in range(len(word_tuple) - 1):
                pair_counts[(word_tuple[i], word_tuple[i+1])] += count
        
        current_id += 1
        if (len(merges)) % 100 == 0:
            print(f"已完成 {len(merges)} 次合并...")
        
    return vocab, merges
