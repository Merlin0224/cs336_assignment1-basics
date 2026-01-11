# 测试脚本片段
from cs336_basics.tokenizer import Tokenizer
tokenizer = Tokenizer.from_files("models/tokenizer_vocab.json", "models/tokenizer_merges.txt", ["<|endoftext|>"])
text = "Once upon a time, there was a little girl named Lily."
ids = tokenizer.encode(text)
decoded = tokenizer.decode(ids)

print(f"原始文本: {text}")
print(f"Token IDs: {ids}")
print(f"解码文本: {decoded}")
print(f"压缩比: {len(text.encode('utf-8')) / len(ids):.2f} 字节/Token")

# 原始文本: Once upon a time, there was a little girl named Lily.
# Token IDs: [439, 455, 259, 404, 44, 407, 282, 259, 405, 454, 509, 366, 46]
# 解码文本: Once upon a time, there was a little girl named Lily.
# 压缩比: 4.08 字节/Token