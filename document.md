

这份报告旨在记录并指导初学者如何从零开始构建一个现代 Transformer 语言模型。本实验基于斯坦福 CS336 课程框架，重点在于**从底层矩阵运算实现所有组件**，而非调用现成的库。

由于时间原因，本项目完整实现了前四个阶段（分词器、架构实现、训练基础设施、TinyStories 训练与生成），第五阶段（消融实验与排行榜冲刺）留作未来改进。显卡为3050，以下是简要的工作内容说明：


### 第一阶段：文本预处理与 BPE 分词器 (Section 2)

这是实验的基石。如果分词器有误，后续模型训练将无法收敛。


1.  **BPE 训练算法**：

    *   实现 `train_bpe` 函数。**关键点**：使用 `re.finditer` 进行预分词，并实现高效的合并（Merging）逻辑（通过缓存对（pair）计数来优化性能，否则在大语料库上会极慢）。

2.  **分词器类实现**：实现 `Tokenizer` 类，包含 `encode`（应用学习到的 merges）和 `decode`（处理非法字节序列时使用 `errors='replace'`）。

3.  **实验验证**：在 TinyStories 上训练分词器，计算压缩比（Compression Ratio），并确保通过 `test_tokenizer.py`。

  

### 第二阶段：模型架构开发 (Section 3)

这一阶段要求从零实现 Transformer 的各个组件。**注意：禁止使用 `torch.nn.Linear` 等高层 API。**

  

1.  **基础组件**：实现 `Linear`（不带 bias）和 `Embedding` 层，并使用 `trunc_normal_` 正确初始化。

2.  **规范化与激活**：实现 `RMSNorm`（注意 float32 向上转型防止溢出）和 `SwiGLU` 激活函数（FFN）。

3.  **旋转位置编码 (RoPE)**：这是模型中最复杂的部分之一。预计算 cos/sin 缓存，并实现对 query/key 的旋转。

4.  **注意力机制**：

    *   实现 `scaled_dot_product_attention`（支持 4D 张量和 boolean mask）。

    *   实现 `MultiHeadSelfAttention`（集成 Causal masking 和 RoPE）。

5.  **组装**：将以上组件整合为 `TransformerBlock`，最后构建完整的 `TransformerLM`。

  

### 第三阶段：训练基础设施 (Section 4 & 5)

有了模型，需要一套稳定的训练系统。


1.  **损失函数**：实现数值稳定的 `cross_entropy`（使用 log-sum-exp 技巧）。

2.  **优化器**：实现 `AdamW`。**关键点**：正确处理状态（first/second moments）和偏置修正（bias correction），以及权重衰减（weight decay）。

3.  **学习率调度**：实现带 Warm-up 的余弦退火（Cosine Annealing）。

4.  **数据加载与检查点**：

    *   实现高效的 `get_batch`（支持 `np.memmap` 以处理大数据集）。

    *   实现 `save_checkpoint` 和 `load_checkpoint`。

5.  **训练脚本**：编写 `train.py`，整合上述所有功能，并加入监控（如 Weights & Biases 或简单的日志记录）。

  

### 第四阶段：模型训练与生成 (Section 6 & 7.2)

1.  **TinyStories 训练**：

    *   使用文档建议的超参数（4 layers, 16 heads, d_model 512）。

    *   目标：验证集 loss 达到 1.45 以下。

2.  **文本生成**：实现带有 **Temperature（温度）** 和 **Nucleus Sampling（Top-p 采样）** 的解码函数。


  

### 第五阶段：消融实验与 Leaderboard 冲刺 (Section 7.3 - 7.5)

1.  **消融研究 (Ablations)**：

    *   移除 RMSNorm 看看会发生什么（训练稳定性实验）。

    *   Pre-norm vs Post-norm 比较。

    *   RoPE vs NoPE（验证位置编码的重要性）。

    *   SwiGLU vs SiLU（验证门控机制的效果）。

2.  **OpenWebText 训练**：在更真实的数据集上进行训练，观察模型泛化能力的差异。

  
  
  

---

  

# 第一阶段：文本预处理与 BPE 分词器 (Tokenizer)

  

分词器是语言模型的“眼睛”。它的质量直接决定了模型处理文本的效率和覆盖度。本项目实现了 **Byte-level BPE (Byte-Pair Encoding)**，这是一种能够处理任何 Unicode 字符串且永远不会产生“未知词（OOV）”的技术。



## 1. 核心理论：从字节开始


不同于早期的分词器，字节级 BPE 将文本视为 **UTF-8 字节流**。
  

### 1.1 Unicode 与 UTF-8 的处理

一个 Unicode 字符在 UTF-8 编码下占用 $1$ 到 $4$ 个字节。如果我们在错误的字节位置切分，会导致解码失败。因此，我们的逻辑是：

1.  将字符串编码为字节序列：$S \rightarrow [b_1, b_2, \dots, b_n]$，其中 $b_i \in [0, 255]$。

2.  在字节层面进行合并，直到合并后的单元能代表高频出现的子词。

  

### 1.2 压缩比 (Compression Ratio)

我们使用压缩比 $CR$ 来衡量分词器的性能。如果原始文本总字节数为 $B_{total}$，分词后的 Token 总数为 $T_{total}$，则：

$$CR = \frac{B_{total}}{T_{total}}$$

**目标：** 在词表大小（Vocab Size）一定的前提下，获得更高的 $CR$。

  

---

  

## 2. BPE 训练算法实现

  

训练过程是一个迭代合并最频繁相邻单元的贪心过程。

  

### 2.1 预分词 (Pre-tokenization)

为了防止跨单词边界合并（例如将“the”的结尾与“apple”的开头合并），我们使用 GPT-2 标准正则表达式进行预分词：

```python

GPT2_PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

```

这将文本切分为一系列“逻辑单词”块。

  

### 2.2 核心合并逻辑

算法流程如下：

1.  **统计频率**：统计所有相邻单元对 $(u_i, u_{i+1})$ 的出现次数。

2.  **选择最优对**：寻找出现频率最高的对 $(u_A, u_B)$。

3.  **平局决策 (Tie-breaking)**：如果多个对频率相同，根据实验要求，选择**字节字典序最大**的那一对：

    $$\text{Pair}_{best} = \arg\max_{(u_i, u_j)} \{ \text{Freq}(u_i, u_j), \text{Lexicographical}(u_i, u_j) \}$$

4.  **执行合并**：在整个语料库中将所有的 $(u_A, u_B)$ 替换为新 Token $u_{new}$，并更新词表：

    $$V_{new} = V_{old} \cup \{ u_A \cdot u_B \}$$

  

**train_bpe()** 具体流程如下：

#### 1. 词汇表初始化

首先建立基础词表，映射到原始的 256 个字节（`bytes([i])`）,紧接着字节 ID 分配特殊 Token（如 `<|endoftext|>`）

```python

    vocab = {i: bytes([i]) for i in range(256)}

    current_id = 256

    for st in special_tokens:

        vocab[current_id] = st.encode("utf-8")

        current_id += 1

```

#### 2. 文本隔离处理

为了防止合并逻辑破坏特殊 Token 的完整性,使用 `re.split` 根据 `special_tokens` 将原始文本切分成多个 **segments**(训练过程只在非特殊 Token 的文本段中进行)。

```python

if special_tokens:

        pattern = "|".join(re.escape(st) for st in special_tokens)

        segments = re.split(pattern, text)

    else:

        segments = [text]

```

  

#### 3. 预分词与计数优化

这是性能优化的关键点。

使用 `GPT2_PAT` 将文本段切分为“单词”块。这限制了合并操作只能在单词内部发生（例如，“the”的结尾不会和紧随其后的“apple”开头合并）。然后统计每个单词出现的次数，并将其存储为元组形式的字节 ID 序列。

```python

    word_counts = Counter()

    for segment in segments:

        if not segment: continue

        for word in re.findall(GPT2_PAT, segment):

            word_counts[tuple(word.encode("utf-8"))] += 1

```

#### 4. 统计相邻对频率

在每一轮迭代中：

*   遍历 `word_counts` 中的每一个单词元组。

*   统计所有相邻 ID 对 $(p_1, p_2)$ 出现的总频次：

    $$\text{Freq}(p_1, p_2) = \sum (\text{单词内对的出现次数} \times \text{单词总频次})$$

如果多个对的频次相同，使用 `key=lambda x: (x[1], vocab[x[0][0]], vocab[x[0][1]])` 进行排序。这表示当 `x[1]` (频率) 相同时，比较 `vocab[p1]` 和 `vocab[p2]` 的**字节内容字典序**。

  
  

#### 5. 合并更新 (Vocabulary Expansion & Merging)

选定 `best_pair` $(p_1, p_2)$ 后，将 $p_1$ 和 $p_2$ 的字节内容拼接，分配一个新的 ID，并将这次合并操作记录在 `merges` 列表中。遍历所有单词，将其中所有的 $(p_1, p_2)$ 替换为 `current_id`。

  
  

### 补充：多进程并行预分词 (Multiprocessing)

正则匹配是计算密集型任务，可以利用 Python 的 `multiprocessing.Pool` 将语料库切块并行处理，加快预分词。

```python
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

```



进程池中的工人函数：处理一个文本块并返回词频统计

```python
def _process_chunk(text_chunk):

    counts = Counter()

    # 使用 finditer 节省内存，避免产生巨大的列表

    for match in re.finditer(GPT2_PAT, text_chunk):

        word = match.group()

        # 将单词转为字节元组

        counts[tuple(word.encode("utf-8"))] += 1

    return counts
```
  
  

现在进入**第二阶段：模型架构开发**。这是本实验最硬核的部分。

  

不同于普通的深度学习项目，本项目要求**从零（From Scratch）实现**所有数学运算。这意味着你不能调用 `torch.nn.Linear` 这种“黑盒”，必须直接操作 `nn.Parameter` 和矩阵乘法。这种“手动挡”的实现方式能让你彻底理解张量在 Transformer 内部是如何流转的。

  
  

# 第二阶段：模型架构开发 —— 手动构建 Transformer

  

在这一阶段，我们实现了符合现代大模型标准（如 Llama 3）的 **Pre-norm Transformer** 架构。

  

## 1. 基础模块：手动实现线性层与嵌入层

  

### 1.1 参数初始化逻辑

神经网络的初始权重决定了训练能否成功。本项目严格遵守 **截断正态分布（Truncated Normal）** 初始化：

*   **Linear 层**：均值为 0，标准差 $\sigma = \sqrt{\frac{2}{d_{in} + d_{out}}}$。

*   **Embedding 层**：均值为 0，标准差为 1。

*   **截断范围**：所有参数均截断在 $[-3\sigma, 3\sigma]$ 范围内，以避免极值导致梯度爆炸。

  

### 1.2 线性变换 (Linear)

在不使用 `nn.Linear` 的情况下，我们手动管理权重矩阵 $W \in \mathbb{R}^{d_{out} \times d_{in}}$。

**数学公式**：

$$y = xW^\top$$

**代码提示**：为了支持批处理，推荐使用 `torch.einsum("...i, oi -> ...o", x, self.weight)`，它能完美处理任意数量的 batch 维度。

  

---

  

## 2. 归一化层：RMSNorm (Root Mean Square Layer Norm)

  

为了提升训练稳定性，我们采用了 **RMSNorm**，它比传统的 LayerNorm 更高效，因为它只进行缩放而不进行平移。

  

### 2.1 核心公式

对于一个输入向量 $a$，其归一化过程为：

$$\text{RMSNorm}(a_i) = \frac{a_i}{\sqrt{\frac{1}{d} \sum_{j=1}^{d} a_j^2 + \epsilon}} \cdot g_i$$

其中 $g$ 是可学习的增益（Gain）参数，初始化为 1。

  

### 2.2 数值稳定性优化

**向上转型（Upcasting）**：在计算平方和与平均值时，必须先将输入转为 `float32`，防止在 FP16/BF16 模式下发生溢出，计算完成后再转回原始类型。

  

---

  

## 3. 位置编码：RoPE (Rotary Position Embeddings)

  

RoPE 是现代 LLM 的标配，它通过旋转矩阵将相对位置信息注入到 Query 和 Key 中。

  

### 3.1 旋转逻辑

对于位置为 $i$ 的维度对 $(x_{2k}, x_{2k+1})$，变换公式为：

$$\begin{pmatrix} x'_{2k} \\ x'_{2k+1} \end{pmatrix} = \begin{pmatrix} \cos \theta_{i,k} & -\sin \theta_{i,k} \\ \sin \theta_{i,k} & \cos \theta_{i,k} \end{pmatrix} \begin{pmatrix} x_{2k} \\ x_{2k+1} \end{pmatrix}$$

其中频率 $\theta_{i,k} = i \cdot \Theta^{-2(k-1)/d}$。

  

### 3.2 高效实现

通过 `torch.arange` 预计算 `cos` 和 `sin` 缓存。在 `forward` 中利用切片 `[..., 0::2]` 和 `[..., 1::2]` 进行向量化运算，避免了低效的循环。

  

---

  

## 4. 激活函数：SwiGLU

  

我们使用了比 ReLU 更强大的 **SwiGLU** 激活函数。

  

### 4.1 数学公式

$$\text{FFN}(x) = W_2(\text{SiLU}(W_1 x) \odot W_3 x)$$

其中

*   $W_1$ 和 $W_3$ 是向上投影（从 $d_{model}$ 到 $d_{ff}$）。

*   $W_2$ 是向下投影（从 $d_{ff}$ 到 $d_{model}$）。

*   $\odot$ 表示元素级乘法（Hadamard product）。

*   $\text{SiLU}(x) = x \cdot \sigma(x)$（即 Swish 激活函数）。

  

### 4.2 维度对齐

为了硬件效率，$d_{ff}$（中间层维度）通常设为 $\frac{8}{3}d_{model}$。在本项目中，我们强制要求 $d_{ff}$ 为 **64 的倍数**，以充分利用英伟达显卡的 Tensor Cores。

  
  
  

## 5. 多头注意力机制 (Multi-Head Attention)

  

这是模型最复杂的部分，涉及频繁的形状变换。

  

### 5.1 执行流程

1.  **投影**：$Q, K, V$ 线性投影。

2.  **切分**：将 $d_{model}$ 拆分为 $h$ 个头，维度为 $d_k$。

3.  **RoPE**：仅对 $Q$ 和 $K$ 应用旋转。

4.  **因果掩码（Causal Mask）**：使用下三角矩阵防止模型“预知未来”：

    $$Mask_{ij} = \begin{cases} 0, & j \le i \\ -\infty, & j > i \end{cases}$$

5.  **计算分数**：$\text{Attention}(Q, K, V) = \text{softmax}(\frac{QK^\top}{\sqrt{d_k}} + Mask)V$。

  

---

  

## 6. 整合：TransformerBlock 与 TransformerLM

  

我们采用 **Pre-norm** 架构（归一化在残差连接之前），这比传统的 Post-norm 训练更容易收敛。

  

**一个 Block 的逻辑结构**：

1.  $x = x + \text{Attention}(\text{RMSNorm}(x))$

2.  $x = x + \text{FFN}(\text{RMSNorm}(x))$

  

最后，通过 `nn.ModuleList` 堆叠多个 Block，并添加 **Final RMSNorm** 和 **Output Linear (LM Head)**，形成完整的语言模型。

  

---

  

现在进入**第三阶段：训练基础设施（Training Infrastructure）**。

  

在拥有了“身体”（模型架构）之后，我们需要为它构建“大脑进化系统”。在这一阶段，我们将实现损失函数、优化器和数据流转系统。为了追求极致的数值稳定性，我们依然拒绝直接使用内置函数。

  

---

  

# 第三阶段：训练基础设施 —— 打造高效教练系统

  

本阶段的目标是实现一套支持大规模、高稳定性训练的工具链。

  

## 1. 损失函数：数值稳定的交叉熵 (Cross Entropy)

  

语言模型的训练目标是最大化预测下一个 Token 的概率。这通常转化为最小化负对数似然（Negative Log-Likelihood）。

  

### 1.1 核心公式

对于单个样本，交叉熵损失为：

$$\ell = -\log \left( \frac{\exp(o_{y})}{\sum_{j=1}^{V} \exp(o_j)} \right) = -o_y + \log \sum_{j=1}^{V} \exp(o_j)$$

其中 $V$ 是词表大小，$o_y$ 是正确类别对应的 Logit。

  

### 1.2 数值稳定技巧：Log-Sum-Exp

当 Logits $o_j$ 很大时，$\exp(o_j)$ 会发生数值溢出。我们采用 **Max-subtraction 技巧**：

$$\log \sum \exp(o_j) = \max(o) + \log \sum \exp(o_j - \max(o))$$

**工程实现**：在计算前向传播时，必须减去当前行的最大值。这确保了 $\exp$ 的指数永远小于或等于 0，结果处于 $(0, 1]$ 之间。

  

---

  

## 2. 优化器：手动实现 AdamW

  

AdamW 是当前大模型训练的行业标准。它修正了 Adam 优化器在处理权重衰减时的逻辑缺陷。

  

### 2.1 状态管理与更新规则

对于每个参数 $\theta$，我们维护一阶动量 $m_t$ 和二阶动量 $v_t$：

1.  **动量更新**：

    $$m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t$$

    $$v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2$$

2.  **偏差修正 (Bias Correction)**：解决训练初期 $m_t, v_t$ 偏向 0 的问题。

    $$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}$$

3.  **解耦权重衰减 (Decoupled Weight Decay)**：这是 AdamW 的核心，衰减直接作用于参数而非梯度。

    $$\theta_{t} = \theta_{t-1} - \eta \left( \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} + \lambda \theta_{t-1} \right)$$

    其中 $\eta$ 是学习率，$\lambda$ 是权重衰减率。

  

---

  

## 3. 学习率调度：余弦退火与 Warmup

  

为了防止训练初期巨大的梯度破坏随机初始化的参数，我们引入了 **Linear Warmup**，随后接 **Cosine Annealing**。

  

### 3.1 调度逻辑

*   **Warmup 阶段** ($t < T_w$)：学习率从 0 线性增加到 $\alpha_{max}$。

*   **退火阶段** ($T_w \le t \le T_c$)：

    $$\alpha_t = \alpha_{min} + \frac{1}{2}(\alpha_{max} - \alpha_{min})\left(1 + \cos\left(\frac{t - T_w}{T_c - T_w}\pi\right)\right)$$

*   **收益**：Warmup 提供了“热身”，余弦退火则确保了模型在训练后期能够平滑地收敛到局部最优。

  

---

  

## 4. 梯度裁剪 (Gradient Clipping)

  

在处理长序列或深层网络时，梯度范数可能瞬间激增。

**算法**：计算所有参数梯度的全局 $L_2$ 范数 $\|g\|_2$。若 $\|g\|_2 > M$，则：

$$g \leftarrow g \cdot \frac{M}{\|g\|_2 + \epsilon}$$

这就像给模型加了一个“保险丝”，防止单次更新幅度过大导致模型崩坏。

  

---

  

## 5. 数据加载与持久化 (DataLoader & Checkpointing)

  

### 5.1 内存映射技术 (Memory Mapping)

在 RTX 3050 的有限内存环境下，我们无法将 4.6 亿个 Token 全部读入内存。

*   **方案**：使用 `np.memmap` 直接在磁盘上操作二进制文件。

*   **优势**：操作系统会负责缓存管理，我们只需要维护一个指针随机采样 Batch 即可，内存占用极低且恒定。

```python
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
```

### 5.2 状态快照 (Checkpointing)

训练可能持续数小时。我们需要保存一个包含以下内容的字典：

*   `model_state_dict`：神经元权重。

*   `optimizer_state_dict`：AdamW 的 $m_t$ 和 $v_t$（不保存会导致恢复后学习率震荡）。

*   `iteration`：当前进度。


---

# 第四阶段：模型训练与生成 

在本阶段，我们将前三个阶段构建的所有组件整合进 `train.py`。

## 1. 训练策略：平衡噪声与效率

在我们的 `train.py` 实现中，选择了最直接、最透明的训练流程。

### 1.1 批次配置
```python
# 模型配置

vocab_size = 10000

context_length = 256

num_layers = 4

num_heads = 16

d_model = 512

d_ff = 1344

theta = 10000.0

  

# 训练配置

batch_size = 16  # 128，如果显存不够(OOM)，调小至 64 或 32以下

max_iters = 10000 * 8 # 步数 = 总Token / (batch_size * context_length)

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
```

### 1.2 核心训练循环

代码严格遵循以下标准流程：

1.  **动态学习率更新**: 每一步迭代都通过 `get_lr_cosine_schedule` 手动计算并覆盖 `optimizer.param_groups` 中的 `lr`。
2.  **前向传播与 Loss 计算**: 使用自定义 `cross_entropy` 计算 Logits 与 Targets 的距离。
3.  **梯度裁剪 (Gradient Clipping)**: 在 `optimizer.step()` 之前，强制执行 $\|g\|_2 \le 1.0$。这一步对于小 Batch 训练至关重要，它抵消了采样随机性带来的梯度冲击。
```python
start_time = time.time()

best_val_loss = float('inf')

  

for it in range(max_iters):

    # 更新学习率

    lr = get_lr_cosine_schedule(it, learning_rate, min_lr, warmup_iters, lr_decay_iters)

    for param_group in optimizer.param_groups:

        param_group['lr'] = lr

  

    # 前向传播

    X, Y = get_batch(train_data)

    logits = model(X)

    loss = cross_entropy(logits, Y)

  

    # 反向传播

    optimizer.zero_grad()

    loss.backward()

    # 梯度裁剪 (Page 34)

    gradient_clipping(model.parameters(), max_norm=1.0)

    # 优化步

    optimizer.step()
```
---

## 2. 指标分析：困惑度 (Perplexity)

我们不仅监控训练 Loss，更关注验证集上的 **Perplexity (PP)**。

### 2.1 计算公式
$$PP = \exp(\text{Validation Loss})$$
### 2.2 实验数据
*   **初始阶段**: `val_loss` 为 9.23，`perplexity` 约为 10,239。
*   **最终阶段 (Iter 10000)**: `val_loss` 降至 1.90 左右，`perplexity` 降至约 **6.68**。
*   **结论**: 困惑度的迅速下降证明了模型从“随机乱撞”到“精准预测”的进化。

```python
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

            save_checkpoint(model, optimizer, it, "models/best_model.pt")
```

---

## 3. 文本生成：自回归解码逻辑

生成的质量是检验模型的唯一标准。我们在 `TransformerLM` 中实现的 `generate` 函数具备以下特性：

1.  **Context Window 动态滑动**: 在生成新 Token 时，始终通过 `curr_ids[:, -self.context_length:]` 截取最新的 256 个 Token。这确保了在长文本生成时，RoPE 旋转位置编码不会超出预计算的缓存边界。
2.  **Top-p (核采样)**: 设定 `top_p=0.9`，有效地过滤了 Logits 分布中的长尾噪声，使生成的故事在逻辑连贯的同时具备多样性。
```python
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
```
---
### 补充：**梯度累加 (Gradient Accumulation)** 和 **自动混合精度 (AMP)**
如果显存太小，可以在训练过程中采用以下方法：
```python
scaler = torch.amp.GradScaler('cuda') # 初始化缩放器

start_time = time.time()

best_val_loss = float('inf')

accumulation_steps = 8  # 16 * 8 = 128 (模拟文档建议的有效 batch size)

  

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
```
效果很好，缩短了近一半的时间。