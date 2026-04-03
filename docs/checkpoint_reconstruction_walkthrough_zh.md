# AlphaRL Checkpoint Reconstruction 源码逐段讲解

这份文档按“真实执行路径”解释这个仓库，适合已经会 Python / PyTorch，但第一次接触本项目的读者。重点不是复述论文，而是帮助你把仓库里的文件、函数、数据目录和运行脚本连成一条完整主线。

## 1. 先纠正项目边界

这个仓库**不是** AlphaRL 训练框架本体，也没有实现 RL 训练循环、采样器、reward model、PPO / GRPO / DAPO 优化器这些训练侧模块。

它当前实现的是一个很聚焦的工具：

1. 读取一个 `base` 模型目录和一个 `full` 模型目录。
2. 对同名参数计算增量 `delta = trained - base`。
3. 对每个二维权重矩阵做低秩 SVD 近似，只保留前 `k%` 奇异子空间。
4. 用重建后的增量加回 base 权重。
5. 把结果重新写成 Hugging Face 风格的 safetensors 分片 checkpoint。

所以它更准确的名字应该是“checkpoint reconstruction utility”，而不是“AlphaRL 全量训练代码”。

## 2. 仓库结构怎么看

先看几个最关键的路径：

- `README.md`
  - 这里更像原始需求说明，讲的是任务背景、输入输出和希望处理的实验对象。
- `alpharl/__init__.py`
  - 包的导出入口，告诉你这个库真正暴露给外部的 API 很少。
- `alpharl/checkpoint_reconstruction.py`
  - 核心实现几乎都在这里，后面会按执行顺序逐段拆开。
- `run_dapo32b_top1.sh`
  - 实际运行入口，负责做依赖检查并调用 Python 模块。
- `tests/test_checkpoint_reconstruction.py`
  - 最小可运行例子，也是理解主流程最省时间的地方。
- `dapo32b/base/`
  - 基础模型 checkpoint。
- `dapo32b/full/`
  - 经过 RL 训练后的完整 checkpoint。
- `dapo32b/approx_top1/`
  - 预期输出目录。在当前仓库里，它已经包含配置和 tokenizer 等支持文件，但没有看到完整重建权重；正常跑完脚本后，这里应当再出现新的 safetensors 分片和 `model.safetensors.index.json`。

一个最重要的认知是：`dapo32b/` 下面这些目录**不是源码**，而是模型数据。真正的程序逻辑集中在 `alpharl/` 和 `tests/`。

## 3. 整个程序从哪开始跑

完整执行链很短，只有一条主线：

1. 你运行 `run_dapo32b_top1.sh`。
2. shell 脚本先检查 `torch` 和 `safetensors` 有没有安装。
3. 然后它执行 `python -m alpharl.checkpoint_reconstruction ...`。
4. Python 进入 `main()`，把命令行参数装进 `ReconstructionConfig`。
5. `main()` 调 `reconstruct_checkpoint(config)`。
6. `reconstruct_checkpoint()` 读取 base / trained 的索引文件，逐个 tensor 做重建。
7. 每处理完一个输出 shard，就写出一个新的 `.safetensors` 分片。
8. 最后再写索引文件和 `reconstruction_stats.json`。

从工程角度讲，这是一个典型的“脚本入口 + 单核心模块 + 一个端到端测试”的小项目。

## 4. 包入口很薄：`alpharl/__init__.py`

先看 [`alpharl/__init__.py`](../alpharl/__init__.py)。

这个文件只有 5 行，重点在第 3-5 行：

- `ReconstructionConfig`
- `main`
- `reconstruct_checkpoint`

也就是说，作者希望别人把这个包当成一个非常小的工具库来用：

- 如果你想在 Python 里直接调用，就用 `ReconstructionConfig` 和 `reconstruct_checkpoint`。
- 如果你想从命令行跑，就走 `main` 对应的模块入口。

这也反过来说明，真正值得精读的文件就是 `alpharl/checkpoint_reconstruction.py`。

## 5. 运行脚本：`run_dapo32b_top1.sh`

看 [`run_dapo32b_top1.sh`](../run_dapo32b_top1.sh) 第 1-37 行。

### 5.1 第 1-5 行：标准 shell 脚本壳子

- `#!/usr/bin/env bash`
  - 指定用 bash 解释脚本。
- `set -euo pipefail`
  - `-e`：任一命令失败就退出。
  - `-u`：未定义变量直接报错。
  - `pipefail`：管道里任何一步失败都算失败。
- `ROOT_DIR=...` 和 `cd "$ROOT_DIR"`
  - 确保你从任何目录调用脚本，最终都在仓库根目录执行。

### 5.2 第 7-18 行：运行前做依赖探测

这段内嵌 Python 不做重建，只检查两个包：

- `torch`
- `safetensors`

如果没装，脚本会直接退出，并给出缺失依赖提示。这样做的好处是：你不会等到 Python 模块跑到一半才发现依赖缺失。

### 5.3 第 20-27 行：默认参数

这里把环境变量和默认值绑定起来：

- `BASE="${BASE:-dapo32b/base}"`
- `FULL="${FULL:-dapo32b/full}"`
- `OUT="${OUT:-dapo32b/approx_top1}"`
- `KEEP_RATIO="${KEEP_RATIO:-0.01}"`
- `DEVICE="${DEVICE:-cuda:0}"`
- `SVD_OVERSAMPLE="${SVD_OVERSAMPLE:-16}"`
- `SVD_NITER="${SVD_NITER:-2}"`
- `SEED="${SEED:-0}"`

这意味着脚本默认就是给你跑“Qwen2.5-32B base + DAPO full -> 输出 top 1% 近似模型”的实验。

### 5.4 第 29-37 行：真正进入 Python

最后一段调用：

```bash
python -m alpharl.checkpoint_reconstruction ...
```

这里没有额外包装逻辑，shell 只负责把参数传给 Python 模块。所以理解脚本之后，核心就全部转移到 `alpharl/checkpoint_reconstruction.py`。

## 6. 核心模块总览：`alpharl/checkpoint_reconstruction.py`

这个文件可以按执行顺序拆成 8 组内容：

1. 可选依赖导入与常量。
2. 配置对象 `ReconstructionConfig`。
3. 分片读取器 `SafeTensorReader`。
4. 若干辅助函数：依赖检查、索引读取、分组、参数校验、设备选择、随机种子。
5. 文件复制逻辑。
6. 单个 tensor 的低秩重建逻辑。
7. 整个 checkpoint 的重建主循环。
8. CLI 参数解析和 `main()`。

下面按真实调用顺序逐段讲。

## 7. 顶部导入和常量

看 `alpharl/checkpoint_reconstruction.py:1-27`。

### 7.1 为什么 `torch` 和 `safetensors` 用 try / except 导入

文件顶部没有直接硬导入：

- `torch`
- `safetensors.safe_open`
- `safetensors.torch.save_file`

而是写成：

```python
try:
    import torch
except ImportError:
    torch = None
```

原因是这个模块希望“可以被导入，但在真正运行前再检查依赖”。这样做有两个好处：

- 代码导入阶段更稳，不会因为缺依赖导致整个模块完全不可见。
- 测试里可以针对“依赖缺失”的情况做条件跳过。

### 7.2 两个常量的意义

- `WEIGHT_INDEX_NAME = "model.safetensors.index.json"`
  - Hugging Face 分片 checkpoint 的索引文件名。
- `SAFE_TENSORS_METADATA = {"format": "pt"}`
  - 写新 shard 时带上的 metadata，表明这是 PyTorch 风格 tensor。

这两个常量很朴素，但让后面代码少写硬编码字符串。

## 8. 配置对象：`ReconstructionConfig`

看 `alpharl/checkpoint_reconstruction.py:30-39`。

这是一个 `@dataclass(frozen=True)`，字段包括：

- `base_model_path`
- `trained_model_path`
- `output_path`
- `keep_ratio`
- `device`
- `svd_oversample`
- `svd_niter`
- `seed`

### 8.1 为什么用 dataclass

这里用 dataclass 的好处是：

- 参数集中，不会在函数间传很多散乱的位置参数。
- 可读性强，创建配置对象时语义很清楚。
- `frozen=True` 表示配置创建后不再被修改，减少流程中参数被意外篡改的风险。

### 8.2 这几个参数分别影响什么

- `keep_ratio`
  - 决定保留多少奇异子空间。
- `device`
  - 决定 SVD 在 CPU 还是 GPU 上做。
- `svd_oversample`
  - 传给 `torch.svd_lowrank` 的额外采样维度，影响低秩近似质量。
- `svd_niter`
  - 随机低秩 SVD 的幂迭代次数，增加后通常更准，但更慢。
- `seed`
  - 让随机 SVD 的结果尽量可复现。

## 9. 分片读取器：`SafeTensorReader`

看 `alpharl/checkpoint_reconstruction.py:42-71`。

这类代码很容易第一次读不明白，因为它不是数学逻辑，而是 I/O 优化逻辑。

### 9.1 它想解决什么问题

大模型 checkpoint 通常被切成多个 `.safetensors` shard。假设你有几千个 tensor，如果每读一个 tensor 都重新打开一次 shard 文件，I/O 会非常浪费。

`SafeTensorReader` 做的事情是：

1. 记录 `tensor_name -> shard_name` 的映射。
2. 第一次访问某个 shard 时，打开这个文件。
3. 把打开后的 handle 缓存起来。
4. 之后同一个 shard 里的其他 tensor 直接复用这个 handle。
5. 整个 reader 用完后统一关闭。

### 9.2 `get_tensor()` 的关键逻辑

`get_tensor()` 在第 51-59 行：

1. 先从 `weight_map` 查 tensor 属于哪个 shard。
2. 看这个 shard 的 handle 有没有缓存。
3. 如果没有，就用 `safe_open(...)` 打开它。
4. 把 context 和 handle 都缓存起来。
5. 最后返回 `handle.get_tensor(tensor_name)`。

这里之所以同时缓存 `_contexts` 和 `_handles`，是因为 `safe_open` 本质上是一个上下文管理器，关闭时要通过 context 的 `__exit__()` 来完成。

### 9.3 为什么它支持 `with`

第 67-71 行实现了 `__enter__` / `__exit__`，所以可以这样用：

```python
with SafeTensorReader(...) as reader:
    tensor = reader.get_tensor(...)
```

这能保证无论中间是否抛异常，最后都能把打开的 shard 关掉。

## 10. 依赖检查、索引读取和预处理

这部分主要看 `alpharl/checkpoint_reconstruction.py:74-137`。

### 10.1 `require_dependencies()`：晚一点报错，但报得更清楚

第 74-85 行做的事情很直接：

- 如果 `torch is None`，记录缺 `torch`。
- 如果 `safe_open` 或 `save_file` 缺失，记录缺 `safetensors`。
- 最后统一抛一个 `RuntimeError`。

这样做比顶部硬崩更友好，因为错误信息是面向“运行者”的，而不是 Python 导入栈。

### 10.2 `read_weight_index()`：先读索引，再谈权重

第 88-93 行读取 `model.safetensors.index.json`。

这个函数体现了一个关键设计：**程序不是遍历目录里所有 safetensors 文件来猜有哪些张量，而是以索引文件为准**。这是更稳妥的做法，因为：

- 索引显式告诉你有哪些 tensor。
- 它告诉你每个 tensor 属于哪个 shard。
- 它避免了依赖文件命名顺序或磁盘遍历顺序。

### 10.3 `group_tensors_by_shard()`：为什么要按 shard 分组

第 96-100 行把：

```python
tensor_name -> shard_name
```

转成：

```python
shard_name -> [tensor_name, ...]
```

这么做是为了后面“按输出 shard 一次性写文件”。如果不先分组，程序就很难做到每个输出 shard 只写一次。

### 10.4 `validate_config()`：提前拦掉危险输入

第 103-115 行做了三类保护：

- `keep_ratio` 必须在 `(0, 1]`。
- `svd_oversample`、`svd_niter` 不能为负数。
- `output_path` 不能和 `base_model_path`、`trained_model_path` 相同。

最后这一点非常关键。因为这个程序会写文件，如果输出目录和输入目录相同，就有机会把原 checkpoint 覆盖掉。

### 10.5 `select_device()`：请求 GPU，但允许优雅回退

第 118-128 行不是无脑信任 `--device`，而是这样判断：

1. 如果请求的是 `cuda...`
2. 先看 `torch.cuda.is_available()`
3. 再试着在这个 device 上分配一个空 tensor
4. 如果失败，就回退到 `"cpu"`

这让脚本更耐用。比如你传了 `cuda:0`，但当前环境没有 GPU，程序不会直接挂，而是继续在 CPU 上跑。

### 10.6 `seed_everything()`：为什么这里也要设随机种子

第 131-136 行设置：

- Python `random`
- `torch.manual_seed`
- `torch.cuda.manual_seed_all`

原因是 `torch.svd_lowrank` 使用的是随机化低秩分解。虽然不是每一步都完全随机，但固定 seed 仍然有助于提高可复现性。

## 11. 支持文件复制：`copy_support_files()`

看 `alpharl/checkpoint_reconstruction.py:139-158`。

这个函数很容易被忽略，但它决定了输出目录是不是一个“能被正常加载”的模型目录。

### 11.1 它复制什么

它遍历 `trained_model_path` 下的文件，只跳过两类：

- `model.safetensors.index.json`
- `*.safetensors`

剩下的文件全部复制到输出目录，比如：

- `config.json`
- `generation_config.json`
- `tokenizer.json`
- `tokenizer_config.json`
- `vocab.json`
- `merges.txt`
- `special_tokens_map.json`
- `README.md`

### 11.2 为什么权重文件和索引文件不在这里复制

因为：

- 权重 `.safetensors` 要由程序自己重建后重新写出。
- 索引文件也要和新的输出权重匹配，所以稍后单独写。

### 11.3 为什么 `LICENSE` 从 base 里兜底复制

第 152-156 行单独处理 `LICENSE`：

- 如果 base 里有 `LICENSE`
- 且输出目录里还没有
- 就把 base 的 `LICENSE` 复制过去

这说明作者把“输出目录尽量完整可分发”也考虑进来了。

## 12. 数学核心：`build_reconstructed_tensor()`

看 `alpharl/checkpoint_reconstruction.py:161-229`。这是整个项目最值得精读的函数。

### 12.1 这个函数的输入输出

输入：

- `base_tensor`
- `trained_tensor`
- `keep_ratio`
- `device`
- `svd_oversample`
- `svd_niter`

输出：

- `reconstructed_tensor`
  - 最终要写进输出 checkpoint 的 tensor。
- `stats`
  - 记录这个 tensor 的处理方式、保留 rank、相对误差和耗时。

### 12.2 为什么先记录 `shape`、`dtype` 和起始时间

第 171-175 行先收集：

- shape
- dtype

再配合 `start_time`，是为了最后输出 `reconstruction_stats.json`。这让程序除了“生成模型”之外，还会留下一个结构化审计文件。

### 12.3 为什么不是所有 tensor 都做 SVD

第 177-186 行有一个非常关键的分支：

```python
if trained_tensor.ndim != 2 or not torch.is_floating_point(trained_tensor):
    ...
    return trained_tensor.contiguous(), stats
```

意思是：

- 只有**二维浮点 tensor**才进入 SVD 近似流程。
- 其他 tensor 直接 passthrough，也就是原样保留 trained tensor。

这和大模型参数结构是吻合的：

- 线性层权重通常是 2D，适合做矩阵 SVD。
- bias、LayerNorm / RMSNorm 权重通常是 1D，不适合这里的矩阵分解路径。

所以这个项目的“近似”主要发生在 attention / MLP 的线性投影权重上，而不是所有参数一视同仁。

### 12.4 为什么先转成 `float32`

第 188-189 行：

```python
base_fp32 = base_tensor.to(device=device, dtype=torch.float32)
trained_fp32 = trained_tensor.to(device=device, dtype=torch.float32)
```

原因有两个：

1. SVD 在 `float32` 上通常比 `bfloat16` / `float16` 更稳定。
2. 即便原始 checkpoint 是低精度，分解和重建阶段也更适合用更高精度做数值运算。

最后再把结果转回训练后模型的原 dtype。

### 12.5 `delta = trained - base` 才是被近似的对象

第 190 行：

```python
delta = trained_fp32 - base_fp32
```

这个项目不是直接对训练后权重做低秩分解，而是对“训练增量”做低秩分解。这个设计和论文假设是对齐的，因为论文关心的是 RL 训练带来的更新子空间，而不是原始模型本身的完整表示。

### 12.6 `kept_rank` 怎么算

第 191-194 行：

```python
rows, cols = delta.shape
min_dim = min(rows, cols)
kept_rank = max(1, math.ceil(keep_ratio * min_dim))
q = min(min_dim, kept_rank + svd_oversample)
```

这里有三个细节：

1. rank 的上界是 `min(rows, cols)`，这是矩阵 rank 的基本事实。
2. 用 `ceil`，说明只要 `keep_ratio * min_dim` 有一点点超过整数边界，就向上取整。
3. `max(1, ...)` 保证至少保留 1 个奇异方向。

举个直观例子：

- 如果一个矩阵形状是 `5120 x 5120`
- `keep_ratio = 0.01`
- 那么 `kept_rank = ceil(0.01 * 5120) = 52`

所以“Top 1%”在这里不是保留 1% 的元素，而是保留 1% 的**奇异维度**。

### 12.7 `torch.svd_lowrank()` 返回的到底是什么

第 196-202 行是整个数学实现的核心：

```python
left, singular_values, right = torch.svd_lowrank(delta, q=q, niter=svd_niter)
left = left[:, :kept_rank]
singular_values = singular_values[:kept_rank]
right = right[:, :kept_rank]
delta_reconstructed = (left * singular_values) @ right.transpose(0, 1)
approximate = base_fp32 + delta_reconstructed
```

这里的含义是：

- `left` 相当于近似左奇异向量矩阵 `U`
- `singular_values` 相当于奇异值向量 `S`
- `right` 相当于右奇异向量矩阵 `V`

如果保留前 `r` 个奇异分量，那么重建公式就是：

```text
delta_reconstructed = U_r diag(S_r) V_r^T
```

代码里写成：

```python
(left * singular_values) @ right.transpose(0, 1)
```

其中 `left * singular_values` 利用了 broadcasting，等价于把 `U_r` 的每一列乘上对应的奇异值。

### 12.8 为什么用 `svd_lowrank` 而不是完整 `torch.linalg.svd`

因为这里面对的是 32B 模型的大矩阵：

- 完整 SVD 很昂贵。
- 这个任务本来就只关心前面一小部分主奇异子空间。

所以使用随机化低秩 SVD 更合理，速度和内存都会更友好。

### 12.9 误差是怎么统计的

第 204-206 行计算：

- `delta_norm = ||delta||_F`
- `residual_norm = ||delta - delta_reconstructed||_F`
- `relative_error = residual_norm / delta_norm`

这里的范数默认就是 Frobenius norm。

这意味着 `relative_frobenius_error` 反映的不是“最终模型输出误差”，而是“这个参数更新矩阵被低秩近似后的相对残差”。

### 12.10 为什么最后转回原 dtype 并搬回 CPU

第 208 行：

```python
reconstructed = approximate.to(dtype=trained_tensor.dtype, device="cpu").contiguous()
```

这么做是因为：

- 输出 checkpoint 的 dtype 应该和训练后模型保持一致。
- 写 safetensors 时，最终 tensor 需要在 CPU 侧组织。
- `.contiguous()` 可以避免写文件时遇到非连续内存布局的问题。

### 12.11 清理显存的逻辑在干什么

第 218-227 行显式 `del` 掉中间变量，最后如果跑在 CUDA 上再执行 `torch.cuda.empty_cache()`。

这不是功能正确性的必要条件，但对 32B 模型这种大规模张量处理很实用，因为：

- 中间变量很大。
- 一层层循环时，显存碎片和峰值占用可能变成实际瓶颈。

## 13. 输出辅助文件：索引和统计

看 `alpharl/checkpoint_reconstruction.py:232-268`。

### 13.1 `write_weight_index()`

第 232-236 行很简单：把 `trained_index` 原样写到输出目录。

这意味着输出 checkpoint 会继承 trained checkpoint 的 shard 布局。也就是说，程序不重新设计分片方式，而是沿用训练后模型已有的布局。

### 13.2 `write_reconstruction_stats()`

第 239-268 行会生成 `reconstruction_stats.json`，里面记录：

- 输入输出路径
- `keep_ratio`
- 请求的 device 和实际选中的 device
- `svd_oversample`
- `svd_niter`
- `seed`
- 拷贝了哪些支持文件
- 总 tensor 数
- 其中多少个 tensor 做了 SVD
- 多少个 tensor 走 passthrough
- 总耗时
- 每个 tensor 的详细统计

这个文件的价值很大，因为它让这段脚本不只是“跑完就结束”，而是留下一个可检查、可比较、可汇报的结果摘要。

## 14. 主流程：`reconstruct_checkpoint()`

看 `alpharl/checkpoint_reconstruction.py:271-329`。这是真正的 orchestrator。

### 14.1 入口三件套：依赖、参数、随机种子

第 272-274 行先做：

1. `require_dependencies()`
2. `validate_config(config)`
3. `seed_everything(config.seed)`

这是典型的“主流程前置防线”。

### 14.2 先对齐两个 checkpoint 的 tensor key

第 276-281 行：

1. 分别读取 base 和 trained 的索引。
2. 拿到各自的 `weight_map`。
3. 比较 key 集合是否完全相同。

如果不同，直接抛错：

```python
Base and trained checkpoints must contain the same tensor keys.
```

这是必要的，因为后续流程默认每个 `tensor_name` 都能在两边找到同名 tensor，并做一一对应的差分。

### 14.3 为什么按 trained shard 来组织输出

第 283-284 行先：

- `selected_device = select_device(config.device)`
- `grouped_trained_shards = group_tensors_by_shard(trained_weight_map)`

关键点在第二句：**输出分组是按 trained checkpoint 的 shard 布局来组织的**。

这带来两个结果：

1. 输出目录更像训练后模型的一个“近似替身”。
2. 只要配置和 tokenizer 也拷过去，Hugging Face 侧的加载路径更自然。

### 14.4 先建输出目录，再复制支持文件

第 286-291 行：

- 创建输出目录。
- 调 `copy_support_files(...)`。

这让配置文件和 tokenizer 文件提前就位，随后程序只需要专注写权重和统计。

### 14.5 双 reader 并行打开：一个读 base，一个读 trained

第 296-298 行：

```python
with SafeTensorReader(base...) as base_reader, SafeTensorReader(trained...) as trained_reader:
```

这表示：

- 同一个 tensor 名会分别从 base 和 trained 取出版本。
- 两个目录各自维护自己的 shard handle 缓存。

### 14.6 外层循环按 shard，内层循环按 tensor

第 299-319 行是嵌套循环：

- 外层：`for shard_name, tensor_names in grouped_trained_shards.items()`
- 内层：遍历这个 shard 中的每个 tensor

内层每次做 4 件事：

1. 读 base tensor。
2. 读 trained tensor。
3. 调 `build_reconstructed_tensor(...)`。
4. 把结果放进 `output_tensors`，同时记录 `tensor_stats`。

当一个 shard 里的 tensor 都处理完后，立即调用 `save_file(...)` 写出这个 shard。

这是一种很合理的内存策略：不会等所有 tensor 都处理完才统一写盘，而是按 shard 分批落盘。

### 14.7 流程结束后还要补两步

循环结束后，第 321-328 行再做：

1. `write_weight_index(config.output_path, trained_index)`
2. `write_reconstruction_stats(...)`

所以最终输出目录应该包含三类内容：

- 重建后的 `.safetensors` shard
- `model.safetensors.index.json`
- 支持文件和 `reconstruction_stats.json`

## 15. CLI 入口：`build_argument_parser()` 和 `main()`

看 `alpharl/checkpoint_reconstruction.py:332-369`。

### 15.1 `build_argument_parser()`

这里定义了所有命令行参数：

- `--base-model-path`
- `--trained-model-path`
- `--output-path`
- `--keep-ratio`
- `--device`
- `--svd-oversample`
- `--svd-niter`
- `--seed`

可以看出 CLI 很克制，没有做复杂模式切换，也没有多余开关。这个工具就是做一件事。

### 15.2 `main()`

`main()` 的逻辑也很标准：

1. 解析参数。
2. 构造 `ReconstructionConfig`。
3. 调 `reconstruct_checkpoint(config)`。
4. 返回 `0`。

这也是为什么 shell 脚本可以非常薄，因为 Python 侧已经把入口组织好了。

## 16. 真实模型目录告诉了我们什么

### 16.1 `dapo32b/full/config.json`

看 [`dapo32b/full/config.json`](../dapo32b/full/config.json)。

从这个文件可以读出当前实验对象的几个关键信息：

- 架构是 `Qwen2ForCausalLM`
- `model_type` 是 `qwen2`
- `num_hidden_layers` 是 `64`
- `hidden_size` 是 `5120`
- `intermediate_size` 是 `27648`
- `torch_dtype` 是 `bfloat16`

这说明这个项目不是玩具实验，而是对真实的大模型 checkpoint 在做后处理。

### 16.2 `dapo32b/full/model.safetensors.index.json`

看 [`dapo32b/full/model.safetensors.index.json`](../dapo32b/full/model.safetensors.index.json)。

这个索引文件里最重要的是 `weight_map`，它把形如：

- `model.layers.0.self_attn.q_proj.weight`
- `model.layers.0.mlp.up_proj.weight`
- `lm_head.weight`

这样的 tensor 名，映射到：

- `model-00001-of-00014.safetensors`
- `model-00002-of-00014.safetensors`
- ...
- `model-00014-of-00014.safetensors`

也就是说，完整模型被切成了 14 个输出 shard。程序之所以一定要先读索引、再按 shard 输出，正是因为实际 checkpoint 不是一个单文件。

### 16.3 为什么 `approx_top1` 目录值得单独说明

当前仓库里的 `dapo32b/approx_top1/` 已经有：

- `config.json`
- tokenizer 相关文件
- `README.md`
- `LICENSE`

但没有看到完整输出权重和索引。对理解项目来说，这反而能说明一件事：

- 这个目录确实是脚本的默认输出目标。
- 支持文件复制逻辑已经能从目录结构上看出来。
- 但完整重建结果是否已经在这份仓库中提交，还要以是否存在新 shard 和索引文件为准。

## 17. 测试文件其实是最小教学样例

看 [`tests/test_checkpoint_reconstruction.py`](../tests/test_checkpoint_reconstruction.py) 第 1-149 行。

这个测试很值得精读，因为它把大模型场景压缩成了一个可验证的 toy 例子。

### 17.1 测试准备了两个参数

- `linear.weight`
  - 一个 `4 x 4` 的二维矩阵。
- `norm.weight`
  - 一个长度为 4 的一维向量。

这两个参数正好对应主程序里的两条路径：

- `linear.weight` 会走 SVD 分支。
- `norm.weight` 会走 passthrough 分支。

### 17.2 为什么 `linear.weight` 会精确重建

测试里构造的 `delta` 实际上是 rank-1 结构，所以即使 `keep_ratio=0.25`，对应 `min_dim=4`，最终：

```python
kept_rank = ceil(0.25 * 4) = 1
```

只保留 1 个奇异方向也足够把这个 toy delta 完全重建出来。因此测试能断言：

- `reconstructed_weight == trained_weight`
- `relative_frobenius_error == 0.0`

这正好验证了“低秩主子空间就足够”的核心思路。

### 17.3 为什么故意把 base / trained 的 shard 布局写反

测试里很有意思的一点是：

- base 把 `linear.weight` 放在 shard 1，把 `norm.weight` 放在 shard 2。
- trained 刻意反过来放。

然后测试断言输出索引必须等于 trained 的布局。

这个设计是在验证：**程序按 tensor 名对齐内容，但按 trained checkpoint 的 shard 布局组织输出**。

### 17.4 测试验证了哪 4 类事情

这个测试主要验证了四件事：

1. 权重和 dtype 保持正确。
2. 支持文件会被复制到输出目录。
3. 输出索引布局继承 trained checkpoint。
4. `reconstruction_stats.json` 会记录正确的统计信息。

对于这样一个小仓库来说，这个测试已经覆盖了最关键的行为边界。

## 18. 一条完整执行路径

如果你从命令行实际运行，默认路径就是：

```bash
bash run_dapo32b_top1.sh
```

它等价于：

```bash
python -m alpharl.checkpoint_reconstruction \
  --base-model-path dapo32b/base \
  --trained-model-path dapo32b/full \
  --output-path dapo32b/approx_top1 \
  --keep-ratio 0.01 \
  --device cuda:0 \
  --svd-oversample 16 \
  --svd-niter 2 \
  --seed 0
```

这条命令背后的真实数据流是：

1. 从 `dapo32b/base` 和 `dapo32b/full` 读取同名 tensor。
2. 对二维浮点权重计算 `delta`。
3. 用随机低秩 SVD 近似 `delta` 的前 1% 奇异维度。
4. 重建近似权重并写回 `dapo32b/approx_top1`。
5. 同时复制模型配置和 tokenizer 文件。
6. 最后产出一个可加载的近似模型目录和一份统计报告。

## 19. 读完整个项目后，你应该记住什么

如果把整个仓库压缩成几句话，最重要的是下面这些结论：

- 它是 checkpoint 后处理工具，不是 RL 训练框架。
- 它只对二维浮点权重做低秩 SVD，其他参数直接沿用 trained 值。
- 它近似的是“训练更新矩阵 `trained - base`”，不是原始权重本身。
- 它沿用 trained checkpoint 的 shard 布局来写输出。
- 它不仅输出新权重，还会补齐配置、tokenizer、索引和重建统计。
- `tests/test_checkpoint_reconstruction.py` 是理解这个仓库最快的入口之一。

如果你下一步准备继续深入，最值得继续追的问题通常有三个：

1. `keep_ratio=0.01` 在真实 32B 模型上会保留每层多少 rank？
2. `reconstruction_stats.json` 里哪些层的相对误差最高？
3. 这种逐层低秩近似对真实下游任务表现损失有多大？

这些问题已经超出“读懂源码”的范围，但会自然衔接到后续实验分析。
