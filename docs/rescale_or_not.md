对，当前代码**没有**实现图片里 Eq.(3) 那种额外放缩。

在 [`alpharl/checkpoint_reconstruction.py`](../alpharl/checkpoint_reconstruction.py#L196) 里，代码做的是标准截断 SVD 重建：

```python
left, singular_values, right = torch.svd_lowrank(delta, q=q, niter=svd_niter)
left = left[:, :kept_rank]
singular_values = singular_values[:kept_rank]
right = right[:, :kept_rank]
delta_reconstructed = (left * singular_values) @ right.transpose(0, 1)
```

这对应的是：

\[
\hat{\Delta W}_k=\sum_{i=1}^k \sigma_i u_i v_i^\top
\]

代码里**没有再做**：

\[
\alpha=\frac{\|\Delta W\|}{\|\hat{\Delta W}_k\|},\qquad
\tilde{\Delta W}_k=\alpha \hat{\Delta W}_k
\]

**这个放缩有没有必要，取决于目标。**

1. 如果目标是“尽量逼近原始 full checkpoint”
  
   不太有必要，甚至默认不该加。  
   原因是截断 SVD 本身就是给定 rank 下最优的 Frobenius 近似；再乘一个 `alpha` 会把保留下来的方向整体放大，通常会让参数误差变大，而不是变小。

2. 如果目标是“复现论文里对子空间贡献的分析口径”
  
   有必要，或者至少应该做成可选开关。  
   因为论文这里想控制“更新强度”一致，只比较“方向/子空间”本身是否有效。  
   不放缩时，rank-1 或 top-k% 更新的范数天然更小，实验结果会混进“方向少了”和“步长变小了”两种因素。

直白一点说：

- **不放缩**：更像“忠实重建原更新”
- **放缩**：更像“固定 update strength，测试这个子空间本身有没有用”

还有一个细节：论文式子写的是 `||·||_2`。如果严格按**矩阵谱范数**理解，那么保留 top-1 奇异值时两边范数其实都等于 `σ1`，`alpha` 基本就是 1，放缩没意义。  
所以这里更合理的理解通常是“把矩阵展平后的 L2 norm”，也就是 **Frobenius norm** 的口径。

结论是：

- 你现在这份代码更偏“工程上的近似 checkpoint 重建”，所以**不放缩是合理的**。
- 如果你想更贴近论文 2.1 节那种“子空间贡献评估”，最好加一个可选的 `rescale` 开关。

---

```python
used_rank = s.shape[0] if rank is None or rank <= 0 else min(rank, s.shape[0])
    u_k = u[:, :used_rank]
    s_k = s[:used_rank]
    vt_k = vt[:used_rank, :]

    update_k = u_k @ torch.diag(s_k) @ vt_k
    topk_norm = torch.linalg.matrix_norm(update_k, ord="fro")
    scale = 1.0
    if topk_norm.item() > 0:
        scale = (fro_norm / topk_norm).item()

    param.data.add_(alpha * scale * update_k.to(dtype=param.data.dtype))
```
这是**有放缩**的实现。

它做了两步：

1. 先做截断 SVD，得到 top-k 低秩更新
```python
update_k = u_k @ torch.diag(s_k) @ vt_k
```

2. 再把这个低秩更新按范数放大/缩小
```python
topk_norm = torch.linalg.matrix_norm(update_k, ord="fro")
scale = fro_norm / topk_norm
```

最后真正加回去的是：

```python
alpha * scale * update_k
```

所以它不是“纯截断 SVD 重建”，而是：

\[
\tilde{\Delta W}_k = \alpha \cdot \frac{\|\Delta W\|_F}{\|\Delta W_k\|_F}\cdot \Delta W_k
\]

前提是你的 `fro_norm` 确实是原始完整更新 `ΔW` 的 Frobenius norm。

**这属于哪一种？**
- 如果按我们前面讨论的分类，它属于“**放缩版**”
- 更准确地说，是“**用 Frobenius norm 做 norm matching 的放缩版**”

**和论文图里的关系**
- 思路上是一样的：保留 top-k 子空间后，再匹配原更新强度
- 但你这段代码明确用的是 `ord="fro"`，也就是 **Frobenius norm**
- 论文截图写成 `||·||_2`，严格数学上更像谱范数；不过很多实现里实际上更常用 Frobenius norm，因为更符合“整体更新能量匹配”的直觉

补一句：
- 如果 `used_rank` 已经等于完整 rank，那么 `update_k` 基本就是原始更新，`scale` 会接近 1
- 真正有意义的是截断后 `used_rank < full_rank` 的情况，这时 `scale` 往往会大于 1

---

核心差别其实就一句话：

- 仓库当前实现是“**纯截断 SVD 重建**”
- 你贴的实现是“**截断 SVD + Frobenius norm 放缩**”

对照当前代码在 [`alpharl/checkpoint_reconstruction.py:188`](../alpharl/checkpoint_reconstruction.py#L188) 到 [`alpharl/checkpoint_reconstruction.py:208`](../alpharl/checkpoint_reconstruction.py#L208) 这一段看，会更清楚。

**1. 数学形式不一样**

当前仓库做的是：

\[
\Delta W_k = \sum_{i=1}^k \sigma_i u_i v_i^\top
\]

对应代码：

```python
left, singular_values, right = torch.svd_lowrank(delta, q=q, niter=svd_niter)
left = left[:, :kept_rank]
singular_values = singular_values[:kept_rank]
right = right[:, :kept_rank]
delta_reconstructed = (left * singular_values) @ right.transpose(0, 1)
approximate = base_fp32 + delta_reconstructed
```

你贴的实现做的是：

\[
\tilde{\Delta W}_k
=
\frac{\|\Delta W\|_F}{\|\Delta W_k\|_F}\Delta W_k
\]

再把它加回参数：

```python
scale = fro_norm / topk_norm
param.data.add_(alpha * scale * update_k)
```

所以它不是“最优低秩近似本身”，而是“保留 top-k 子空间后，再把整体强度拉回原更新范数”。

**2. 优化目标不一样**

当前仓库的目标更像：

- 让重建后的权重尽量接近 `trained`
- 也就是尽量减小 `||ΔW - ΔW_k||`

这是标准低秩近似目标。

你贴的实现目标更像：

- 保留 top-k 子空间方向
- 但不希望因为截断后范数变小，导致更新“力度”也一起变小
- 所以把 `ΔW_k` 再放大到和原 `ΔW` 差不多强

这更像论文里的“**子空间贡献分析**”口径，而不是最忠实的 checkpoint 重建口径。

**3. 最终行为差别**

如果同一个层的完整更新是 `ΔW`：

- 当前仓库输出的是 `base + ΔW_k`
- 你的实现输出的是 `base + scale * ΔW_k`

这会带来三个实际差别。

第一，当前仓库的结果通常**更接近原始 full checkpoint**。  
因为截断 SVD 本来就是给定 rank 下的最优 Frobenius 近似；再乘一个 `scale`，通常会偏离这个最优点。

第二，你的实现通常**更接近“保留原更新强度”**。  
尤其当 `k` 很小、丢掉了很多奇异值时，`||ΔW_k||_F` 会明显小于 `||ΔW||_F`，这时 `scale > 1`，更新会被整体放大。

第三，你的实现会**改变保留奇异值本身的数值**。  
截断 SVD 保留的是原来的前 `k` 个奇异值；放缩后，相当于把这些奇异值统一乘了一个常数。  
所以它保留了“子空间方向”，但不再保留“原始 top-k 奇异值大小”。

**4. 什么时候两者差不多，什么时候差很多**

如果原更新本来就几乎 rank-1 主导，比如奇异值像：

\[
[10,\ 0.1,\ 0.05,\dots]
\]

那么：

- `||ΔW||_F` 和 `||ΔW_1||_F` 很接近
- `scale` 会非常接近 1
- 放缩和不放缩几乎没差

但如果更新能量分布更分散，比如：

\[
[10,\ 10,\ 10]
\]

只保留 rank-1 时：

- `||ΔW_1||_F = 10`
- `||ΔW||_F = \sqrt{10^2+10^2+10^2} \approx 17.32`
- `scale ≈ 1.732`

这时放缩会非常明显，相当于把保留的第一主方向强行放大了 73.2%。

**5. 你仓库当前这版和你贴的实现，还有一个工程差别**

当前仓库用的是 `torch.svd_lowrank(...)`，见 [`alpharl/checkpoint_reconstruction.py:196`](../alpharl/checkpoint_reconstruction.py#L196)。  
这意味着它是“**近似的低秩 SVD**”，适合大模型矩阵。

你贴的代码里 `u, s, vt` 已经是现成的，看起来更像前面先做了完整 SVD 或别的分解。  
所以除了“放不放缩”之外，两者还可能有一层差别：

- 当前仓库：更省内存、更适合 32B 权重
- 你那段：如果前面是 full SVD，会更精确，但更贵

**6. 怎么理解哪种更适合你的项目**

如果你的目标是：

- “生成一个尽量接近 `full` 的近似 checkpoint”

那当前仓库这种**不放缩**更自然。

如果你的目标是：

- “验证 top-k 子空间本身是否解释了 RL 增益”
- “控制 update strength，不让 rank 变小自动带来步长变小”

那你贴的这种**放缩版**更贴近论文分析设定。

所以它们不是谁绝对更对，而是回答的问题不同：

- 当前仓库回答：`top-k` 低秩近似后，能多像原 checkpoint？
- 你的实现回答：如果只保留这个子空间，但把更新强度补回来，它还能不能起作用？

