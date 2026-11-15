# 打卡指南 · Issue #284

> 模型架构与训练 | 202611 Task0 · 解码器主干与训练起点
> Issue: https://github.com/datawhalechina/llm-algo-leetcode/issues/284
> 打卡仓库（fork）: https://github.com/iOPEN-Xing/llm-algo-leetcode

---

## 一、整体流程（两步走）

```
第 1 步：在 GitHub issue #284 下上传「测试通过截图」        ← 这是正式打卡凭据
   ↓
第 2 步：把「issue 截图」发到微信群                        ← 截图要含三要素
```

### 为什么是两步

issue 里的截图证明「代码真的跑通了」；微信群截图证明「是你本人打的卡」。
所以第 2 步的截图必须**包含第 1 步的截图**，不能只发本地终端的输出。

### 三要素（微信群截图里必须清晰可见）

| # | 要素 | 怎么保证 |
|---|---|---|
| 1 | **issue 标题** | 截图时把 issue 页面**标题栏**和**评论区**一起框进去 |
| 2 | **GitHub ID** | 评论者头像/名字那一行可见（`iOPEN-Xing`） |
| 3 | **微信群昵称** | 群成员列表里自己的昵称可见，或在截图上补一行文字 |

> 小技巧：把 issue 页面缩放到 80%，用 macOS `⇧⌘4` 截「窗口」而不是「区域」，
> 一次就能把标题栏 + 评论框 + 侧栏头像全部带上。

---

## 二、打卡正文模板（贴在 issue 评论框里）

```markdown
微信群昵称：<你的群昵称>
环境：Mac（Apple Silicon arm64）
系统：macOS 26.6.2
AI 占比 vs 人工占比：70：30
任务项编号：4.1 最小打卡 · LLaMA3 Block 组装与残差顺序

## (1) 因果掩码的作用

因果掩码限制每个位置只能读取**自身与此前位置**的信息，
即位置 t 的注意力只能覆盖 j ≤ t。

训练时序列是并行送入的。如果不加因果掩码，位置 t 的表示就能"看到"
未来位置 t+1...T 的 token id，而这些位置恰恰是它要预测的答案。
结果是：训练时模型能从输入里直接"读出"标签，loss 降得很漂亮，
但推理时没有未来 token 可读，分布完全对不上 —— 训练/推理不一致 (train-test leakage)。

实际后果：
- 训练 loss 异常低（远低于正常 next-token 预测应有的水平）
- 生成的文本在真实推理场景下崩坏、重复或胡言乱语
- 严重时模型学到"复制输入"的捷径，而非语言结构

## (2) LLaMA3 Block 的组件顺序

主状态 h 依次经过（两次 Pre-Norm + 两次残差）：

1. Pre-Norm₁:  h'₁ = Norm₁(h)
2. 注意力更新:  h'₂ = h'₁ + Attention(h'₁)
3. Pre-Norm₂:  h'₃ = Norm₂(h'₂)
4. MLP 更新:    h_out = h'₃ + MLP(h'₃)

写成公式：
  h'     = h + Attention(Norm₁(h))
  h_next = h' + MLP(Norm₂(h'))

两个关键点：
- 第二次 Pre-Norm 读到的必须是**已经包含注意力更新**的状态 h'，不是原始 h
- 注意力跨词元混合信息，MLP 只在单个词元内重组通道；两者不能合并成一次无序求和

## (3) 补全代码 + 运行截图

- notebook: 02_PyTorch_Algorithms/05_LLaMA3_Block_Tutorial.ipynb
- 补全 TODO 1（Attention 残差更新）、TODO 2（MLP 残差更新）、TODO 3（多层顺序堆叠）
- 测试结果：✅ Decoder Block：残差顺序、梯度连通与层堆叠测试通过
- 截图见下（附在评论中）

当前卡点：无
```

### 增项 4.2 / 4.3 的补充段落

**4.2 验证 loss 比训练 loss 还小**

最可能的原因：**验证集和训练集共用同一份 batch 契约（同一 batch 来源、
同一套 shuffle / drop_last / 归一化逻辑），导致验证集没有真正独立**——
比如验证时不小心复用了训练时的 `DataLoader`（忘了 `shuffle=False`，
或者 iter 了训练集），或者验证集数据本身被训练集污染/重复采样。

定位检查点（按这个顺序加，成本递增）：

| 位置 | 加什么检查 | 能区分什么 |
|---|---|---|
| ① 数据构造处 | 打印 train/val 的**样本 id 前若干个**，或对输入做 hash | 若两者重叠 → 数据泄漏，立刻确诊 |
| ② DataLoader | 断言 `train_loader.dataset is not val_loader.dataset`，且 val 的 `shuffle=False` | 排除"忘了改 shuffle" |
| ③ 采样器 | 确认 val 用的是 `SequentialSampler` 而非 `RandomSampler` | 排除验证集被打乱 |
| ④ 训练循环 | 每步分别记录 train_loss 和 eval_loss 到**两条独立曲线**，并记录 epoch | 排除"打印变量写错、其实记的是 train loss" |
| ⑤ 最直接 | 用**固定 held-out 文件**跑一次 eval，绕过所有 DataLoader 逻辑 | 若此时正常 → 问题一定在数据管线，不在模型 |

一句话判据：**若 eval loss 稳定地低于 train loss 且差距随训练扩大，
几乎必然是验证数据泄漏或验证时评估了训练集**，而不是"模型过拟合"——
过拟合一定是 train loss 低于 eval loss，方向相反。

**4.3 三组 Block 变体的单变量对比**

固定输入、深度、宽度、种子、训练预算，只改一个字段：

| 变体 | Norm 位置 | Norm 类型 | FFN 激活 | 梯度传播 | 激活尺度 |
|---|---|---|---|---|---|
| A | Pre | RMSNorm | ReLU | 残差主干上**有一条恒等路径**直通，梯度不经过 Norm | RMSNorm 不断言均值，重中心化前 RMS 已受控；ReLU 负半轴梯度为 0，有"死神经元"风险，尺度分布更尖 |
| B | Pre | LayerNorm | SwiGLU | 同 A，恒等路径完整 | LayerNorm **额外重中心化**，把均值也压到 0，均值漂移被抑制得更彻底；SwiGLU 门控可正可负，门可学会"关掉"某通道，尺度分布更宽但可控 |
| C | **Post** | RMSNorm | SwiGLU | 残差相加**之后**才归一化，梯度每层都要穿过 Norm 的雅可比，**恒等路径被切断** | 归一化后的输出尺度被强行拉回，但残差累积的漂移直接进入下一层输入，深层容易尺度爆炸/消失 |

结论方向：
- **Norm 位置**对梯度的影响最大 —— Pre-Norm 保留恒等路径，是深层可训练的前提；
  Post-Norm 归一化点后移，梯度被 Norm 反复缩放，深层容易不稳
- **Norm 类型**：RMSNorm 省掉减均值一步（省算力），LayerNorm 多一个重中心化，
  对"均值漂移"更鲁棒，代价是稍多计算
- **FFN 激活**：ReLU 单调无界、无门控；SwiGLU 有符号门控，可以学会有选择地
  放大/抑制通道，配合 Pre-Norm 给深层更稳的激活尺度

---

## 三、任务清单与当前进度

打卡粒度可选三档，**优先保住最小打卡**：

> (1) 仅 4.1　(2) 4.1 + 4.2　(3) 4.1 + 4.2 + 4.3

| 任务项 | 交付物 | 对应 notebook | 状态 |
|---|---|---|---|
| **4.1 (1)** | 文字回答：因果掩码作用 + 不加会怎样 | 阅读 `topic_discussion/llm_architecture_evolution/01_transformer_decoder.md` | ⬜ 待写 |
| **4.1 (2)** | 文字回答：LLaMA3 Block 完整组件顺序 | 同上 + `05_LLaMA3_Block_Tutorial.ipynb` Step 2 | ⬜ 待写 |
| **4.1 (3)** | **补全代码 + 运行截图** | `02_PyTorch_Algorithms/05_LLaMA3_Block_Tutorial.ipynb` | ✅ **已完成** |
| **4.2** | 文字回答：验证 loss 偏小的原因 + 定位检查 | `00_Prerequisites/12_PyTorch_Minimal_Training_Interface.ipynb` | ⬜ 待写 |
| **4.3 (1)** | 文字回答：A/B/C 三变体对比 | `02_PyTorch_Algorithms/60_Decoder_Block_Stability_Benchmark.ipynb` | ⬜ 待写 |
| **4.3 (2)** | **补全代码 + 运行截图** | `02_PyTorch_Algorithms/60_Decoder_Block_Stability_Benchmark.ipynb` | ⬜ **未做** |

### 已完成部分的验证记录

`05_LLaMA3_Block_Tutorial.ipynb`（已 commit `c07bed7`，已 push 到 `feature/task0-decoder-block`）：

```
[cell 9] ✅ Decoder Block：残差顺序、梯度连通与层堆叠测试通过
[cell 15] ✅ Loss 下降: 3.182898 → 2.430521
          ✅ 最小数据逻辑与训练循环通过
```

### 4.3 剩余工作（若要冲第 3 档）

`60_Decoder_Block_Stability_Benchmark.ipynb` 有 4 个 TODO，全部集中在 cell 7：

| TODO | 函数 | 要做什么 |
|---|---|---|
| 1 | `validate_block_spec` | 校验 `norm_position ∈ {pre_norm, post_norm}`、`norm_kind ∈ {rmsnorm, layernorm}`、`activation ∈ {swiglu, gelu}`；`d_ff` 必须是 `d_model` 的整数倍（SwiGLU 要等宽），否则 `ready=False` |
| 2 | `validate_single_block_change` | 比较 baseline/candidate 的 `norm_position`/`norm_kind`/`activation`，**恰好一个字段不同**且等于 `declared_field` 才 `ready=True`；返回 `changed_fields` 列表 |
| 3 | `run_block_stability_probe` | 堆 `depth` 层 `_ToyDecoderBlock`，用固定 seed 造输入，前向+反传，记录 `input_grad_norm`、`activation_std`、`finite`、`param_count` |
| 4 | `decide_block_candidate` | `finite=False` → `reject`；仅 CPU 探针证据 → `tune`；匹配 checkpoint 的真实证据 → `accept` |

测试断言（cell 8）已写好，跑通即输出：
`测试通过：Block 契约、单变量比较、稳定性探针与决策口径一致。`

> ⚠️ 注意 `_ToyDecoderBlock` 里 `activation` 只实现了 `swiglu` 和 `gelu` 两条分支。
> issue 说的 "ReLU FFN" 对应代码里的 `gelu` 分支（`_ffn` 的 else 分支用 `F.gelu`）。
> 写对比分析时要说清这个映射，别直接写 ReLU 代码对不上。


---

## 四、环境与执行

本机环境已就绪（仓库根目录）：

```bash
cd /Users/xing/Desktop/llm-algo-leetcode-main
source .venv/bin/activate
```

Python 3.12.16 + torch 2.15.0 (CPU)。

**注意 PATH 顺序**：`~/.local/bin` 里有一个损坏的 `jupyter-kernelspec`，
用 jupyter 命令时需要让 venv 优先：

```bash
PATH="$PWD/.venv/bin:$PATH" python -m jupyter nbconvert \
  --to notebook --execute --inplace \
  02_PyTorch_Algorithms/05_LLaMA3_Block_Tutorial.ipynb
```

Jupyter 内核已注册为 `llm-algo`（显示名 `Python 3.12 (llm-algo)`）。

---

## 五、git remote 配置

```bash
origin    git@github.com:iOPEN-Xing/llm-algo-leetcode.git        # 你的 fork
upstream  git@github.com:datawhalechina/llm-algo-leetcode.git   # 官方
```

提交你的 notebook 改动：

```bash
git checkout -b feature/task0-decoder-block
git add 02_PyTorch_Algorithms/05_LLaMA3_Block_Tutorial.ipynb
git commit -m "feat: complete LLaMA3 Decoder Block assembly (TODO 1-3)"
git push -u origin feature/task0-decoder-block
```

---

## 六、checklist

### 最小打卡（4.1）—— 最小可交付

- [ ] `05_LLaMA3_Block_Tutorial.ipynb` TODO 1–3 已补全
- [ ] 测试输出 `✅ Decoder Block：残差顺序、梯度连通与层堆叠测试通过`
- [ ] notebook 已保存到磁盘（Cmd+S）
- [ ] 改动已 commit + push 到 fork
- [ ] 截图已上传到 issue #284（作为附件）
- [ ] issue 评论正文已填（按第二节模板，含 4.1(1)(2)(3) 三问）
- [ ] 微信群截图含三要素：issue 标题 / GitHub ID / 微信群昵称
- [ ] 微信群截图已发出

### 增项（4.2 / 4.3）

- [ ] 4.2 答案写进评论正文
- [ ] 4.3(1) 三变体对比表写进评论正文
- [ ] `60_Decoder_Block_Stability_Benchmark.ipynb` TODO 1–4 已补全
- [ ] 测试输出 `测试通过：Block 契约、单变量比较、稳定性探针与决策口径一致。`
- [ ] 4.3 截图已上传
