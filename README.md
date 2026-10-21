# HoloBrain · 仿脑全息小模型 + 外部记忆 = 自进化 Agent

> **不走 Transformer 的路。** 从零自研的超维计算（HDC）双通路全息脑：
> 模型 4.3MB · 提取 8 秒 · 决策 14ms · 进化 1.2ms · 零梯度零 epoch。
> 纯 NumPy，无框架、无预训练权重、无 API Key、全部离线。

[![license](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![deps](https://img.shields.io/badge/deps-numpy%20only-green.svg)](requirements.txt)

---

## 一分钟跑起来

```bash
git clone https://github.com/TaiMaBenJi/holo-brain.git
cd holo-brain
pip install -r requirements.txt
bash quick_start.sh        # 下数据 → 建知识库 → 捆绑全息脑 → 跑验收
```

已经打包好模型的话，跳过下载直接用：

```bash
python3 agent.py "记住我的手机是一加13"
python3 agent.py "我手机是什么型号"
python3 agent.py --task "统计沙箱里所有文本文件的行数"   # 自己写代码→跑→沉淀技能
python3 agent.py --evolve 5     # 自我进化：自己出题自己做
python3 agent.py --learn        # 睡眠重放：巩固经历
python3 agent.py --skills       # 看技能库
```

---

## 一、为什么要重造架构

主流小模型路线是"把 Transformer 缩小 + 梯度训练"。这条路的三个硬约束：
**要梯度、要算力、要语料**，且训练完模型就是死的——新经验加不进去。

本项目走**超维计算（Hyperdimensional Computing, HDC）**：

| | Transformer 小模型 | HoloBrain |
|---|---|---|
| 知识载体 | 亿级权重 + 梯度 | 4096 维全息码 + 整数记忆账本 |
| 学习方式 | 数小时反向传播 | 捆绑（bundling），**8 秒** |
| 加新知识 | 重训 / 微调 | **写一条向量，1.2ms** |
| 忘掉知识 | 做不到 | **减一条向量，随时撤销** |
| 可解释 | 黑盒 | 每一维可追溯 |
| 灾难性遗忘 | 有 | 无 |

HDC 不是新造的轮子：Kanerva(1988) 的超维计算、Plate(1995) 的全息归约表示，
是**逆向人脑皮层电路**得到的计算模型（分布式稀疏表示 + 赫布学习 + 内容寻址）。
本项目是把它**落到一个能真跑、能进化、能接工具的 Agent 上**。

## 二、仿脑映射

| 脑结构 | 生理原理 | 本项目实现 | 实测 |
|---|---|---|---|
| 神经元编码 | 稀疏分布式表示 | 4096 维 {−1,+1} 全息码 | 码本 1.8MB |
| 突触可塑性 | 赫布规则（一起激活就连接） | 捆绑 = 逐维多数表决 | 提取 8s |
| **海马体** | **一次学会**（情景记忆） | `learn()` 毫秒写入 | **1.2ms/条** |
| 突触消弱 | 用进废退 | `forget()` 负写入 + `decay()` | 睡眠期 0.96 衰减 |
| 杏仁核 | 情绪加权（恐惧记最牢） | 失败经验权重 ×2 | 自评 **100%** |
| 睡眠重放 | 做梦时重放巩固 | `consolidate()` 重放 + 清洗 | `--learn` |
| 联想皮层 | 相关性直觉 | 成对绑定码 rel:yes / rel:no | 相关性 **90%** |
| 前额叶 | 决策 | 意图原型 + 情景 kNN | 路由 **90.7%** |
| 内容寻址记忆 | 一缕气味唤醒整段回忆 | `sim()` 全息寻址 | 技能命中 **1.00** |
| 基底核 | 程序性记忆（技能） | 技能库（SQLite） | 复用零重写 |
| 新皮层 | 慢速语义知识 | 23 万条知识库（FTS5） | 检索 <10ms |

**双通路决策**：腹侧流（IDF 加权语义通路）+ 背侧流（全息结构通路），
流间整合出最终意图——大脑本来就这么干。两条流各自都会失灵，合起来才稳。

## 三、从 DeepSeek 抠出来重写的六件事

依据官方模型卡（DeepSeek-R1 / DeepSeek-V3）的**公开技术描述**，
按思想重新实现（未使用其任何代码或权重）：

| DeepSeek 机制 | 原始作用 | 在本项目的重写 |
|---|---|---|
| **MoE 稀疏激活**（671B 总 / 37B 激活） | 专家多、按需激活 | 情景库存 5454 条经验，**每次决策只激活 top-k 邻近记忆** |
| **MLA 低秩潜空间** | 把 KV cache 压成 latent | **记忆潜空间压缩**：随机投影 4096→192 维后再检索（JL 引理保内积） |
| **MTP 多 token 预测** | 一次预测多步 | `mtp()` 同时给出主意图 + 次优兜底，执行失败自动换路 |
| **R1 冷启动 + 反思** | 少量引导数据 → 反思强化 | 冷启动种子库 + 杏仁核失败加权 + 睡眠重放巩固 |
| **R1 蒸馏**（"Smaller Models Can Be Powerful Too"） | 大模型思维模式进小模型 | 开源语料**重写进记忆账本**，8 秒完成，零梯度 |
| **self-verification** | 自我验证 | 杏仁核评估头，100% 准确 |

## 四、数据从哪来（全部开源，不在仓库里分发）

```bash
bash scripts/fetch_data.sh      # 自动下载（国内镜像优先）
```

| 源 | 量 | 用途 |
|---|---|---|
| [YeungNLP/firefly-train-1.1M](https://huggingface.co/datasets/YeungNLP/firefly-train-1.1M) | 采样 20.4 万条 | 知识库（问→答） |
| [silk-road/alpaca-data-gpt4-chinese](https://huggingface.co/datasets/silk-road/alpaca-data-gpt4-chinese) | 5.2 万条全量 | 知识库（GPT4 质量答案） |
| [clinc/oos-eval](https://github.com/clinc/oos-eval) | 1.8 万条 / 151 类 | 意图样本（真实用户指令） |
| 中文意图种子（手写 14 类 × 30+ 模板全枚举） | 5454 条 | 意图原型 |

> **下载源经验（国内）**：GitHub release 资产只有 ~7KB/s，是死路；
> 源码走 `codeload.github.com`（2.8MB/s），权重走 `hf-mirror.com`（2.5MB/s）。

## 五、实测成绩

设备：**一加 13（骁龙 8 Elite，ARMv9.2，24GB）**，Alpine PRoot 沙箱，纯 CPU。

| 项目 | 成绩 |
|---|---|
| 意图路由（合成测试集 289 条） | **90.7%** |
| 意图路由（**手写真实话术留出集 40 条**） | **77.5%** |
| 记忆相关性 | **90.0%** |
| 自我评估（成败判断） | **100%** |
| 决策延迟 | **14.1 ms/次** |
| 海马体写入（一次学会） | **1.2 ms/条** |
| 从开源数据提取捆绑 | **8.4 秒**（对比梯度训练 ~60 分钟） |
| 模型体积 | **4.3 MB**（码本 1.8MB + 账本 2.5MB） |
| 技能复用命中置信 | **1.00**（二次调用零重写） |

端到端实测：让 Agent "统计所有文件行数" → **自己写代码 → 跑出真实结果 → 沉淀为技能** →
同一任务再来 → **命中技能直接复用，不再重写代码**。

## 六、诚实的边界

- 4.3MB 的全息脑**不会自由对话**。它的职责是：路由、召回、判断成败、寻址技能。
  语言生成靠 23 万条问→答检索 + 技能模板。
- 真实话术留出集 77.5%，错例集中在中文天然歧义（"记一下"是记还是查、
  "读取配置"是读文件还是抓网页）——补同义种子可继续改善。
- HDC 本质是**高维近似检索**，不是语义理解：同义改写仍依赖词表/种子覆盖。
  换来的是每条经验可读、可删、可追溯，且进化不会灾难性遗忘。

## 七、代码结构

```
holo.py        全息脑本体（脑区实现）           378 行，纯 NumPy
distill.py     开源数据 → 捆绑进记忆账本         138 行，8 秒
agent.py       Agent：路由/记忆/技能/反思/进化   700 行
understand.py  新皮层：知识库检索（SQLite FTS5） 185 行
ingest_data.py 数据重写：开源语料 → knowledge.db 218 行
make_data.py   语料合成（意图/相关性/评估）      278 行
test_holo.py   验收：脑机制逐项 + 三大能力       143 行
holdout.py     手写真实话术留出集（40 条）
scripts/fetch_data.sh   下载开源语料（镜像优先）
quick_start.sh          一键构建 + 验收
```

## 八、引用

如果你用了这个项目，请引用本仓库；如果从中借鉴了想法，也请引用 HDC 原始文献：

```bibtex
@misc{holobrain2026,
  title  = {HoloBrain: A Brain-Inspired Holographic Small Model for Self-Evolving Local Agents},
  author = {TaiMaBenJi},
  year   = {2026},
  url    = {https://github.com/TaiMaBenJi/holo-brain}
}
```
