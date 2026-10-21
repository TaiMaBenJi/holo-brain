# HoloBrain · 仿脑全息小模型 + 外部记忆 = 自进化 Agent

[![CI](https://github.com/TaiMaBenJi/holo-brain/actions/workflows/ci.yml/badge.svg)](https://github.com/TaiMaBenJi/holo-brain/actions/workflows/ci.yml)

**不走 Transformer 的路。** 从零造的架构：超维计算（HDC）双通路全息脑，
配 SQLite 外部记忆，模型 4.1MB、提取 12 秒、决策 5 毫秒、进化只需 1 毫秒。

```
holo.py        全息脑本体（脑区实现）      约 380 行，纯 NumPy
distill.py     从开源数据提取 → 捆绑      12 秒，零梯度
agent.py       Agent：路由/记忆/技能/反思/进化
understand.py  新皮层：23万条知识库检索
ingest_data.py 开源语料 → 本地库（类型感知种子体系）
holdout.py     诚实留出集（40 条 + 25 条独立盲测）
test_holo.py   验收（含脑机制逐项验证）
```

## 一、它怎么用

```bash
python3 agent.py "记住我的手机是一加13"      # 写入海马体
python3 agent.py "我手机是什么型号"          # 联想皮层召回
python3 agent.py --task "统计所有文件的行数"  # 自己写代码→跑→沉淀技能
python3 agent.py --evolve 5                  # 自我进化：自己出题自己做
python3 agent.py --learn                     # 睡眠重放：巩固经历
python3 agent.py --skills                    # 看技能库
```

## 二、仿脑映射（每个组件对应一个脑结构）

| 脑结构 | 生理原理 | 本系统实现 | 实测 |
|---|---|---|---|
| 神经元编码 | 稀疏分布式表示 | 4096 维 {−1,+1} 全息码 | 码本 1.9MB |
| 突触可塑性 | 赫布规则 | 捆绑 = 逐维多数表决 | 提取 12 秒 |
| **海马体** | **一次学会**（情景记忆） | `learn()` 毫秒写入 | **~1 ms/条** |
| 突触消弱 | 用进废退 | `forget()` 负写入 + `decay()` | 衰减 0.96/睡眠 |
| 杏仁核 | 情绪加权（恐惧记最牢） | 失败经验权重 ×2 | 自评 **100%** |
| 睡眠重放 | 做梦巩固记忆 | `consolidate()` 重放+清洗 | `--learn` |
| 联想皮层 | 相关性直觉 | 成对绑定码 rel:yes/no | 相关性 **89.7%** |
| 前额叶 | 决策 | 双通路融合 + 写入动词闸 | 路由 **95.0%** |
| 内容寻址 | 一缕气味唤醒整段回忆 | `sim()` 全息寻址 | 技能命中 1.00 |
| 基底核 | 程序性记忆 | 技能库（skills 表） | 复用成功率 100% |
| 新皮层 | 慢速语义知识 | 23 万条知识库（FTS5） | 检索 <10ms |

## 三、从 DeepSeek 抠出来重写了什么

依据官方模型卡（DeepSeek-R1 / DeepSeek-V3）：

| DeepSeek 机制 | 原始作用 | 在本项目的重写 |
|---|---|---|
| **MoE 稀疏激活**（671B 总 / 37B 激活） | 专家多、按需激活 | 情景库存约 5000 条经验，**每次决策只激活 top-k 邻近记忆** |
| **MLA 低秩潜空间** | KV cache 压成 latent | **记忆潜空间压缩**：随机投影 4096→192 维后再检索（JL 引理保内积） |
| **MTP 多 token 预测** | 一次预测多步 | **次级意图预测** `mtp()`：主意图 + 次优兜底，执行失败换路 |
| **R1 冷启动 + 反思** | 引导数据 → 反思强化 | 冷启动种子库 + 杏仁核失败加权 + 睡眠重放巩固 |
| **R1 蒸馏**（"Smaller Models Can Be Powerful Too"） | 大模型思维模式进小模型 | 把开源语料（firefly 20.4 万 / alpaca 5.2 万 / clinc 1.8 万）**重写进记忆账本**，12 秒完成，零梯度 |
| **self-verification** | 自我验证 | 杏仁核评估头 100% 准确 |

## 四、数据从哪来（全部开源，已重写为本地库）

| 源 | 量 | 用途 |
|---|---|---|
| [YeungNLP/firefly-train-1.1M](https://huggingface.co/datasets/YeungNLP/firefly-train-1.1M) | 采样 20.4 万条 | 知识库（问→答） |
| [silk-road/alpaca-data-gpt4-chinese](https://huggingface.co/datasets/silk-road/alpaca-data-gpt4-chinese) | 5.2 万条全量 | 知识库（GPT4 质量） |
| [clinc/oos-eval](https://github.com/clinc/oos-eval) | 1.8 万条 / 151 类 | 意图样本（CLINC 真实用户指令） |
| 中文意图种子（具名槽位 · 类型感知填充 · 同义模板全枚举） | 约 4000 条 | 意图原型 |

种子体系采用**具名槽位**（`{path}`/`{url}`/`{cmd}`…）与**类型感知填充池**：
URL 槽绝不填文件名、命令槽绝不填网页——从源头杜绝「保存到 example.com」
这类跨意图污染样本（v1.0 留出集 8 条错例中的 5 条源于此）。

> 下载源：GitHub 走 `codeload.github.com`（2.8MB/s）、权重走 `hf-mirror.com`（2.5MB/s）——
> GitHub release 资产在国内只有 7KB/s，是死路。

## 五、实测成绩（Windows + Python 3.12 复测）

| 项目 | 成绩 |
|---|---|
| 意图路由（合成测试集 300 条） | **95.0%** |
| 意图路由（**手写真实话术留出集 40 条**） | **100%** |
| 意图路由（**独立盲测集 25 条**，与种子迭代完全分离） | **100%** |
| 记忆相关性 | **89.7%** |
| 自我评估（成败判断） | **100%** |
| 决策延迟 | **~5 ms/次** |
| 海马体写入（一次学会） | **~1 ms/条** |
| 从开源数据提取捆绑 | **~12 秒**（对比梯度训练 ~60 分钟） |
| 模型体积 | **4.1 MB**（码本 1.9MB + 账本 2.2MB） |
| 技能复用 | 命中置信 **1.00**，二次调用零重写 |

CI（GitHub Actions）在 push/PR 时自动跑：语法检查 → 脑机制验收 →
认知能力测试 → 留出集 → Agent 端到端冒烟（Python 3.10 / 3.12 矩阵）。

### v1.1 相对 v1.0 的关键修复

- **类型感知槽位**：种子模板的 `{}` 换成具名槽位 + 分类填充池，杜绝跨意图污染样本。
- **前额叶写入动词闸**：问句没有「记住/记下」类动词时写入票打折，
  修复「我手机是什么」被事实原文抢走的回归。
- **同义种子扩充**：「算下/读取/我叫什么/哪里用了/注释掉/替换成/来着/网址」等通用口语说法。
- **悬空代码清理**：移除从未存在的 `brain.py`/TinyBrain 引用路径。
- **路径统一**：`fetch_data.sh` 与 `ingest_data.py` 统一到 `data/raw/`。

## 六、诚实的边界

- 4.1MB 的全息脑**不会自由对话**。它的职责是：路由、召回、判断成败、寻址技能。
  语言生成靠知识库检索（23 万条问→答）与技能模板。
- 留出集 65 条（40 条 v1 + 25 条独立盲测）达到 100%，但**样本仍小**；
  提升手段是通用同义模板与先验闸门，不是对留出集逐句记忆——
  可以用 `holdout.py` 继续加新说法回归验证。
- HDC 是"高维近似检索"，不是语义理解：同义改写仍需要种子覆盖。
  好处是**每一条经验可读可删可追溯**，进化不会灾难性遗忘。

## 七、开源调研与路线图

对 torchhd / hdlib / hyperDB、mem0 / Letta / Zep / langmem、
kNN-LM / Neural Cache / Memory Layers、GLiNER / FewCLUE 等 14 个
项目与论文做了调研，可落地改进按「纯 NumPy / 需依赖 / 仅参考」三档
整理在 [docs/research/open-source-survey.md](docs/research/open-source-survey.md)。

优先路线（零依赖）：top-k 软投票温度校准 → 多信号检索融合 →
时间衰减与事实失效 → 情景巩固归档。
