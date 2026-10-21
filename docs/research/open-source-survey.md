# holo-brain 开源调研报告

调研日期:2026-10-21。目标:为「HDC 双通路 + SQLite 记忆」自进化 Agent 找可落地改进。
状态更新(同日):本轮已完成「类型感知槽位填充 + 前额叶写入动词闸 + 同义种子扩充」,真实话术留出集 77.5%→100%(65/65,含 25 条独立盲测集);下表为后续路线。

## 一、超维计算库与研究

- **torchhd** — https://github.com/hyperdimensional-computing/torchhd
  PyTorch VSA 库,内置 MAP/BSC/HRR/FHRR/VTB 等 8 种绑定模型、hash_table(majority bundling)与原型分类。值得借鉴:VTB 非交换绑定可区分「角色×内容」,复数乘法绑定(FHRR)可纯 NumPy(numpy.fft)平移。
- **hdlib** — https://github.com/cumbof/hdlib
  纯 Python VSA 库,提供多级 bundling(求和/多数决/加权)与绑定 API。值得借鉴:编码器、绑定算子、相似度度量三者解耦的接口设计。
- **hyperDB** — https://github.com/jdagdelen/hyperDB
  本地向量库(SQLite 存储 + numpy 索引),增量 upsert + kNN。与 holo-brain 的 SQLite 外挂记忆同构,可参考其索引重建与增量插入策略。
- **近期研究**:GHRR 非交换绑定可表达 attention(https://arxiv.org/abs/2405.09689);「Attention as Binding」提出 binding/unbinding 头与超维记忆层(https://arxiv.org/abs/2512.14709);序列位置可用 Fractional Power Encoding 实现移位等变(https://arxiv.org/abs/2608.28334)。

## 二、Agent 记忆系统

- **mem0** — https://github.com/mem0ai/mem0
  v3 采用单次 ADD-only 抽取(不 UPDATE/DELETE)+ 实体链接;检索时语义、BM25、实体三路并行打分融合,并做时间感知排序。多信号融合可直接搬。
- **letta**(MemGPT)— https://github.com/letta-ai/letta
  分层记忆:可自我编辑的 core memory 块 + archival 外档 + recall 会话史;容量饱和时由 agent 自己摘要落盘。可抄「核心/情景分层 + 记忆自编辑」。
- **graphiti**(Zep)— https://github.com/getzep/graphiti
  时序知识图,每条事实带有效期窗,新信息让旧事实「失效而非删除」;语义+BM25+图遍历混合检索。可抄「事实失效标记」与混合检索。
- **langmem** — https://github.com/langchain-ai/langmem
  热路径即时写入、后台 manager 异步提取/巩固/去重的双速记忆;支持 agent 自主调用记忆工具。可抄「热存储/后台巩固」两阶段结构。

## 三、检索式/非参数语言模型

- **kNN-LM**(论文)— https://arxiv.org/abs/1911.00172
  用 datastore 近邻按相似度与先验概率插值,纯查表增强。可移植核心:`softmax(sim/τ)` 加权检索分数 + 插值系数。
- **Neural Cache**(论文)— https://arxiv.org/abs/1707.06878
  就地缓存近期键值、top-k 检索、权重即键相似度,无需额外索引。与情景记忆 kNN 同构,可加「命中计数」作置信信号。
- **Memory Layers**(Meta 2024)— https://huggingface.co/facebook/memory-layers
  可学 key-value 记忆库替代 FFN,推理时 top-k 检索。验证「参数换可检索记忆」路线,但需训练,仅学术参考。
- **knn-transformers**(RetoMaton)— https://github.com/neulab/knn-transformers
  为 kNN datastore 建自动机,压缩检索空间并缓存。可抄检索结果缓存/去重思路。

## 四、中文意图/小样本(免训练向)

- **GLiNER** — https://github.com/urchade/GLiNER
  零样本开放标签抽取:把意图名当标签类型,零训练输出意图+槽位,适合做开放集兜底。
- **FewCLUE** — https://github.com/CLUEbenchmark/FewCLUE
  中文少样本基准(IFLYTEK/TNEWS 等),用来校准本项目意图路由准确率的相对位置。
- **SetFit** — https://github.com/huggingface/setfit
  8-shot 对比学习+线性头,作为「允许轻量微调」时的对照上界(非免训练,仅基线)。

## 可落地改进清单

### (a) 纯 NumPy 零依赖(建议先做)

1. **多信号检索融合**:余弦 ⊕ 字符 n-gram BM25 ⊕ 访问频次,各自 softmax 后加权(mem0 三路融合思想)。收益:精确话术类意图 +1~3pt;代价:约 80 行,半天。
2. **top-k 软投票 + 温度校准**:邻居按 `exp(sim/τ)` 加权投票替代 argmax(kNN-LM)。收益:得到校准置信,反哺自评估头与拒识;代价:约 40 行。
3. **时间衰减与事实失效**:SQLite 加 last_seen,相似度乘 `exp(-λΔt)`;被用户否定的话术打 stale 标记(graphiti)。收益:过期话术污染下降;代价:约 50 行。
4. **情景巩固**:后台把同意图的低频记忆 majority-bundle 成「原型码」并归档原条目(langmem 双速 + torchhd bundling)。收益:记忆膨胀后检索精度/速度保持;代价:约 150 行。
5. **角色/轮次绑定**:user/assistant 片段先乘位置码再绑定(torchhd FHRR 复数乘法)。收益:多轮上下文意图串扰下降;代价:约 100 行,需回归测试。

### (b) 需新依赖

6. **jieba + BM25**:字符级 BM25 可零依赖,jieba 分词版对口语变体更稳。收益:检索鲁棒性;代价:一个纯 Python 依赖。
7. **GLiNER-tiny 中文**:零训练开放意图+槽位,作 HDC 低置信时兜底。收益:未知意图(开放集)识别↑;代价:引入 torch、模型约 100MB,与 4.3MB 卖点冲突——只做可选插件。
8. **bge-small-zh 双通路**:向量检索与 HDC 码并行、晚期分数融合。收益:语义泛化明显↑;代价:torch+模型体积,仅插件模式。

### (c) 仅学术参考

9. **Memory Layers**:需梯度训练 key/value,与零训练冲突;只借「top-k 检索替代稠密层」思想,用原型码做无梯度版。
10. **GHRR / Attention-as-binding**:非交换绑定能编码顺序,但需重构编码器;先在百行 toy 验证再谈迁移。
11. **kNN-LM / RetoMaton 原版**:依赖神经 LM hidden state,holo-brain 无 LM,仅插值公式可移植。
