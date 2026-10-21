# -*- coding: utf-8 -*-
"""HoloBrain 全息脑 —— 按人脑原理造的自研小模型（超维计算 HDC 架构）。

【仿脑映射】每一个组件都对应一个脑结构：

  脑结构            生理原理                     本系统实现
  ────────────────  ───────────────────────────  ─────────────────────────────
  神经元编码        稀疏分布式表示               2048维 {−1,+1} 全息码
  突触可塑性        赫布规则(一起激活就连接)     捆绑 = 逐维多数表决
  海马体            一次学会(情景记忆)           learn() 毫秒写入, 零梯度
  突触消弱          用进废退 / 遗忘              forget() 负写入 + decay() 衰减
  杏仁核            情绪加权(恐惧记最牢)         weight 参数: 失败经验 ×2
  睡眠重放          做梦时重放巩固记忆           consolidate() 重放+全脑衰减
  联想皮层          相关性直觉                   成对绑定码 rel:yes / rel:no
  前额叶            决策                         意图原型 14 个
  基底核            程序性记忆(技能)             技能库(配合 agent)
  新皮层            慢速语义知识                 23万条知识库(检索)
  内容寻址记忆       一缕气味唤醒整段回忆         sim() 全息寻址

模型 = 字符码本(~10MB) + 每个脑区一个几KB的记忆累积器。无梯度、无 epoch、无优化器。
"""
import hashlib
import json
import math
import os
import re

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
D = 4096                    # 全息码维度（≈果蝇蘑菇体神经元数, HDC 文献标准量级）
MAXLEN = 96
INTENTS = ["write_file", "read_file", "list_dir", "search_code", "run_cmd",
           "run_python", "add_memory", "search_memory", "add_note",
           "fetch_url", "chat", "task", "confirm", "deny"]


def base_vec(ch):
    """任何字符（含没见过的）都有稳定全息码 —— 确定性哈希生成，无词表边界。"""
    h = int(hashlib.sha256(ch.encode("utf-8")).hexdigest()[:16], 16)
    rng = np.random.default_rng(h)
    return (rng.integers(0, 2, D, dtype=np.int8) * 2 - 1)


def _grams(text):
    """字符 2/3-gram 词袋（腹侧流用）。"""
    s = re.sub(r"\s+", "", text)
    c = [c for c in s]
    out = set()
    for n in (2, 3):
        for i in range(len(c) - n + 1):
            out.add("".join(c[i:i + n]))
    if not out and c:
        out.add(c[0])
    return out


def overlap_fallback(a, b):
    """冷启动兜底（原型未建立时）：字符二元组重叠。"""
    def bg(s):
        words = re.findall(r"[a-zA-Z0-9_]+", s.lower())
        cjk = re.findall(r"[\u4e00-\u9fff]", s)
        t = words + cjk + ["".join(cjk[i:i + 2]) for i in range(len(cjk) - 1)]
        return set(t)
    A, B = bg(a), bg(b)
    if not A or not B:
        return 0.0
    return len(A & B) / min(len(A), len(B))


class HoloBrain:
    def __init__(self):
        self.base = {}      # 字符码本
        self.accum = {}     # 各脑区的记忆累积器（整数账本）
        self.proto = {}     # 各脑区的原型 = sign(累积器)
        self._ep = {}       # 编码缓存
        # 情景记忆库（海马体）：逐条存，kNN 全息寻址
        self.ep_vecs = []   # List[np.int8 (D,)]
        self.ep_lab = []    # List[str]
        self.ep_txt = []    # List[str] 原文（腹侧流语义匹配用）
        self.ep_grams = []  # List[frozenset] 预计算词袋（避免每次重算）
        self.df = {}        # n-gram 文档频次（算 IDF：显著性加权）
        self.ndocs = 0
        self.inv = {}       # 倒排索引: gram -> np.array(样本下标)（腹侧流高速路径）
        self._E = None      # 情景码矩阵缓存（float32，背侧流用）
        self._Z = None      # MLA 式潜空间记忆（DeepSeek-V3：压缩后再检索）
        self.latent_dim = 192
        self._dirty = True
        self._R = None

    # ---------------- 基础算子（脑的"物理"）
    def B(self, ch):
        v = self.base.get(ch)
        if v is None:
            v = base_vec(ch)
            self.base[ch] = v
        return v

    @staticmethod
    def roll(v, k):
        return np.roll(v, k % D)

    @staticmethod
    def bind(a, b):
        return a * b

    @staticmethod
    def sim(a, b):
        return float(np.dot(a.astype(np.int16), b.astype(np.int16)) / D)

    # ---------------- 感知：文本 → 全息码（位置无关的 n-gram 全息包）
    def encode(self, text):
        """标准 HDC 文本编码：短语内旋转保留语序，全局不旋转保持位置不变性。

        之前给 n-gram 又叠了"全局位置旋转"，导致同一句短语出现在不同位置
        编码就不同 → 检索力崩掉（实测路由 58%）。去掉全局位置后是袋装 n-gram。
        """
        key = ("t", text)
        hit = self._ep.get(key)
        if hit is not None:
            return hit
        s = re.sub(r"\s+", "", text)[:MAXLEN] or "<empty>"
        codes = [self.B(c) for c in s]
        acc = np.zeros(D, dtype=np.int32)
        # 2/3/4-gram 全息绑定（短语内旋转，不加全局位置）
        for n in (2, 3, 4):
            for i in range(len(codes) - n + 1):
                g = codes[i]
                for k in range(1, n):
                    g = g * self.roll(codes[i + k], k)
                acc += g
        # 单字（位置无关）
        for v in codes:
            acc += v
        # 英文整词
        for w in re.findall(r"[a-zA-Z]{2,}", s):
            wv = np.ones(D, dtype=np.int8)
            for j, ch in enumerate(w):
                wv = wv * self.roll(self.B(ch), j)
            acc += wv
        out = np.where(acc > 0, 1, -1).astype(np.int8)
        if len(self._ep) > 4096:
            self._ep.clear()
        self._ep[key] = out
        return out

    def encode_pair(self, a, b):
        key = ("p", a, b)
        hit = self._ep.get(key)
        if hit is not None:
            return hit
        out = self.bind(self.encode(a), self.roll(self.encode(b), 511))
        if len(self._ep) > 4096:
            self._ep.clear()
        self._ep[key] = out
        return out

    # ---------------- 海马体：一次学会 / 用进废退 / 情绪加权
    def _reindex(self):
        """构建倒排索引（腹侧流：只看命中的样本，O(命中数) 而非 O(总样本)）。"""
        self.inv = {}
        for i, g in enumerate(self.ep_grams):
            for x in g:
                self.inv.setdefault(x, []).append(i)
        self.inv = {k: np.array(v, dtype=np.int32) for k, v in self.inv.items()}
        if self.ep_vecs:
            self._E = np.stack(self.ep_vecs[-60000:]).astype(np.float32)
            # MLA 式潜空间：随机投影把 4096 维码压到 latent_dim 维再检索
            # （Johnson–Lindenstrauss：内积近似保持，存储降 20 倍）
            rng = np.random.default_rng(20261020)
            R = (rng.integers(0, 2, (D, self.latent_dim), dtype=np.int8) * 2 - 1)
            self._R = R
            self._Z = (self._E @ R.astype(np.float32)).astype(np.float32)
        self._dirty = False      # 重建完毕，别每次都重建

    def _acc(self, name):
        if name not in self.accum:
            self.accum[name] = np.zeros(D, dtype=np.int32)
            self.proto[name] = np.ones(D, dtype=np.int8)
        return self.accum[name]

    def learn(self, name, text, sign=+1, weight=1.0, vec=None, episodic=True):
        """写记忆。sign=+1 学会 / −1 撤销；weight=情绪权重（杏仁核标记）。

        情景记忆同步入库：脑的情景记忆本来就是逐条存的（kNN 寻址用）。
        """
        acc = self._acc(name)
        v = vec if vec is not None else self.encode(text)
        acc += int(round(sign * weight)) * v.astype(np.int32)
        self.proto[name] = np.where(acc > 0, 1, -1).astype(np.int8)
        if episodic and sign > 0 and name.startswith("intent:"):
            self.ep_vecs.append(v)
            self.ep_lab.append(name.split(":", 1)[1])
            self.ep_txt.append(text)
            gs = _grams(text)
            self.ep_grams.append(frozenset(gs))
            for g in gs:
                self.df[g] = self.df.get(g, 0) + 1
            self.ndocs += 1
            self._dirty = True

    # ---------------- 突触可塑性：用进废退 / 遗忘 / 睡眠重放
    def forget(self, name, text):
        """撤销一条经验（突触弱化到抵消）。"""
        self.learn(name, text, sign=-1, episodic=False)

    def decay(self, factor=0.97):
        """用进废退：全脑突触轻微弱化（睡眠期的记忆清洗）。"""
        for k in self.accum:
            self.accum[k] = (self.accum[k] * factor).astype(np.int32)
            self.proto[k] = np.where(self.accum[k] > 0, 1, -1).astype(np.int8)

    def consolidate(self, replays):
        """睡眠重放：衰减全脑 → 重放要巩固的经验（重要事件因反复强化而留存）。
        replays = [(脑区名, 文本, 情绪权重), ...]"""
        self.decay(0.96)
        for name, text, w in replays:
            self.learn(name, text, weight=w)

    # ---------------- 联想皮层：相关性
    def related_learn(self, a, b, related=True, weight=1.0):
        name = "rel:yes" if related else "rel:no"
        acc = self._acc(name)
        acc += int(round(weight)) * self.encode_pair(a, b).astype(np.int32)
        self.proto[name] = np.where(acc > 0, 1, -1).astype(np.int8)

    def related(self, a, b):
        if "rel:yes" not in self.proto and "rel:no" not in self.proto:
            return overlap_fallback(a, b)
        p = self.encode_pair(a, b)
        s_yes = self.sim(p, self.proto["rel:yes"]) if "rel:yes" in self.proto else 0.0
        s_no = self.sim(p, self.proto["rel:no"]) if "rel:no" in self.proto else 0.0
        return max(0.0, min(1.0, (s_yes - s_no) / 0.20 / 2 + 0.5))

    def _idf(self, gram):
        if not self.ndocs:
            return 1.0
        return math.log((self.ndocs + 1) / (self.df.get(gram, 0) + 1)) + 1.0

    def ventral(self, text):
        """腹侧流（语义通路）：IDF 加权的字符 n-gram 词袋覆盖率。

        罕见模式获得更大增益 —— 对应皮层「高信息模式增益更大」。
        倒排索引只扫命中样本，复杂度 O(命中数)。
        每个能力取其最好匹配的一条，样本多的类不会靠数量碾压。
        """
        if self._dirty or not self.inv:
            self._reindex()
        qg = _grams(text)
        if not qg or not self.ep_lab:
            return []
        acc = np.zeros(len(self.ep_lab), dtype=np.float32)
        tot = 0.0
        for g in qg:
            w = self._idf(g)
            tot += w
            idx = self.inv.get(g)
            if idx is not None and len(idx):
                np.add.at(acc, idx, w)
        if tot <= 0:
            return []
        acc /= tot
        best = {}
        for i, lab in enumerate(self.ep_lab):
            if acc[i] > best.get(lab, 0.0):
                best[lab] = float(acc[i])
        return sorted(best.items(), key=lambda t: -t[1])[:8]

    def route(self, text, topk=5):
        """前额叶决策 = 腹侧流(语义) ⊕ 背侧流(结构全息) 双通路融合。

        腹侧流擅长同义/近义，背侧流擅长词组结构 —— 单用任一条都不够，
        大脑就是双通路 + 流间整合。
        """
        votes = {}
        for lab, s in self.ventral(text):
            votes[lab] = votes.get(lab, 0.0) + s * 0.8
        if self.ep_vecs:
            if self._dirty or self._E is None:
                self._reindex()
            v = self.encode(text)
            sims = (self._E @ v.astype(np.float32)) / D
            best = {}
            for i in np.argsort(-sims)[:40]:
                lab = self.ep_lab[i]
                if float(sims[i]) > best.get(lab, -2.0):
                    best[lab] = float(sims[i])
            for lab, s in best.items():
                votes[lab] = votes.get(lab, 0.0) + s * 0.25
            # 潜空间检索（MLA 精神：压缩后再比较），额外一票
            if self._Z is not None:
                zq = (v.astype(np.float32) @ self._R.astype(np.float32))
                zs = (self._Z @ zq) / (D * self.latent_dim)
                bl = {}
                for i in np.argsort(-zs)[:20]:
                    lab = self.ep_lab[i]
                    if float(zs[i]) > bl.get(lab, -2.0):
                        bl[lab] = float(zs[i])
                for lab, s in bl.items():
                    votes[lab] = votes.get(lab, 0.0) + s * 0.08
            v = self.encode(text)
            for n, p in self.proto.items():
                if n.startswith("intent:"):
                    lab = n.split(":", 1)[1]
                    votes[lab] = votes.get(lab, 0.0) + self.sim(v, p) * 0.10
        if not votes:
            return "chat", 0.0, {}
        # 前额叶先验：写入类意图需要写入动词。
        # 事实原文（"记住我手机是一加13"）与问句（"我手机是什么"）共享大量字面，
        # 但问句没有"记住/记下"类动词 → 写入票打折，避免事实抢走召回。
        _ADD_VERBS = ("记住", "记下", "帮我记", "记一下", "记好", "别忘了",
                      "存到记忆", "存进记忆", "以后记住", "帮我记住", "存到本机记忆")
        if "add_memory" in votes and not any(k in text for k in _ADD_VERBS):
            votes["add_memory"] *= 0.55
        rank = sorted(votes.items(), key=lambda kv: -kv[1])
        intent = rank[0][0]
        conf = 1.0 if len(rank) == 1 else \
            max(0.0, min(1.0, (rank[0][1] - rank[1][1]) / 0.25 / 2 + 0.5))
        return intent, conf, {k: round(v, 3) for k, v in rank[:5]}

    def mtp(self, text, topk=2):
        """MTP 多 token 预测（DeepSeek-V3 思想）：一次给出前 k 个意图与置信。
        第一个是主意图，第二个是次优兜底（主意图执行失败时换路）。"""
        it, conf, rank = self.route(text)
        items = list(rank.items())[:topk]
        if len(items) < 2:
            return it, None, conf, 0.0
        return items[0][0], items[1][0], conf, items[1][1]

    def ok_or_not(self, text, vec=None):
        if "eval:ok" not in self.proto:
            return 0.5
        v = vec if vec is not None else self.encode(text)
        s_ok = self.sim(v, self.proto["eval:ok"])
        s_bad = self.sim(v, self.proto["eval:bad"])
        return max(0.0, min(1.0, (s_ok - s_bad) / 0.20 / 2 + 0.5))

    # ---------------- 内容寻址：全息回忆
    def nearest(self, text, items, encode_key=True):
        v = self.encode(text)
        best, best_s = None, -2.0
        for key, payload in items:
            kv = self.encode(key) if encode_key else key
            s = self.sim(v, kv)
            if s > best_s:
                best, best_s = payload, s
        return best, best_s

    # ---------------- 存取（模型文件 = 码本 + 记忆账本）
    def save(self, path=os.path.join(HERE, "holo_brain.npz")):
        if self.ep_vecs:
            np.savez_compressed(path,
                                **{("acc:" + k): v for k, v in self.accum.items()},
                                **{("base:" + k): v for k, v in self.base.items()},
                                ep_vecs=np.stack(self.ep_vecs),
                                ep_lab=np.array(self.ep_lab),
                                ep_txt=np.array([json.dumps(self.ep_txt, ensure_ascii=False)]),
                                ep_grams=np.array([json.dumps(
                                    [sorted(g) for g in self.ep_grams], ensure_ascii=False)]),
                                df=np.array([json.dumps(self.df)]),
                                ndocs=np.array([self.ndocs]))
        else:
            np.savez_compressed(path,
                                **{("acc:" + k): v for k, v in self.accum.items()},
                                **{("base:" + k): v for k, v in self.base.items()})

    def load(self, path=os.path.join(HERE, "holo_brain.npz")):
        d = np.load(path)
        for k in d.files:
            if k.startswith("acc:"):
                self.accum[k[4:]] = d[k]
                self.proto[k[4:]] = np.where(d[k] > 0, 1, -1).astype(np.int8)
            elif k.startswith("base:"):
                self.base[k[5:]] = d[k]
        if "ep_vecs" in d.files:
            self.ep_vecs = list(d["ep_vecs"])
            self.ep_lab = list(d["ep_lab"])
            self.ep_txt = json.loads(str(d["ep_txt"][0])) if "ep_txt" in d.files else []
            if "ep_grams" in d.files:
                self.ep_grams = [frozenset(g) for g in
                                 json.loads(str(d["ep_grams"][0]))]
            if "df" in d.files:
                self.df = json.loads(str(d["df"][0]))
                self.ndocs = int(d["ndocs"][0])
        self._dirty = True
