# -*- coding: utf-8 -*-
"""检索式理解引擎：把开源训练数据重写成「查表即理解」。

不训练。几十万条真实指令/问答就是模型本身：
  route(text)  意图路由 —— 在 CLINC150 + 中文种子示例里做 kNN 投票
  answer(text) 知识问答 —— 在 firefly/alpaca 的 问→答 对里检索最佳答案
  score(a,b)   相关性打分（字符二元组重叠 + 可选小模型加权）

fast path: SQLite FTS5(自切词) 粗排 → 精确重叠率精排。
"""
import json
import os
import re
import sqlite3

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DB = os.path.join(HERE, "knowledge.db")
VOCAB = os.path.join(HERE, "vocab.json")
NPZ = os.path.join(HERE, "brain.npz")
META = os.path.join(HERE, "brain_meta.json")
INTENTS = ["write_file", "read_file", "list_dir", "search_code", "run_cmd",
           "run_python", "add_memory", "search_memory", "add_note",
           "fetch_url", "chat", "task", "confirm", "deny"]


def bigrams(text):
    words = re.findall(r"[a-zA-Z0-9_]+", text.lower())
    cjk = re.findall(r"[\u4e00-\u9fff]", text)
    toks = words + cjk + ["".join(cjk[i:i + 2]) for i in range(len(cjk) - 1)]
    return list(dict.fromkeys(toks))[:64]


def overlap(a, b):
    """重叠系数：|A∩B| / min(|A|,|B|) —— 短文本检索的稳健打分。"""
    A, B = set(bigrams(a)), set(bigrams(b))
    if not A or not B:
        return 0.0
    return len(A & B) / min(len(A), len(B))


class Understand:
    def __init__(self):
        self.db = sqlite3.connect(DB, check_same_thread=False) \
            if os.path.exists(DB) else None
        self.seeds = []                       # 中文种子常驻内存（~1400 行，绕过FTS召回）
        if self.db:
            try:
                self.seeds = self.db.execute(
                    "SELECT text,intent FROM examples WHERE src='seed'").fetchall()
            except Exception:
                self.seeds = []
        self._brain = None

    # ---------------- 可选：训练好的小模型作第四票（有就用，没有不影响）
    def brain(self):
        if self._brain is None and os.path.exists(NPZ) and os.path.exists(META) \
                and os.path.exists(VOCAB):
            try:
                from brain import TinyBrain
                cfg = json.load(open(META, encoding="utf-8"))["cfg"]
                b = TinyBrain(**cfg)
                b.load(NPZ)
                b.load_vocab(VOCAB)
                self._brain = b
            except Exception:
                self._brain = False
        return self._brain or None

    # ---------------- 意图路由（kNN 投票：每意图只取最强样本，防数量碾压）
    def route(self, text, topn=40):
        if not self.db:
            return "chat", 0.0, {}
        best_by_intent = {}

        def vote(intent, s, etext):
            if intent not in best_by_intent or s > best_by_intent[intent][0]:
                best_by_intent[intent] = (s, etext)

        # ---- 第一层：种子内存精扫（无召回损失；空白不影响匹配）
        t1 = re.sub(r"\s+", "", text.strip())
        for etext, intent in self.seeds:
            t2 = re.sub(r"\s+", "", etext.strip())
            if t1 == t2:
                s = 3.0
            elif (t1 in t2 or t2 in t1) and min(len(t1), len(t2)) >= max(
                    2, int(0.4 * max(len(t1), len(t2)))):
                s = max(overlap(t1, t2), 1.6)    # 包含关系要过长度闸
            else:
                s = overlap(t1, t2)
            vote(intent, s * 1.25, t2)

        # ---- 第二层：FTS 长尾（CLINC 英文等）
        has_cjk = bool(re.search(r"[\u4e00-\u9fff]", text))
        toks = bigrams(text)
        fts_q = " OR ".join('"%s"' % t for t in toks if t) or '""'
        try:
            rows = self.db.execute(
                "SELECT e.intent, e.text, e.src FROM examples_fts f "
                "JOIN examples e ON e.id=f.rowid WHERE examples_fts MATCH ? "
                "LIMIT 500", (fts_q,)).fetchall()
        except Exception:
            rows = []
        for intent, etext, src in rows:
            if src == "seed" or (has_cjk and not re.search(r"[\u4e00-\u9fff]", etext)):
                continue                         # 种子已内存精扫；中文不吃英文样本
            vote(intent, overlap(text, etext), etext)

        # ---- 第三层：小模型 tie-break
        b = self.brain()
        if b:
            try:
                ids = np.array([b.encode(text)], dtype=np.int64)
                lg = b.predict(ids, "intent")[0]
                e = np.exp(lg - lg.max()); sm = e / e.sum()
                top = int(lg.argmax())
                if sm[top] > 0.5:
                    vi = INTENTS[top]
                    cur = best_by_intent.get(vi, (0, ""))[0]
                    vote(vi, cur + float(sm[top]) * 0.6,
                         best_by_intent.get(vi, ("", ""))[1])
            except Exception:
                pass

        if not best_by_intent:
            return "chat", 0.0, {}
        rank = sorted(best_by_intent.items(), key=lambda kv: -kv[1][0])
        intent = rank[0][0]
        if len(rank) > 1 and rank[1][1][0] > 0:
            conf = min(1.0, (rank[0][1][0] / (rank[1][1][0] + 1e-9) - 1) / 2 + 0.5)
        else:
            conf = 1.0
        detail = {"rank": [(i, round(v[0], 2)) for i, v in rank[:5]],
                  "example": rank[0][1][1][:60]}
        if conf < 0.42:
            if self.answer(text):
                return "chat", conf, detail
        return intent, conf, detail

    def answer(self, text, k=3):
        if not self.db:
            return None
        toks = bigrams(text)
        fts_q = " OR ".join('"%s"' % t for t in toks if t) or '""'
        try:
            rows = self.db.execute(
                "SELECT k.q, k.a, k.src FROM knowledge_fts f "
                "JOIN knowledge k ON k.id=f.rowid WHERE knowledge_fts MATCH ? "
                "ORDER BY rank LIMIT 80", (fts_q,)).fetchall()
        except Exception:
            return None
        qb = set(bigrams(text))
        NOISE_MARKERS = ("自然语言推理", "前提：", "题目：", "阅读理解", "翻译成",
                         "根据以下", "下面这段", "写作文", "续写")
        scored = []
        for q, a, src in rows:
            if len(q) > 220:                     # 超长问题什么都沾边，纯噪声
                continue
            if any(mk in q for mk in NOISE_MARKERS):
                continue                         # 任务式包装题（NLI/作文/翻译）
            cb = set(bigrams(q))
            if not cb:
                continue
            cov = len(qb & cb) / max(1, len(qb))          # 查询覆盖率
            contain = len(qb & cb) / max(1, len(cb))      # 命中占样本比
            bonus = 0.15 if src == "alpaca" else 0.0      # GPT4 答案质量先验
            score = cov * 0.7 + contain * 0.3 + bonus
            scored.append((score, q, a))
        scored.sort(key=lambda t: -t[0])
        if not scored or scored[0][0] < 0.45:
            return None
        return [(round(s, 2), q, a) for s, q, a in scored[:k]]

    # ---------------- 相关性（个人记忆/技能复用）
    def score(self, a, b):
        s = overlap(a, b)
        br = self.brain()
        if br:
            try:
                ids = np.array([br.encode_pair(a, b)], dtype=np.int64)
                s = 0.6 * s + 0.4 * float(br.predict(ids, "mem")[0])
            except Exception:
                pass
        return s
