# -*- coding: utf-8 -*-
"""从开源数据提取知识 → 捆绑进全息脑（不是训练，是写记忆账本，秒级完成）。

提取源（已重写为本地数据的开源语料）：
  CLINC150（151类真实用户指令） + 中文手写种子  → 意图原型
  记忆问答主题对（make_data.MEM_SUBJECTS）      → 相关性原型
  行动结果样本（make_data.EVAL_OK/BAD）         → 自我评估原型
"""
import json
import os
import re
import random
import time

from holo import HoloBrain, D, INTENTS
import make_data as MD

HERE = os.path.dirname(os.path.abspath(__file__))


def expand_seeds():
    """意图种子：具名槽位类型感知填充（URL 槽绝不填文件名，杜绝跨意图污染）。

    单槽模板：全枚举（精确匹配必须存在，不靠随机抽奖）；
    多槽模板：确定性采样（组合空间太大时）。
    """
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "ing", os.path.join(HERE, "ingest_data.py"))
    ing = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ing)
    random.seed(3)
    out = []                                    # (intent, text)
    for intent, tpls in ing.SEEDS.items():
        for tpl in tpls:
            names = re.findall(r"\{(\w+)\}", tpl)
            if not names:
                out.append((intent, tpl))
                continue
            pools = [ing.SLOTS.get(nm) for nm in names]
            if any(p is None for p in pools):
                continue                        # 未知槽位：宁缺毋滥
            space = 1
            for p in pools:
                space *= len(p)
            if space <= 40:                     # 小空间：全枚举
                from itertools import product
                for vals in product(*pools):
                    out.append((intent, tpl.format(
                        **{names[i]: vals[i] for i in range(len(names))})))
            else:                               # 大空间：确定性采样
                for _ in range(16):
                    out.append((intent, tpl.format(
                        **{nm: random.choice(ing.SLOTS[nm]) for nm in names})))
    # CLINC150 真实英文指令（已按映射表折进 14 意图）
    import glob
    cands = glob.glob(os.path.join(HERE, "data", "raw", "oos-eval*",
                                   "data", "data_full.json"))
    if cands:
        d = json.load(open(cands[0], encoding="utf-8"))
        for split in ("train", "val"):
            for text, intent in d.get(split, []):
                mapped = "chat" if intent == "oos" else ing.CLINC_MAP.get(intent)
                if mapped:
                    out.append((mapped, text))
    return out


def main(save_path=None):
    t0 = time.time()
    brain = HoloBrain()
    random.seed(2026)

    # ---------------- 意图
    pairs = expand_seeds()
    for intent, text in pairs:
        brain.learn("intent:" + intent, text)
    print("意图: %d 条经验 → %d 个原型" % (len(pairs), len(brain.accum)), flush=True)

    # 第二套语料：make_data 的原始模板族（与种子库不同的说法，扩大覆盖）
    MD_TPL = MD.__dict__.get("TPL", {})
    SLOT = {"p": ["notes.txt", "todo.md", "main.py", "日志.log", "配置.yaml", "a.md"],
            "c": ["你好世界", "会议改到三点", "库存42件", "hello world", "待办写周报"],
            "cmd": ["ls -la", "pwd", "git status", "du -sh .", "date"],
            "code": ["print(1+1)", "2**32", "sum(range(100))", "len('abc')"],
            "q": ["手机型号", "项目代号", "服务器地址", "会议时间", "喜欢的语言"],
            "m": ["我叫小明", "我住在成都", "我喜欢Python", "我手机是一加13",
                  "会议改到周三三点", "服务器是10.0.0.5"],
            "n": ["灵感：做本地助手", "周会要点", "购物清单", "读书摘记"],
            "u": ["https://example.com", "github.com/openai", "hf-mirror.com"]}
    n2 = 0
    for intent, tpls in MD_TPL.items():
        if intent not in INTENTS:
            continue
        for tpl in tpls:
            names = re.findall(r"\{(\w+)\}", tpl)
            if not names:
                brain.learn("intent:" + intent, tpl)
                n2 += 1
                continue
            if any(nm not in SLOT for nm in names):
                continue                        # 未知槽位：跳过
            for _ in range(14):
                vals = {nm: random.choice(SLOT[nm]) for nm in names}
                try:
                    brain.learn("intent:" + intent, tpl.format(**vals))
                    n2 += 1
                except Exception:
                    pass
    print("第二套模板族: %d 条经验" % n2, flush=True)

    # ---------------- 记忆相关性（正/负经验对 —— 与推理用同一套成对编码）
    subs = MD.MEM_SUBJECTS
    n_rel = 0
    for _ in range(400):
        _, qs, ms = random.choice(subs)
        brain.related_learn(random.choice(qs), random.choice(ms), related=True)
        n_rel += 1
    for _ in range(400):
        a, b = random.sample(range(len(subs)), 2)
        brain.related_learn(random.choice(subs[a][1]),
                            random.choice(subs[b][2]), related=False)
        n_rel += 1
    for _ in range(250):
        _, qs, _ = random.choice(subs)
        brain.related_learn(random.choice(qs), random.choice(MD.FILLER), related=False)
        n_rel += 1
    print("联想皮层(相关性): %d 条经验" % n_rel, flush=True)

    # ---------------- 自我评估（失败经验情绪权重 ×2 —— 杏仁核机制）
    n_ev = 0
    for _ in range(400):
        brain.learn("eval:ok", random.choice(MD.EVAL_OK), weight=1.0)
        brain.learn("eval:bad", random.choice(MD.EVAL_BAD), weight=2.0)  # 失败记得更牢
        n_ev += 2
    print("杏仁核(自我评估): %d 条经验" % n_ev, flush=True)

    # ---------------- 保存（模型 = 码本 + 记忆账本）
    out = save_path or os.path.join(HERE, "holo_brain.npz")
    brain.save(out)
    size = os.path.getsize(out)
    print("保存 %s: %.2f MB（其中码本 %.2f MB）"
          % (out, size / 1048576, len(brain.base) * D / 1048576), flush=True)
    print("提取+捆绑用时: %.1f 秒（零梯度、零epoch）" % (time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
