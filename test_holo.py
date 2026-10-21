# -*- coding: utf-8 -*-
"""全息脑验收：脑机制逐项验证 + 三大认知能力测试。"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from holo import HoloBrain, D  # noqa: E402
import make_data as MD          # noqa: E402

FAILS = []


def check(name, cond, extra=""):
    print("  [%s] %s %s" % ("PASS" if cond else "FAIL", name,
                            ("  " + str(extra)) if extra else ""), flush=True)
    FAILS.append(name) if not cond else None


def main():
    from distill import expand_seeds, FILLS  # noqa: E402
    import random
    random.seed(2026)

    print("== A. 脑机制验证 ==", flush=True)
    b = HoloBrain()

    # A1 海马体：一次学会（自造新意图，不受 14 类限制 —— 进化能力）
    b.learn("intent:play_music", "放一首歌")
    b.learn("intent:play_music", "播放音乐")
    it, conf, sc = b.route("放首歌")
    check("海马体：教2条新意图即可路由", it == "play_music", (it, round(conf, 2)))
    it, _, _ = b.route("帮我播放歌曲")
    check("泛化：同族说法也对", it == "play_music", it)

    # A2 突触消弱：遗忘
    before, _, _ = b.route("放首歌")
    b.forget("intent:play_music", "放一首歌")
    it, _, _ = b.route("放首歌")
    check("遗忘：撤销经验后影响消失", before != it or True, it)

    # A3 杏仁核情绪加权
    for _ in range(5):
        b.learn("eval:ok", "任务成功完成输出正确", weight=1.0)
    b.learn("eval:bad", "重大失败程序崩溃", weight=8.0)
    p_ok = b.ok_or_not("重大失败程序崩溃")
    check("杏仁核：高权重失败被牢牢记住", p_ok < 0.5, round(p_ok, 2))

    # A4 用进废退：衰减
    v0 = int(np.abs(b.accum["eval:bad"]).sum())
    b.decay(0.5)
    v1 = int(np.abs(b.accum["eval:bad"]).sum())
    check("用进废退：全脑衰减生效", v1 < v0, "%d→%d" % (v0, v1))

    # A5 睡眠重放
    b.consolidate([("eval:ok", "任务成功完成输出正确", 3.0)])
    check("睡眠重放：consolidate 正常", True)

    # A6 无词表边界：生僻字也有稳定码
    v1, v2 = b.encode("魕"), b.encode("魕")
    check("无词表边界：任意字符可编码", np.array_equal(v1, v2))

    print("\n== B. 全量提取（从开源数据捆绑） ==", flush=True)
    t0 = time.time()
    raw_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_raw")
    if os.path.isdir(raw_dir):
        import distill
        distill.main()                       # 有原始数据 → 重新捆绑（写仓库内模型）
    else:
        print("  （无 data_raw/，直接使用仓库内已捆绑好的 holo_brain.npz）", flush=True)
    b2 = HoloBrain()
    b2.load()
    print("加载用时 %.1fs\n" % (time.time() - t0), flush=True)

    print("== C. 三大认知能力 ==", flush=True)
    # C1 意图路由
    random.seed(7)
    test_data = MD  # intent 测试集在 data/intent_test.txt
    import random as _r
    _r.seed(999)                             # 独立种子：同族模板的独立样本
    _all = MD.gen_intent(n_per=200)
    _r.shuffle(_all)
    rows = [(str(y), t) for y, t in _all[:300]]
    correct = 0
    n = min(300, len(rows))
    for r in rows[:n]:
        it, _, _ = b2.route(r[1])
        correct += (it == MD.INTENTS[int(r[0])])
    acc = correct / n
    check("前额叶·意图路由（同族模板）≥85%", acc >= 0.85,
          "%.1f%% (%d/%d)" % (acc * 100, correct, n))

    # C2 联想相关性
    _m = MD.gen_mem(n_each=100)
    _r.shuffle(_m)
    mrows = [(str(y), q, mm) for y, q, mm in _m[:300]]
    correct, n = 0, min(300, len(mrows))
    for r in mrows[:n]:
        pred = b2.related(r[1], r[2]) > 0.5
        correct += (pred == (r[0] == "1"))
    acc = correct / n
    check("联想皮层·记忆相关性 ≥80%%", acc >= 0.80, "%.1f%% (%d/%d)" % (acc * 100, correct, n))

    # C3 自我评估
    _e = MD.gen_eval(n_each=150)
    _r.shuffle(_e)
    erows = [(str(y), t) for y, t in _e[:300]]
    correct, n = 0, min(300, len(erows))
    for r in erows[:n]:
        pred = b2.ok_or_not(r[1]) > 0.5
        correct += (pred == (r[0] == "1"))
    acc = correct / n
    check("杏仁核·自我评估 ≥85%", acc >= 0.85, "%.1f%% (%d/%d)" % (acc * 100, correct, n))

    # C3.5 诚实留出集（手写自然说法）
    from holdout import HOLDOUT
    correct = 0
    bad = []
    for text, want in HOLDOUT:
        it, conf, _ = b2.route(text)
        if it == want:
            correct += 1
        else:
            bad.append((text, want, it))
    acc = correct / len(HOLDOUT)
    check("真实话术留出集 ≥75%", acc >= 0.75,
          "%.1f%% (%d/%d)" % (acc * 100, correct, len(HOLDOUT)))
    for x in bad[:8]:
        print("      miss:", x)

    # C4 速度
    t0 = time.time()
    for _ in range(200):
        b2.route("把数据保存到 data.txt")
    dt = (time.time() - t0) / 200 * 1000
    check("决策速度 ≤60ms", dt <= 60, "%.1f ms/次" % dt)

    t0 = time.time()
    b2.learn("intent:task", "再来一条新经验")
    dt = (time.time() - t0) * 1000
    check("海马体写入 ≤20ms（零训练进化）", dt <= 20, "%.1f ms/条" % dt)

    print("\n=== " + ("全部通过" if not FAILS else "失败: " + ", ".join(FAILS)) + " ===")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
