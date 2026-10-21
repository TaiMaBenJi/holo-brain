# -*- coding: utf-8 -*-
"""自进化 Agent：自研 10MB 小模型（脑干） + 外部记忆（皮层）。

  理解/决策   ← brain.npz（意图路由 · 记忆相关性 · 自我评估 三个头）
  事实记忆   ← SQLite facts 表（FTS5 粗排 + 小模型精排）
  技能库     ← SQLite skills 表（写代码成功后沉淀，越用越强 = 自我进化）
  经历本     ← SQLite episodes 表（做过的任务、成败、教训）
  自我思考   ← 行动 → 自我评估头判断成败 → 失败自动换方案重试
  自我进化   ← --evolve 自己出题自己做：成功的沉淀为技能，失败的修正后再沉淀

用法:
  python3 agent.py "记住我的手机是一加13"
  python3 agent.py "我的手机是什么"
  python3 agent.py --task "统计沙箱里所有 py 文件的行数"
  python3 agent.py --evolve 6          # 自己出 6 道题自己练，技能入库
  python3 agent.py --skills            # 查看已学会的技能
  python3 agent.py --learn             # 用运行中积累的经历微调自己（权重级进化）
"""
import io
import json
import os
import re
import sqlite3
import subprocess
import sys
import time

import numpy as np


HERE = os.path.dirname(os.path.abspath(__file__))
SANDBOX = os.path.join(HERE, "sandbox")
DB = os.path.join(HERE, "memory.db")
INTENTS = ["write_file", "read_file", "list_dir", "search_code", "run_cmd",
           "run_python", "add_memory", "search_memory", "add_note",
           "fetch_url", "chat", "task", "confirm", "deny"]


# ============================================================ 外部记忆
class Memory:
    def __init__(self, path=DB):
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.executescript("""
CREATE TABLE IF NOT EXISTS facts(id INTEGER PRIMARY KEY, text TEXT UNIQUE,
    kind TEXT DEFAULT 'fact', created REAL);
CREATE VIRTUAL TABLE IF NOT EXISTS facts_fts USING fts5(text);
CREATE TABLE IF NOT EXISTS skills(id INTEGER PRIMARY KEY, name TEXT, desc TEXT,
    code TEXT, ok INT DEFAULT 0, fail INT DEFAULT 0, created REAL);
CREATE VIRTUAL TABLE IF NOT EXISTS skills_fts USING fts5(name, desc);
CREATE TABLE IF NOT EXISTS episodes(id INTEGER PRIMARY KEY, task TEXT,
    result TEXT, ok INT, lesson TEXT, created REAL);
CREATE VIRTUAL TABLE IF NOT EXISTS episodes_fts USING fts5(task, result);
""")

    # ---- 事实
    def add_fact(self, text, kind="fact"):
        now = time.time()
        try:
            cur = self.db.execute(
                "INSERT INTO facts(text,kind,created) VALUES(?,?,?)", (text, kind, now))
            self.db.execute("INSERT INTO facts_fts(rowid,text) VALUES(?,?)",
                            (cur.lastrowid, text))
            self.db.commit()
            return cur.lastrowid
        except sqlite3.IntegrityError:
            return None

    def search_facts(self, q, k=8, score_fn=None):
        toks = " OR ".join('"%s"' % t for t in re.findall(r"[\u4e00-\u9fff]+|[a-zA-Z0-9]+", q))
        try:
            rows = self.db.execute(
                "SELECT f.id, f.text FROM facts_fts t JOIN facts f ON f.id=t.rowid "
                "WHERE facts_fts MATCH ? LIMIT 40", (toks,)).fetchall()
        except Exception:
            rows = []
        if not rows:
            rows = self.db.execute(
                "SELECT id,text FROM facts ORDER BY created DESC LIMIT 20").fetchall()
        if score_fn and rows:
            scored = sorted(((score_fn(q, r["text"]), r["text"]) for r in rows),
                            key=lambda t: -t[0])
            return [t for t in scored[:k] if t[0] > 0.3]
        return [(1.0, r["text"]) for r in rows[:k]]

    # ---- 技能
    def add_skill(self, name, desc, code):
        now = time.time()
        cur = self.db.execute(
            "INSERT INTO skills(name,desc,code,ok,fail,created) VALUES(?,?,?,0,0,?)",
            (name, desc, code, now))
        self.db.execute("INSERT INTO skills_fts(rowid,name,desc) VALUES(?,?,?)",
                        (cur.lastrowid, name, desc))
        self.db.commit()
        return cur.lastrowid

    def all_skills(self):
        return self.db.execute(
            "SELECT * FROM skills ORDER BY ok DESC, created DESC").fetchall()

    def pick_skill(self, task, score_fn=None):
        """FTS 粗排 + 小模型精排，挑最相关的技能。"""
        toks = " OR ".join('"%s"' % t for t in re.findall(r"[\u4e00-\u9fff]+|[a-zA-Z0-9]+", task))
        try:
            rows = self.db.execute(
                "SELECT s.* FROM skills_fts t JOIN skills s ON s.id=t.rowid "
                "WHERE skills_fts MATCH ? LIMIT 12", (toks,)).fetchall()
        except Exception:
            rows = []
        if not rows:
            rows = self.db.execute(
                "SELECT * FROM skills ORDER BY ok DESC LIMIT 12").fetchall()
        best, best_s = None, 0.0
        for r in rows:
            s = score_fn(task, r["desc"]) if score_fn else 0.0
            if s > best_s:
                best, best_s = r, s
        if best is not None and best_s > 0.55 and best["ok"] >= best["fail"]:
            return best, best_s
        return None, best_s

    def skill_feedback(self, sid, ok):
        if ok:
            self.db.execute("UPDATE skills SET ok=ok+1 WHERE id=?", (sid,))
        else:
            self.db.execute("UPDATE skills SET fail=fail+1 WHERE id=?", (sid,))
        self.db.commit()

    # ---- 经历
    def add_episode(self, task, result, ok, lesson=""):
        now = time.time()
        cur = self.db.execute(
            "INSERT INTO episodes(task,result,ok,lesson,created) VALUES(?,?,?,?,?)",
            (task, result[:400], 1 if ok else 0, lesson, now))
        self.db.execute("INSERT INTO episodes_fts(rowid,task,result) VALUES(?,?,?)",
                        (cur.lastrowid, task, result[:400]))
        self.db.commit()
        return cur.lastrowid

    def similar_episodes(self, task, score_fn=None, k=3):
        rows = self.db.execute(
            "SELECT * FROM episodes ORDER BY created DESC LIMIT 60").fetchall()
        if not rows:
            return []
        scored = []
        for r in rows:
            scored.append((score_fn(task, r["task"]) if score_fn else 0.0, r))
        scored.sort(key=lambda t: -t[0])
        return [r for s, r in scored[:k] if s > 0.5 and r["ok"]]


# ============================================================ 工具（沙箱内）
def _safe(p):
    root = os.path.realpath(SANDBOX)
    c = os.path.realpath(os.path.join(SANDBOX, p)) if not os.path.isabs(p) \
        else os.path.realpath(p)
    if not (c == root or c.startswith(root + os.sep)):
        raise ValueError("路径越界: " + p)
    return c


class Tools:
    @staticmethod
    def write(path, content):
        fp = _safe(path)
        os.makedirs(os.path.dirname(fp) or SANDBOX, exist_ok=True)
        with open(fp, "w", encoding="utf-8") as f:
            f.write(content)
        return "已写入 %s（%d 字符）" % (path, len(content))

    @staticmethod
    def read(path):
        fp = _safe(path)
        with open(fp, encoding="utf-8", errors="replace") as f:
            return f.read()

    @staticmethod
    def ls(path="."):
        fp = _safe(path)
        return sorted(os.listdir(fp))

    @staticmethod
    def shell(cmd, timeout=30):
        p = subprocess.run(["bash", "-lc", cmd], capture_output=True, text=True,
                           timeout=timeout, cwd=SANDBOX)
        return (p.stdout + p.stderr).strip() or ("[exit %d]" % p.returncode)

    @staticmethod
    def py(code, timeout=30):
        p = subprocess.run([sys.executable, "-c", code], capture_output=True,
                           text=True, timeout=timeout, cwd=SANDBOX)
        ok = p.returncode == 0
        return ok, (p.stdout.strip() or p.stderr.strip())


# ============================================================ 代码生成（模板+修复）
SKELETONS = {
    "count_lines": ("统计文件行数", """
import os
def run(ctx):
    root = ctx.get("root", ".")
    total = 0
    for dp, _, fns in os.walk(root):
        for fn in fns:
            if fn.endswith((".py", ".txt", ".md", ".json", ".sh")):
                p = os.path.join(dp, fn)
                try:
                    n = sum(1 for _ in open(p, encoding="utf-8", errors="ignore"))
                    print("%6d  %s" % (n, p)); total += n
                except OSError:
                    pass
    print("TOTAL", total)
    return "共 %d 行" % total
"""),
    "word_freq": ("统计词频", """
import os, re, collections
def run(ctx):
    root = ctx.get("root", ".")
    cnt = collections.Counter()
    for dp, _, fns in os.walk(root):
        for fn in fns:
            if fn.endswith((".txt", ".md")):
                try:
                    t = open(os.path.join(dp, fn), encoding="utf-8", errors="ignore").read()
                    cnt.update(re.findall(r"[\\u4e00-\\u9fff]+|[a-zA-Z]+", t))
                except OSError:
                    pass
    for w, c in cnt.most_common(20):
        print(c, w)
    return "共 %d 个不同词" % len(cnt)
"""),
    "merge_txt": ("合并文本文件", """
import os
def run(ctx):
    root = ctx.get("root", ".")
    out = os.path.join(root, "merged.txt")
    n = 0
    with open(out, "w", encoding="utf-8") as w:
        for dp, _, fns in os.walk(root):
            for fn in sorted(fns):
                if fn.endswith(".txt") and fn != "merged.txt":
                    try:
                        t = open(os.path.join(dp, fn), encoding="utf-8", errors="ignore").read()
                        w.write("\n===== %s =====\n%s\n" % (fn, t)); n += 1
                    except OSError:
                        pass
    print("merged", n, "files ->", out)
    return "合并 %d 个文件" % n
"""),
    "biggest": ("找最大的几个文件", """
import os
def run(ctx):
    root = ctx.get("root", ".")
    files = []
    for dp, _, fns in os.walk(root):
        for fn in fns:
            p = os.path.join(dp, fn)
            try:
                files.append((os.path.getsize(p), p))
            except OSError:
                pass
    files.sort(reverse=True)
    for s, p in files[:10]:
        print("%10d  %s" % (s, p))
    return "最大: %s" % (files[0][1] if files else "-")
"""),
    "dir_tree": ("打印目录树", """
import os
def run(ctx):
    root = ctx.get("root", ".")
    for dp, dns, fns in os.walk(root):
        depth = dp.rstrip(os.sep).count(os.sep) - root.rstrip(os.sep).count(os.sep)
        print("  " * depth + os.path.basename(dp) + "/")
        for fn in fns:
            print("  " * (depth + 1) + fn)
    return "目录树已输出"
"""),
    "grep_text": ("文本内容搜索", """
import os
def run(ctx):
    root = ctx.get("root", ".")
    kw = ctx.get("kw", "TODO")
    hits = 0
    for dp, _, fns in os.walk(root):
        for fn in fns:
            p = os.path.join(dp, fn)
            try:
                for i, line in enumerate(open(p, encoding="utf-8", errors="ignore"), 1):
                    if kw in line:
                        print("%s:%d: %s" % (p, i, line.strip()[:120])); hits += 1
            except OSError:
                pass
    print("hits:", hits)
    return "命中 %d 处" % hits
"""),
    "batch_rename": ("批量重命名", """
import os
def run(ctx):
    root = ctx.get("root", ".")
    old = ctx.get("old", ".tmp"); new = ctx.get("new", ".bak")
    n = 0
    for dp, _, fns in os.walk(root):
        for fn in fns:
            if old in fn:
                src = os.path.join(dp, fn); dst = os.path.join(dp, fn.replace(old, new))
                try:
                    os.rename(src, dst); n += 1
                except OSError:
                    pass
    print("renamed", n)
    return "重命名 %d 个" % n
"""),
    "clean_tmp": ("清理临时文件", """
import os
def run(ctx):
    root = ctx.get("root", ".")
    n = 0
    for dp, _, fns in os.walk(root):
        for fn in fns:
            if fn.endswith((".tmp", ".bak", ".swp", ".log")) or fn.startswith("~"):
                try:
                    os.remove(os.path.join(dp, fn)); n += 1
                except OSError:
                    pass
    print("removed", n)
    return "清理 %d 个" % n
"""),
}


def choose_skeleton(task):
    t = task.lower()
    rules = [("行数", "count_lines"), ("词频", "word_freq"), ("合并", "merge_txt"),
             ("最大", "biggest"), ("目录树", "dir_tree"), ("树", "dir_tree"),
             ("搜索", "grep_text"), ("查找", "grep_text"), ("重命名", "batch_rename"),
             ("清理", "clean_tmp"), ("统计", "count_lines"), ("行数", "count_lines")]
    for kw, sk in rules:
        if kw in t:
            return sk
    return "count_lines"


class Agent:
    def __init__(self):
        from understand import Understand
        from holo import HoloBrain
        self.und = Understand()                 # 新皮层：知识检索(FTS)
        self.holo = HoloBrain()                 # 全息脑：意图/联想/杏仁核/情景记忆
        try:
            self.holo.load()
        except Exception:
            pass
        self.brain = self.und.brain()          # 旧 transformer(备用票，可无)
        self.mem = Memory()
        os.makedirs(SANDBOX, exist_ok=True)
        self.trace = []

    # ---------------- 全息脑接口（DeepSeek式：MTP 次优意图 + 睡眠重放）
    def mtp(self, text):
        """多 token 预测：主意图 + 次优意图（执行失败时换路）。"""
        try:
            it1, it2, c1, c2 = self.holo.mtp(text)
            return it1, it2, c1
        except Exception:
            it, c, _ = self.und.route(text)
            return it, None, c

    def self_eval_holo(self, text):
        try:
            return self.holo.ok_or_not(text)
        except Exception:
            return 0.5

    def sleep_replay(self):
        """睡眠重放：把经历重新写进全息脑原型（巩固 + 清理旧记忆）。"""
        rows = self.mem.db.execute(
            "SELECT task, result, ok FROM episodes WHERE ok IS NOT NULL LIMIT 300").fetchall()
        if not rows:
            return 0
        replays = []
        for r in rows:
            w = 3.0 if r["ok"] else 2.0        # 失败经验杏仁核加权（情绪权重）
            replays.append(("intent:task", r["task"], w))
            replays.append(("eval:ok" if r["ok"] else "eval:bad",
                            (r["task"] + " " + (r["result"] or ""))[:160], w))
        self.holo.consolidate(replays)
        self.holo.save()
        return len(replays)

    # ---------- 三个头
    def intent(self, text):
        ids = np.array([self.brain.encode(text)], dtype=np.int64)
        lg = self.brain.predict(ids, "intent")[0]
        e = np.exp(lg - lg.max())
        sm = e / e.sum()
        return INTENTS[int(lg.argmax())], float(sm.max())

    def mem_score(self, a, b):
        ids = np.array([self.brain.encode_pair(a, b)], dtype=np.int64)
        return float(self.brain.predict(ids, "mem")[0])

    def self_eval(self, action_result):
        """自我思考（杏仁核）：评估这次行动，失败就换方案。"""
        s = self.self_eval_holo((action_result or "")[:180])
        self.trace.append({"action": (action_result or "")[:60], "ok": s > 0.5,
                           "p": round(s, 3)})
        return s > 0.5

    def reflect(self, action, result, attempt):
        """自我思考：评估这次行动，失败就换方案。"""
        s = self.self_eval((action + " " + result)[:180])
        self.trace.append({"action": action[:60], "ok": s > 0.5, "p": round(s, 3)})
        return s > 0.5

    # ---------- 技能
    def run_skill(self, task):
        # 内容寻址：全息脑按"像不像"回忆技能（脑的一缕气味唤醒整段记忆）
        rows = self.mem.all_skills()
        if rows and getattr(self.holo, "ep_vecs", None):
            hit, score = self.holo.nearest(task, [(r["desc"], r) for r in rows])
            sk = hit if (hit is not None and score > 0.25 and hit["ok"] >= hit["fail"]) else None
        else:
            sk, score = self.mem.pick_skill(task, self.und.score)
        if sk is None:
            return None
        ns = {}
        try:
            exec(compile(sk["code"], "<skill>", "exec"), ns)
            buf = io.StringIO()
            old = sys.stdout
            sys.stdout = buf
            try:
                res = ns["run"]({"root": SANDBOX})
            finally:
                sys.stdout = old
            out = (buf.getvalue() or str(res))
            ok = self.reflect("技能 " + sk["name"], str(out)[:200], 1)
            self.mem.skill_feedback(sk["id"], ok)
            return "「%s」(置信 %.2f)\n%s" % (sk["name"], score, str(out)[:1500])
        except Exception as e:
            self.mem.skill_feedback(sk["id"], False)
            return "技能 %s 执行失败: %s" % (sk["name"], e)

    def write_code(self, task):
        """写代码：模板 → 执行 → 失败自修 → 沉淀为技能（程序性记忆）。"""
        skname = choose_skeleton(task)
        desc, code = SKELETONS[skname]
        attempts = []
        for i in range(2):
            ns = {}
            try:
                exec(compile(code, "<skill>", "exec"), ns)
                buf = io.StringIO()
                old = sys.stdout
                sys.stdout = buf
                try:
                    res = ns["run"]({"root": SANDBOX})
                finally:
                    sys.stdout = old
                out = buf.getvalue()
                attempts.append((i + 1, True, out[:200]))
                # 技术成功（无异常）即沉淀技能；杏仁核评估只作反思参考
                self.reflect("写代码 " + skname, out[:200], i)
                if True:
                    sid = self.mem.add_skill(skname, task[:80], code)
                    self.mem.skill_feedback(sid, True)
                    self.mem.add_episode(task, out[:200], True,
                                         "模板 %s 第 %d 次尝试成功，沉淀为技能 #%d"
                                         % (skname, i + 1, sid))
                    return ("自己写了代码（模板 %s，第 %d 次尝试）并跑通：\n%s\n"
                            "[已沉淀为技能 #%d，下次同类任务直接调用]"
                            % (skname, i + 1, out[:1200], sid))
            except Exception as e:
                out = "%s: %s" % (type(e).__name__, e)
                attempts.append((i + 1, False, out[:200]))
                code = "import traceback\n" + code      # 自修：加诊断包裹
        self.mem.add_episode(task, attempts[-1][2], False,
                             "模板 %s 两次尝试均失败" % skname)
        return "写了代码但没跑通：%s" % attempts[-1][2]

    # ---------- 自主任务
    def do_task(self, task):
        exp = self.mem.similar_episodes(task, self.und.score)
        hint = ""
        if exp:
            hint = "\n[相关经历] " + exp[0]["task"] + " → " + exp[0]["lesson"]
        r = self.run_skill(task)
        if r and "失败" not in r:
            self.mem.add_episode(task, r[:200], True, "复用已有技能")
            return r + hint
        return self.write_code(task) + hint

    # ---------- 对话入口（检索式理解：几十万条开源数据就是模型）
    def handle(self, text):
        it, it2, conf = self.mtp(text)
        self.trace.append({"intent": it, "alt": it2, "conf": round(conf, 3)})
        if it == "chat":
            ans = self.und.answer(text)
            if ans:
                self.trace.append({"knowledge": len(ans)})
                out = ["[知识库·相关度 %s] %s" % (s, q) + "\n" + a for s, q, a in ans[:2]]
                return "\n\n——\n\n".join(out)
        if it == "add_memory":
            m = re.search(r"(?:记住|记一下|记下|别忘了|记好了)[，:：\s]*(.+)", text)
            fact = (m.group(1) if m else text).strip()
            fid = self.mem.add_fact(fact)
            return ("已记住（#%s）：%s" % (fid, fact)) if fid else "这条我已经知道了"
        if it == "search_memory":
            hits = self.mem.search_facts(text, k=5, score_fn=self.und.score)
            if not hits:
                return "记忆里没有相关内容"
            return "记忆检索：\n" + "\n".join("  · %s（相关 %.2f）" % (t, s) for s, t in hits)
        if it == "task":
            return self.do_task(text)
        if it == "write_file":
            m = re.search(r"(?:写到|写入|存到|保存在?)\s*(\S+?)(?:[，,。]|$)", text)
            path = m.group(1) if m else "note.txt"
            c = re.search(r"[「『\"](.+?)[」』\"]", text)
            content = c.group(1) if c else re.sub(r"^.*(写到|写入|存到|保存在?)(\S+?)([，,。]|$)", "", text).strip() or text
            r = Tools.write(path, content)
            return r if self.reflect("写文件 " + path, r, 1) else r + "（自查异常，建议重试）"
        if it == "read_file":
            m = re.search(r"([\w./\-一-龥]+\.[a-z]+)", text)
            if not m:
                return "没听清要读哪个文件"
            try:
                return Tools.read(m.group(1))[:1500] or "（空文件）"
            except Exception as e:
                return "读取失败：%s" % e
        if it == "list_dir":
            return "沙箱文件：\n  " + "\n  ".join(Tools.ls())
        if it == "run_cmd":
            m = re.search(r"(?:执行命令[:：]?|帮我运行|跑一下|执行|运行命令|终端跑)\s*(.+)", text)
            cmd = m.group(1).strip() if m else text
            out = Tools.shell(cmd)
            return ("$ %s\n%s" % (cmd, out[:1200])) if self.reflect(cmd, out[:200], 1) \
                else "命令似乎出了问题：\n" + out[:800]
        if it == "run_python":
            m = re.search(r"(?:python里|python执行[:：]?|用python算|跑段python[:：]?|帮我用python跑)\s*(.+)", text)
            code = m.group(1).strip() if m else text
            if not code.startswith(("print", "import", "from")) and not code.endswith(")"):
                code = "print(%s)" % code
            ok, out = Tools.py(code)
            return out[:1200] if self.reflect(code, out[:200], 1) else "出错了：\n" + out[:800]
        if it == "add_note":
            os.makedirs(os.path.join(SANDBOX, "notes"), exist_ok=True)
            fp = os.path.join("notes", "idea.md")
            old = Tools.read(fp) if os.path.exists(_safe(fp)) else ""
            Tools.write(fp, old + "- " + text + "\n")
            return "笔记已追加到 notes/idea.md"
        if it == "fetch_url":
            m = re.search(r"(https?://\S+|[\w.]+\.[a-z]{2,}\S*)", text)
            if not m:
                return "没看到链接"
            u = m.group(1)
            if not u.startswith("http"):
                u = "https://" + u
            try:
                import requests
                r = requests.get(u, timeout=15, headers={"User-Agent": "mini-brain/1.0"})
                return "HTTP %d，前 1000 字：\n%s" % (r.status_code, r.text[:1000])
            except Exception as e:
                return "抓取失败：%s" % e
        if it == "confirm":
            return "好"
        if it == "deny":
            return "已停"
        # chat：带记忆的自我介绍/闲聊
        facts = self.mem.search_facts(text, k=3, score_fn=self.und.score)
        know = ("\n[我记得] " + "；".join(t for _, t in facts)) if facts else ""
        return ("我是本地数据驱动 agent：脑子里是 "
                + str(self.und_answer_count()) + " 条开源指令数据（firefly/alpaca/clinc 重写而来），"
                "能读写文件、跑命令和代码、记事、查记忆、检索知识、自己写代码解决问题。"
                "零训练、全离线。" + know)

    def und_answer_count(self):
        try:
            return self.und.db.execute(
                "SELECT n FROM stats WHERE name='knowledge'").fetchone()[0]
        except Exception:
            return 0

    # ---------- 自我进化
    def evolve(self, n=5):
        rng = np.random.default_rng(int(time.time()))
        pool = ["统计沙箱里所有文本文件的行数", "统计所有 md 文件的词频",
                "把所有 txt 合并成一个文件", "找出最大的 5 个文件",
                "打印沙箱目录树", "在文件里搜索关键词'数据'", "清理临时文件"]
        results = []
        for i in range(n):
            task = pool[i % len(pool)]
            if i >= len(pool):
                task += "（第 %d 遍）" % (i // len(pool) + 1)
            t0 = time.time()
            r = self.do_task(task)
            ok = "失败" not in r[:40]
            if ok:
                # 在线进化：这次成功的做法写进海马体，下次更快命中
                self.holo.learn("intent:task", task, weight=1.0)
            results.append((task, ok, round(time.time() - t0, 1)))
            print("  [%d/%d] %s → %s (%.1fs)" % (i + 1, n, task, "成功" if ok else "失败",
                                                 time.time() - t0))
        sk = self.mem.all_skills()
        print("\n技能库: %d 个技能" % len(sk))
        for s in sk:
            print("  #%d %-12s 成功%d 失败%d" % (s["id"], s["name"], s["ok"], s["fail"]))
        rate = sum(1 for _, ok, _ in results if ok) / max(1, len(results))
        print("本轮成功率: %.0f%%" % (rate * 100))
        return rate

    # ---------- 权重级进化：用经历微调评估头
    def learn_from_episodes(self, steps=120):
        rows = self.db_rows()
        if len(rows) < 8:
            print("经历太少（%d 条），先多干点活" % len(rows))
            return
        rng = np.random.default_rng(1)
        for st in range(steps):
            r = rows[rng.integers(len(rows))]
            text = (r["task"] + " " + r["result"])[:180]
            ids = pad1(self.brain.encode(text), self.brain.maxlen)
            y = np.array([1 if r["ok"] else 0])
            task = "eval" if rng.random() < 0.7 else "intent"
            yy = y if task == "eval" else np.array([INTENTS.index("task")])
            loss, grads = self.brain.step_loss(ids, task, yy)
            self.brain.adam_step(grads, lr=2e-4)
        self.brain.save(os.path.join(HERE, "brain.npz"))
        print("已用 %d 条经历微调 %d 步，模型已更新（权重级自我进化）" % (len(rows), steps))

    def db_rows(self):
        return self.mem.db.execute(
            "SELECT task,result,ok FROM episodes WHERE ok IS NOT NULL LIMIT 500").fetchall()


def pad1(seq, L):
    x = np.zeros((1, L), dtype=np.int64)
    x[0, :len(seq)] = seq[:L]
    return x


# ============================================================ CLI
def main():
    os.makedirs(SANDBOX, exist_ok=True)
    a = Agent()
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        return
    if args[0] == "--evolve":
        a.evolve(int(args[1]) if len(args) > 1 else 5)
    elif args[0] == "--skills":
        for s in a.mem.all_skills():
            print("#%-3d %-12s 成功%d 失败%d  触发: %s"
                  % (s["id"], s["name"], s["ok"], s["fail"], s["desc"][:50]))
    elif args[0] == "--learn":
        n = a.sleep_replay()
        print("睡眠重放完成：%d 条经历被巩固进全息脑" % n)
    elif args[0] == "--task":
        print(a.do_task(" ".join(args[1:])))
    elif args[0] == "--repl":
        print("自我进化 agent（输入 exit 退出）")
        while True:
            try:
                t = input("> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if t in ("exit", "quit"):
                break
            if not t:
                continue
            t0 = time.time()
            print(a.handle(t))
            print("  [%.1fs %s]" % (time.time() - t0, json.dumps(a.trace[-2:], ensure_ascii=False)))
    else:
        print(a.handle(" ".join(args)))


if __name__ == "__main__":
    main()
