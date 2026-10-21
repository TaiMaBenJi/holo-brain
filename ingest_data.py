# -*- coding: utf-8 -*-
"""重写开源训练数据 → 本地知识/示例库（不训练，直接把数据变成理解能力）。

输入（data/raw/）:
  firefly.jsonl   ~110万条中文多任务指令（我们采样其中一部分）  {kind,instruction,input,output}
  alpaca_zh.jsonl 5.2万条 GPT-4 中文指令                       {instruction,input,output}
  clinc oos-eval  151类意图 23.7K条真实用户指令（英文）          data_full.json

输出（knowledge.db）:
  knowledge 表  问题→答案（FTS5 + 二元组分词，检索即理解）
  examples 表   文本→意图（kNN 检索式意图路由）
  词表统计划表
"""
import glob
import json
import os
import re
import sqlite3
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "data", "raw")
DB = os.path.join(HERE, "knowledge.db")

# ---------------- 生成中文意图示例（手写种子：真实说法 × 40+模板/意图）
SEEDS = {
    "write_file": ["把{}写到{}", "把{}写进{}", "创建文件{}，内容是{}", "新建{}",
                   "往{}里写点东西", "帮我写个文件{}", "保存到{}", "生成{}",
                   "在{}里记一下", "写个{}"],
    "read_file": ["看看{}", "读一下{}", "打开{}", "{}里写了什么", "查看{}",
                  "显示{}的内容", "帮我看下{}", "我要看{}", "读{}"],
    "list_dir": ["看看目录里有什么", "列出当前文件", "有哪些文件", "列一下目录",
                 "现在沙箱里都有啥", "看看都有些什么文件", "文件列表",
                 "当前目录内容", "这个文件夹里有什么", "帮我列个目录", "ls",
                 "看看都有什么", "展示一下目录"],
    "search_code": ["搜索{}", "找一下{}", "帮我搜{}", "在代码里找{}",
                    "哪里用到了{}", "查一下{}出现的位置", "搜一下{}",
                    "查找{}", "有没有关于{}的内容"],
    "run_cmd": ["执行命令：{}", "帮我运行 {}", "跑一下 {}", "执行 {}",
                "运行命令 {}", "在终端里执行 {}", "终端跑 {}", "{}"],
    "run_python": ["用python算 {}", "跑段python：{}", "执行python代码 {}",
                   "python 里 {} 结果是多少", "帮我用 python 跑 {}",
                   "运行 {}", "python执行：{}", "帮我算一下 {}"],
    "add_memory": ["记住{}", "帮我记一下：{}", "记下{}", "别忘了{}",
                   "记住这件事：{}", "把{}存到记忆里", "以后记住{}",
                   "帮我记住，{}", "我的{}，记一下"],
    "search_memory": ["我的{}是什么", "还记得{}吗", "我之前说过{}吗", "查一下{}",
                      "帮我查记忆：{}", "{}是什么来着", "你还记得{}吗",
                      "之前记的{}是多少", "找找关于{}的记忆", "{}是多少"],
    "add_note": ["记个笔记：{}", "帮我记笔记 {}", "写条笔记：{}",
                 "记到本子上：{}", "新笔记：{}", "把{}记到笔记里"],
    "fetch_url": ["打开{}", "帮我抓取{}的内容", "看看这个网页{}", "访问{}",
                  "把{}的内容拉下来", "读一下这个链接{}", "{} 上写了什么"],
    "task": ["统计一下所有文件的行数", "统计词频", "把这些文件合并成一个",
             "找出最大的几个文件", "打印目录树", "在文件里搜索关键词",
             "批量重命名文件", "清理临时文件", "帮我整理一下这些文件",
             "给这个目录做个索引", "分析一下这些数据", "做个汇总报告",
             "统计代码行数", "把日志按日期分类", "压缩备份这个目录"],
    "confirm": ["对", "是的", "没错", "好", "可以", "确认", "行", "就这样",
                "嗯", "OK", "好的，就这样办", "没问题"],
    "deny": ["不对", "不是", "取消", "算了", "不要了", "停止", "别做了",
             "取消吧", "停下", "这个不对，重来"],
    "chat": ["你好", "在吗", "你是谁", "你能做什么", "介绍一下自己",
             "什么是机器学习", "怎么学编程", "今天天气怎么样", "讲个笑话",
             "帮我解释一下这个概念"],
}

# CLINC 意图 → 我们的 14 意图 映射（检索式路由共享一套动作空间）
CLINC_MAP = {
    "transfer": "task", "deposit": "task", "balance": "search_memory",
    "transactions": "search_memory", "spending_history": "search_memory",
    "search_spendings": "search_code", "searching_accounts": "list_dir",
    "pay_bill": "task", "schedule_transfer": "task", "cancel_transfer": "deny",
    "exchange_rate": "fetch_url", "interest_rates": "fetch_url",
    "get_spending_history": "search_memory", "account_blocked": "chat",
    " greet": "chat", "greeting": "chat", "goodbye": "chat", "thank_you": "chat",
    "user_answer": "confirm", "affirm": "confirm", "cancel": "deny",
    "deny": "deny", "helpfulness": "chat", "instructions": "chat",
    "who_made_you": "chat", "who_are_you": "chat", "definition": "chat",
    "calculator": "run_python", "measurement_conversion": "run_python",
    "ingredient_substitution": "search_code", "recipe_search": "search_code",
    "weather": "fetch_url", "news_query": "fetch_url", "email": "write_file",
    "todo_update": "write_file", "add_to_list": "write_file", "remove_from_list": "write_file",
    "calendar_update": "write_file", "create_reminder": "add_note", "reminder": "add_note",
    "update_reminder": "add_note", "delete_reminder": "deny", "find_reminder": "search_memory",
    "calendar": "search_memory", "todo_list": "search_memory", "lists": "search_memory",
}
STOP_OOS = "oos"          # out-of-scope → chat


def bigrams(text):
    """中文按字+二元组、英文按词（ys5o 教训：FTS5 unicode61 不切中文，必须自己切）。"""
    words = re.findall(r"[a-zA-Z0-9_]+", text.lower())
    cjk = re.findall(r"[\u4e00-\u9fff]", text)
    toks = words + cjk + ["".join(cjk[i:i + 2]) for i in range(len(cjk) - 1)]
    return list(dict.fromkeys(toks))[:64]


def main():
    limit_kb = int(sys.argv[1]) if len(sys.argv) > 1 else 0      # 0 = 不限
    t0 = time.time()
    if os.path.exists(DB):
        os.remove(DB)
    db = sqlite3.connect(DB, check_same_thread=False)
    db.executescript("""
PRAGMA journal_mode=WAL;
CREATE TABLE knowledge(id INTEGER PRIMARY KEY, q TEXT, a TEXT, src TEXT);
CREATE VIRTUAL TABLE knowledge_fts USING fts5(q, a);
CREATE TABLE examples(id INTEGER PRIMARY KEY, text TEXT, intent TEXT, src TEXT);
CREATE VIRTUAL TABLE examples_fts USING fts5(toks);
CREATE TABLE stats(name TEXT PRIMARY KEY, n INTEGER);
""")
    ins_k = "INSERT INTO knowledge(q,a,src) VALUES(?,?,?)"
    ins_kf = "INSERT INTO knowledge_fts(rowid,q,a) VALUES(?,?,?)"
    ins_e = "INSERT INTO examples(text,intent,src) VALUES(?,?,?)"
    ins_ef = "INSERT INTO examples_fts(rowid,toks) VALUES(?,?)"

    n_k = n_e = 0

    # ---------------- firefly（中文多任务指令，量最大） {kind, input, target}
    fp = os.path.join(RAW, "firefly.jsonl")
    if os.path.exists(fp):
        batch, batchf = [], []
        with open(fp, encoding="utf-8", errors="ignore") as f:
            for line in f:
                try:
                    d = json.loads(line)
                except Exception:
                    continue
                q = (d.get("input") or "").strip()
                a = (d.get("target") or "").strip()
                kind = d.get("kind") or ""
                if not q or not a or len(a) > 2000:
                    continue
                if len(a) < 3 and kind not in ("Summary",):
                    continue
                batch.append((q[:800], a[:2000], "firefly"))
                batchf.append((q[:800], a[:2000]))
                if len(batch) >= 2000:
                    db.executemany(ins_k, batch)
                    db.executemany(ins_kf, [(None, q, a) for q, a in batchf])
                    n_k += len(batch)
                    batch, batchf = [], []
        if batch:
            db.executemany(ins_k, batch)
            db.executemany(ins_kf, [(None, q, a) for q, a in batchf])
            n_k += len(batch)
        print("firefly  → knowledge %d 条" % n_k, flush=True)

    # ---------------- alpaca-gpt4-zh（质量最高的中文指令）
    ap = os.path.join(RAW, "alpaca_zh.jsonl")
    if os.path.exists(ap):
        batch, batchf = [], []
        with open(ap, encoding="utf-8", errors="ignore") as f:
            for line in f:
                try:
                    d = json.loads(line)
                except Exception:
                    continue
                q = (d.get("instruction") or "").strip()
                inp = (d.get("input") or "").strip()
                a = (d.get("output") or "").strip()
                if not q or not a or len(a) > 2000:
                    continue
                if inp and len(inp) < 300:
                    q = q + "\n" + inp
                batch.append((q[:800], a[:2000], "alpaca"))
                batchf.append((q[:800], a[:2000]))
                if len(batch) >= 2000:
                    db.executemany(ins_k, batch)
                    db.executemany(ins_kf, [(None, q, a) for q, a in batchf])
                    n_k += len(batch)
                    batch, batchf = [], []
        if batch:
            db.executemany(ins_k, batch)
            db.executemany(ins_kf, [(None, q, a) for q, a in batchf])
            n_k += len(batch)
        print("alpaca   → knowledge 累计 %d 条" % n_k, flush=True)

    # ---------------- CLINC150（151 类真实用户指令 → 检索式意图路由样本）
    cands = glob.glob(os.path.join(RAW, "oos-eval*", "data", "data_full.json"))
    if cands:
        d = json.load(open(cands[0], encoding="utf-8"))
        for split in ("train", "val", "train_oos", "val_oos"):
            for text, intent in d.get(split, []):
                mapped = "chat" if intent == STOP_OOS else CLINC_MAP.get(intent, "task")
                toks = " ".join(bigrams(text))
                db.execute(ins_e, (text[:300], mapped, "clinc"))
                db.execute(ins_ef, (None, toks))
                n_e += 1
        print("clinc    → examples %d 条" % n_e, flush=True)


    fills = ["notes.txt", "todo.md", "数据", "配置", "行数", "词频", "成都",
             "我的手机型号", "服务器地址", "会议时间", "example.com", "pwd",
             "main.py", "日记.md", "一段话", "读后感", "2**32", "python脚本"]
    import random
    random.seed(3)
    for intent, tpls in SEEDS.items():
        for tpl in tpls:
            for _ in range(10):
                n = tpl.count("{}")
                text = tpl.format(*[random.choice(fills) for _ in range(n)])
                toks = " ".join(bigrams(text))
                db.execute(ins_e, (text[:300], intent, "seed"))
                db.execute(ins_ef, (None, toks))
                n_e += 1
    # CLINC 英文样本写入时加标记（查询含中文时在检索侧过滤）
    db.execute("INSERT INTO stats VALUES('knowledge',?)", (n_k,))
    db.execute("INSERT INTO stats VALUES('examples',?)", (n_e,))
    db.commit()
    print("完成: knowledge %d 条, examples %d 条, 用时 %.0fs"
          % (n_k, n_e, time.time() - t0))


if __name__ == "__main__":
    main()
