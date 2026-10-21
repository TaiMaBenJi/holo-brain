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

# ---------------- 槽位填充池（类型感知：URL 槽绝不填文件名，杜绝跨意图污染）
SLOTS = {
    "path": ["notes.txt", "todo.md", "config.json", "main.py", "日记.md", "readme.md",
             "data.csv", "脚本.sh", "tmp.txt", "配置.yaml", "日志.log", "report.md",
             "数据", "配置", "list.txt"],
    "content": ["你好世界", "会议改到三点", "库存42件", "hello world", "待办写周报",
                "一段话", "读后感", "接口地址 /api/v1", "id,name,age", "用户名 admin"],
    "cmd": ["ls -la", "pwd", "df -h", "git status", "pip list", "du -sh .", "date",
            "whoami", "uname -a", "free -h", "curl https://example.com",
            "wget -q https://example.com", "ping -c 3 example.com"],
    "code": ["print(1+1)", "2**32", "len('abcdef')", "sum(range(100))", "17*23",
             "[i*i for i in range(10)]", "print('你好')"],
    "mem": ["手机型号", "服务器地址", "会议时间", "项目代号", "喜欢的语言",
            "常驻城市", "读书口味", "宠物名字", "早餐偏好", "体重目标",
            "生日", "邮箱", "老家的城市", "待办事项"],
    "topic": ["TODO", "FIXME", "redis", "数据", "配置", "行数", "词频",
              "函数名", "接口地址", "main"],
    "note": ["灵感：做本地助手", "周会要点", "购物清单", "读书摘记", "健身计划"],
    "url": ["https://example.com", "http://news.qq.com", "github.com/openai",
            "www.zhihu.com/question/123", "https://hf-mirror.com"],
    "fact": ["我叫小明", "我住在成都", "我喜欢Python", "我手机是一加13",
             "会议改到周三三点", "服务器是10.0.0.5", "我养了只猫叫年糕",
             "我目标是减到70公斤", "我爱看科幻小说", "项目代号崔共界",
             "我生日是3月5号", "我的邮箱是 a@b.com"],
    "ext": ["py", "md", "txt", "json", "sh"],
    "old": ["2024", "旧名", "TODO", "旧"],
    "new": ["2025", "新名", "FIXME", "新"],
}

# ---------------- 生成中文意图示例（手写种子：真实说法 × 同义模板/意图）
# 槽位一律具名 {path}/{content}/…，由 SLOTS 提供类型正确的填充。
SEEDS = {
    "write_file": ["把{content}写到{path}", "把{content}写进{path}",
                   "创建文件{path}，内容是{content}", "新建{path}",
                   "往{path}里写点东西", "帮我写个文件{path}", "保存到{path}",
                   "生成{path}", "在{path}里记一下", "写个{path}",
                   "把{content}存到{path}", "在{path}中写入{content}", "写文件{path}"],
    "read_file": ["看看{path}", "读一下{path}", "打开{path}", "{path}里写了什么",
                  "查看{path}", "显示{path}的内容", "帮我看下{path}", "我要看{path}",
                  "读{path}", "读取{path}", "阅读{path}", "打开文件{path}",
                  "把{path}打开", "帮我把{path}打开", "打开{path}看看"],
    "list_dir": ["看看目录里有什么", "列出当前文件", "有哪些文件", "列一下目录",
                 "现在沙箱里都有啥", "看看都有些什么文件", "文件列表",
                 "当前目录内容", "这个文件夹里有什么", "帮我列个目录", "ls",
                 "看看都有什么", "展示一下目录", "目录下有哪些文件", "目录下都有什么",
                 "目录里有什么", "列出目录内容",
                 "有哪些{ext}文件", "目录下有哪些{ext}文件", "列出所有{ext}文件"],
    "search_code": ["搜索{topic}", "找一下{topic}", "帮我搜{topic}", "在代码里找{topic}",
                    "哪里用到了{topic}", "查一下{topic}出现的位置", "搜一下{topic}",
                    "查找{topic}", "有没有关于{topic}的内容",
                    "代码里哪里用了{topic}", "哪里用到过{topic}",
                    "搜搜代码里哪里用了{topic}", "在文件里搜索{topic}",
                    "全文搜索{topic}", "在文件里找找{topic}", "里头找找{topic}",
                    "帮我找找{topic}", "{topic}出现在哪", "{topic}出现在哪里",
                    "代码里{topic}在哪"],
    "run_cmd": ["执行命令：{cmd}", "帮我运行 {cmd}", "跑一下 {cmd}", "执行 {cmd}",
                "运行命令 {cmd}", "在终端里执行 {cmd}", "终端跑 {cmd}",
                "帮我敲个命令 {cmd}", "{cmd} 这条命令帮我跑了", "在终端跑一下{cmd}"],
    "run_python": ["用python算 {code}", "跑段python：{code}", "执行python代码 {code}",
                   "python 里 {code} 结果是多少", "帮我用 python 跑 {code}",
                   "python执行：{code}", "帮我算一下 {code}",
                   "算下{code}", "算一下{code}", "计算{code}", "计算一下{code}",
                   "{code}等于多少", "帮我算算{code}", "{code}的结果是多少"],
    "add_memory": ["记住{fact}", "帮我记一下：{fact}", "记下{fact}", "别忘了{fact}",
                   "记住这件事：{fact}", "把{fact}存到记忆里", "以后记住{fact}",
                   "帮我记住，{fact}", "记一下{fact}"],
    "search_memory": ["我的{mem}是什么", "还记得{mem}吗", "我之前说过{mem}吗",
                      "查一下{mem}", "帮我查记忆：{mem}", "{mem}是什么来着",
                      "你还记得{mem}吗", "之前记的{mem}是多少", "找找关于{mem}的记忆",
                      "我记的{mem}是多少", "我上次说的{mem}", "之前说的{mem}",
                      "我说过的{mem}", "我叫什么", "我是谁", "我的名字是什么",
                      "{mem}是啥来着", "我{mem}是啥", "查查{mem}是啥",
                      "我{mem}是啥来着", "我的{mem}是什么来着"],
    "add_note": ["记个笔记：{note}", "帮我记笔记 {note}", "写条笔记：{note}",
                 "记到本子上：{note}", "新笔记：{note}", "把{note}记到笔记里",
                 "帮我记一下笔记，{note}", "笔记 {note}"],
    "fetch_url": ["打开{url}", "帮我抓取{url}的内容", "看看这个网页{url}", "访问{url}",
                  "把{url}的内容拉下来", "读一下这个链接{url}", "{url} 上写了什么",
                  "抓一下{url}", "打开网址{url}", "访问一下{url}",
                  "把结果发到{url}", "发到{url}", "发送到{url}",
                  "打开这个网址{url}", "看看这个网址{url}"],
    "task": ["统计一下所有文件的行数", "统计词频", "把这些文件合并成一个",
             "找出最大的几个文件", "打印目录树",
             "批量重命名文件", "清理临时文件", "帮我整理一下这些文件",
             "给这个目录做个索引", "分析一下这些数据", "做个汇总报告",
             "统计代码行数", "把日志按日期分类", "压缩备份这个目录",
             "把{path}里的{old}替换成{new}", "全文替换{old}成{new}",
             "把{path}里的{old}全部替换", "注释掉{path}里的代码",
             "把{path}里的代码注释掉", "给{path}加上注释", "修改{path}的内容",
             "整理{path}", "格式化{path}", "把这些图片按尺寸分类",
             "数一下一共有多少个文件", "数数有多少个{ext}文件",
             "一共多少个文件", "统计一下{ext}文件的个数"],
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


    import random
    random.seed(3)
    for intent, tpls in SEEDS.items():
        for tpl in tpls:
            names = re.findall(r"\{(\w+)\}", tpl)
            if not names:
                text = tpl
                db.execute(ins_e, (text[:300], intent, "seed"))
                db.execute(ins_ef, (None, " ".join(bigrams(text))))
                n_e += 1
                continue
            for _ in range(10):          # 有槽位：类型感知随机填充 ×10
                text = tpl.format(**{nm: random.choice(SLOTS[nm]) for nm in names})
                db.execute(ins_e, (text[:300], intent, "seed"))
                db.execute(ins_ef, (None, " ".join(bigrams(text))))
                n_e += 1
    # CLINC 英文样本写入时加标记（查询含中文时在检索侧过滤）
    db.execute("INSERT INTO stats VALUES('knowledge',?)", (n_k,))
    db.execute("INSERT INTO stats VALUES('examples',?)", (n_e,))
    db.commit()
    print("完成: knowledge %d 条, examples %d 条, 用时 %.0fs"
          % (n_k, n_e, time.time() - t0))


if __name__ == "__main__":
    main()
