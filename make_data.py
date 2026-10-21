# -*- coding: utf-8 -*-
"""造训练数据：中文意图 + 记忆相关性 + 自我评估，模板+词槽+噪声合成，零外部依赖。

三份数据对应自研小模型的三个头：
  intent : 意图路由（14 类）
  mem    : 记忆相关性（query↔memory 二分类）
  eval   : 自我评估（这次行动成功了吗 → 继续还是重试）
"""
import os
import random
import re

random.seed(20261020)
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
os.makedirs(OUT, exist_ok=True)

# ============================================================ 意图（14 类）
INTENTS = ["write_file", "read_file", "list_dir", "search_code", "run_cmd",
           "run_python", "add_memory", "search_memory", "add_note",
           "fetch_url", "chat", "task", "confirm", "deny"]

PATHS = ["notes.txt", "todo.md", "config.json", "main.py", "日记.md", "readme.md",
         "data.csv", "脚本.sh", "tmp.txt", "配置.yaml", "index.html", "app.py",
         "report.md", "list.txt", "cache.log"]
CONTENTS = ["你好世界", "今天买牛奶", "会议改到三点", "hello world", "库存 42 件",
            "待办：写周报", "接口地址 /api/v1", "用户名 admin", "测试数据一行",
            "id,name,age"]
CMDS = ["ls -la", "pwd", "df -h", "git status", "pip list", "du -sh .",
        "date", "whoami", "uname -a", "free -h"]
PYCODE = ["print(1+1)", "print('你好')", "2**32", "len('abcdef')",
          "[i*i for i in range(10)]", "sum(range(100))"]
QUERIES = ["手机型号", "我的偏好", "项目代号", "服务器地址", "会议时间",
           "喜欢的语言", "老家的城市", "待办事项", "宠物名字", "读书口味"]
NOTES = ["灵感：做一个本地助手", "周会要点", "购物清单", "读书摘记", "健身计划",
         "改 bug 思路", "明天待办"]
URLS = ["https://example.com", "http://news.qq.com", "github.com/openai",
        "www.zhihu.com/question/123", "https://hf-mirror.com"]
CHITCHAT = ["你好", "在吗", "你是谁", "讲个笑话", "你会什么", "介绍一下你自己",
            "嗨", "hello", "你好呀", "晚安", "谢谢", "今天心情不错"]
TASKS = ["把这个目录里的 py 文件都加上类型注解", "整理笔记并生成索引",
         "统计代码行数写进报告", "把所有 txt 合并成一个文件",
         "找出最大的三个文件", "写个脚本统计词频", "清理临时文件",
         "批量重命名图片", "生成目录树", "对比两个文件的差异"]
CONFIRMS = ["对", "是的", "没错", "好", "可以", "确认", "行", "就这样", "嗯", "OK"]
DENYS = ["不对", "不是", "取消", "算了", "不要了", "停止", "别做了", "不对哦", "取消吧"]

TPL = {
 "write_file": [
   "把{c}写到{p}", "创建文件{p}，内容是{c}", "新建{p}，写入{c}", "往{p}写{c}",
   "存一份{c}到{p}", "写{p}：{c}", "保存{c}到文件{p}", "在{p}里保存{c}",
   "生成{p}，内容为{c}", "把这段话存到{p}：{c}"],
 "read_file": [
   "看看{p}", "读一下{p}", "打开{p}", "帮我看下{p}的内容", "查看{p}",
   "显示{p}的内容", "我要看{p}", "{p}的内容是什么", "读{p}给我", "看下{p}写了啥"],
 "list_dir": [
   "看看目录里有什么", "列出当前文件", "有哪些文件", "列一下目录",
   "现在沙箱里都有啥", "看看都有些什么文件", "文件列表", "当前目录内容",
   "这个文件夹里有什么", "帮我列个目录", "ls", "看看都有什么"],
 "search_code": [
   "搜索{q}相关的代码", "找一下包含{q}的文件", "帮我搜{q}", "在代码里找{q}",
   "哪里用到了{q}", "查一下{q}出现的位置", "搜一下{q}这个关键词",
   "查找{q}", "有没有关于{q}的内容", "帮我找找{q}"],
 "run_cmd": [
   "执行命令：{cmd}", "帮我运行 {cmd}", "跑一下 {cmd}", "执行 {cmd}",
   "运行命令 {cmd}", "帮我敲个命令 {cmd}", "在终端里执行 {cmd}",
   "终端跑 {cmd}", "{cmd} 这条命令帮我跑了"],
 "run_python": [
   "用python算 {code}", "跑段python：{code}", "执行python代码 {code}",
   "python 里 {code} 结果是多少", "帮我用 python 跑 {code}", "运行 {code}",
   "python执行：{code}", "{code} 用 python 算下"],
 "add_memory": [
   "记住我叫小明", "帮我记一下：我住在成都", "记下我喜欢Python",
   "别忘了会议改到周三三点", "记住这件事：服务器是10.0.0.5",
   "我养了只猫叫年糕，记一下", "把我的手机型号存到记忆里：一加13",
   "记好了：我目标是减到70公斤", "以后记住我爱看科幻小说",
   "帮我记住，项目代号崔共界", "记一下我的生日是3月5号"],
 "search_memory": [
   "我的{q}是什么", "还记得{q}吗", "我之前说过{q}吗", "查一下{q}",
   "帮我查记忆：{q}", "{q}是什么来着", "你还记得{q}吗", "查{q}",
   "之前记的{q}是多少", "找找关于{q}的记忆"],
 "add_note": [
   "记个笔记：{n}", "帮我记笔记 {n}", "写条笔记：{n}", "记到本子上：{n}",
   "笔记 {n}", "帮我记一下笔记，{n}", "新笔记：{n}", "把{n}记到笔记里"],
 "fetch_url": [
   "打开{u}", "帮我抓取{u}的内容", "看看这个网页{u}", "访问{u}",
   "把{u}的内容拉下来", "读一下这个链接{u}", "{u} 上写了什么", "抓一下{u}"],
 "chat": CHITCHAT,
 "task": TASKS,
 "confirm": CONFIRMS,
 "deny": DENYS,
}

MEM_SUBJECTS = [
    ("手机型号", ["手机", "手机型号", "用的什么手机", "手机是哪款"],
     ["用户的手机是一加13", "本命机是一加 13，代号 PJZ110", "手机型号一加十三"]),
    ("常驻城市", ["住在哪个城市", "在哪个城市", "家在哪", "坐标"],
     ["用户住在成都", "常驻城市成都", "坐标四川成都"]),
    ("喜欢的语言", ["喜欢的编程语言", "常用什么语言", "写代码用什么"],
     ["用户喜欢用 Python", "最常用中文和 Python", "偏爱 Python 写脚本"]),
    ("项目代号", ["项目代号", "项目叫什么", "代号"],
     ["项目代号是崔共界", "当前项目：崔共界", "代号「崔共界」"]),
    ("早餐", ["早上吃什么", "早餐偏好", "早饭"],
     ["用户早餐爱吃豆浆油条", "早餐：豆浆+油条", "早饭习惯吃清淡的粥"]),
    ("会议时间", ["会议几点", "开会时间", "例会"],
     ["每周三下午三点例会", "例会时间是周三 15:00", "周三 3 点开周会"]),
    ("服务器地址", ["服务器ip", "服务器在哪", "主机地址"],
     ["主服务器 10.0.0.5", "服务器地址 10.0.0.5:22", "内网主机 10.0.0.5"]),
    ("体重目标", ["体重目标", "想瘦多少", "减重目标"],
     ["目标减到 70kg", "体重目标 70 公斤", "打算减重到七十公斤"]),
    ("宠物", ["养了什么宠物", "宠物叫什么", "猫还是狗"],
     ["养了一只橘猫叫年糕", "宠物：橘猫「年糕」", "家里有只猫叫年糕"]),
    ("读书偏好", ["喜欢看什么书", "读书口味", "最近在读什么"],
     ["偏爱科幻小说", "最近在读《三体》", "爱看科幻和历史类"]),
]
FILLER = ["今天天气不错", "股票又跌了", "刚才吃了个苹果", "外面在下雨",
          "楼下在装修", "明天要早起", "这部剧挺好看的", "周末想去爬山",
          "咖啡喝多了睡不着", "公交车好挤"]

MEMVALUES = ["我叫小明", "我住在成都", "我喜欢Python", "我手机是一加13",
             "会议改到周三三点", "服务器是10.0.0.5", "我养了只猫叫年糕",
             "我目标是减到70公斤", "我爱看科幻小说", "项目代号崔共界",
             "我生日是3月5号", "我的邮箱是 a@b.com"]

# ============================================================ 自我评估（第 3 头）
# (文本 = 行动+结果摘要, 标签: 1=成功继续 / 0=失败需重试)
EVAL_OK = [
    "写文件 notes.txt 成功，共 12 字符，已确认存在",
    "命令执行完成，退出码 0，输出共 3 行",
    "python 执行成功，输出 42",
    "记忆已写入，id=7，检索验证命中",
    "文件读取成功，共 5 行内容",
    "脚本运行通过，输出与预期一致",
    "网页抓取成功，HTTP 200，内容 1200 字",
    "搜索命中 3 个文件，均已列出",
    "目录列出成功，共 8 个文件",
    "笔记已保存到 notes/idea.md",
    "代码第一次运行就通过，结果正确",
    "记忆检索命中 1 条，正是要找的内容",
    "工具调用返回正常，无报错",
    "重命名完成，全部文件已更新",
    "合并完成，输出文件 120 行",
]
EVAL_BAD = [
    "写文件失败：路径越界，拒绝访问",
    "命令执行超时，已等 60 秒无响应",
    "python 报错 NameError: name 'x' is not defined",
    "记忆写入失败：数据库被锁",
    "文件不存在：notes.txt (No such file)",
    "脚本运行报错 SyntaxError: invalid syntax",
    "网页抓取失败 HTTP 404",
    "搜索无结果，0 个文件命中",
    "目录不存在，无法列出",
    "笔记保存失败：磁盘只读",
    "代码运行报 TypeError: unsupported operand",
    "记忆检索 0 命中，没有相关内容",
    "工具调用失败：权限被拒绝",
    "重命名失败：目标文件已存在",
    "合并失败：输入文件为空",
    "执行报错：Permission denied",
    "下载中断，只拿到一半数据",
    "JSON 解析失败：Expecting value",
]


def typo(s):
    r = random.random()
    if r < 0.18:
        s = re.sub(r"[，。：,．]", "", s)
    elif r < 0.30:
        s += random.choice(["呢", "吧", "啊", "哈", "谢谢", "麻烦了"])
    elif r < 0.36 and len(s) > 4:
        i = random.randrange(1, len(s) - 1)
        s = s[:i] + s[i] + s[i:]
    return s


def fill(t):
    return (t.replace("{p}", random.choice(PATHS))
             .replace("{c}", random.choice(CONTENTS))
             .replace("{cmd}", random.choice(CMDS))
             .replace("{code}", random.choice(PYCODE))
             .replace("{q}", random.choice(QUERIES))
             .replace("{m}", random.choice(MEMVALUES))
             .replace("{n}", random.choice(NOTES))
             .replace("{u}", random.choice(URLS)))


def gen_intent(n_per=900):
    """每类目标 n_per 条；组合空间不足时用噪声补，绝不死循环。"""
    rows = []
    for yi, name in enumerate(INTENTS):
        tpl = TPL[name]
        # 估算组合空间：无词槽模板 = 模板数，有词槽 = 模板数 × 各槽取值
        space = 0
        for t in tpl:
            k = 1
            for slot, pool in [("{p}", PATHS), ("{c}", CONTENTS), ("{cmd}", CMDS),
                               ("{code}", PYCODE), ("{q}", QUERIES), ("{m}", MEMVALUES),
                               ("{n}", NOTES), ("{u}", URLS)]:
                if slot in t:
                    k *= len(pool)
            space += max(k, 1)
        target = min(n_per, max(space * 3, 40))     # 有噪声加持，空间×3 也够
        seen, tries, cnt = set(), 0, 0
        while cnt < target and tries < target * 40:
            tries += 1
            s = fill(random.choice(tpl))
            if random.random() < 0.45:
                s = typo(s)
            if s in seen and cnt > space * 2:
                continue
            seen.add(s)
            rows.append((yi, s))
            cnt += 1
    random.shuffle(rows)
    return rows


def gen_mem(n_each=3000):
    rows, subs = [], MEM_SUBJECTS
    for _ in range(n_each):
        _, qs, ms = random.choice(subs)
        rows.append((1, random.choice(qs), random.choice(ms)))
    for _ in range(n_each):
        a, b = random.sample(range(len(subs)), 2)
        rows.append((0, random.choice(subs[a][1]), random.choice(subs[b][2])))
    for _ in range(n_each):
        _, qs, _ = random.choice(subs)
        rows.append((0, random.choice(qs), random.choice(FILLER)))
    random.shuffle(rows)
    return rows


def gen_eval(n_each=2000):
    rows = []
    for _ in range(n_each):
        rows.append((1, random.choice(EVAL_OK)))
    for _ in range(n_each):
        rows.append((0, random.choice(EVAL_BAD)))
    random.shuffle(rows)
    return rows


def main():
    ir = gen_intent()
    n = max(1, len(ir) // 10)
    with open(os.path.join(OUT, "intent_test.txt"), "w", encoding="utf-8") as f:
        for y, s in ir[:n]:
            f.write("%d\t%s\n" % (y, s))
    with open(os.path.join(OUT, "intent_train.txt"), "w", encoding="utf-8") as f:
        for y, s in ir[n:]:
            f.write("%d\t%s\n" % (y, s))

    mr = gen_mem()
    n = max(1, len(mr) // 10)
    with open(os.path.join(OUT, "mem_test.txt"), "w", encoding="utf-8") as f:
        for y, q, m in mr[:n]:
            f.write("%d\t%s\t%s\n" % (y, q, m))
    with open(os.path.join(OUT, "mem_train.txt"), "w", encoding="utf-8") as f:
        for y, q, m in mr[n:]:
            f.write("%d\t%s\t%s\n" % (y, q, m))

    er = gen_eval()
    n = max(1, len(er) // 10)
    with open(os.path.join(OUT, "eval_test.txt"), "w", encoding="utf-8") as f:
        for y, s in er[:n]:
            f.write("%d\t%s\n" % (y, s))
    with open(os.path.join(OUT, "eval_train.txt"), "w", encoding="utf-8") as f:
        for y, s in er[n:]:
            f.write("%d\t%s\n" % (y, s))

    print("意图: train %d / test %d" % (len(ir) - n, n))
    print("记忆: train %d / test %d" % (len(mr) - n, n))
    print("评估: train %d / test %d" % (len(er) - n, n))


if __name__ == "__main__":
    main()
