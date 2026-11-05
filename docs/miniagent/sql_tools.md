# `miniagent/sql_tools.py` 详解

## 这个文件干什么

**让 agent 只读地查一个数据库**：先看有哪些表、每张表有哪些列，再按结构化条件取数据。

四个工具：`list_tables` / `describe_table` / `query_table` / `count_rows`。

```python
from miniagent import SQLPolicy, SQLSandbox
from miniagent.sql_tools import make_sql_tools, mysql_connector

box = SQLSandbox(
    mysql_connector(host="...", user="agent", password="...", database="shop"),
    SQLPolicy(allowed_tables=frozenset({"orders", "customers"}), max_rows=50),
)
agent = Agent(backend, [*make_sql_tools(box), calculator])
```

---

## 1. 为什么按 DB-API 2.0 写，而不是直接绑 pymysql

两者都实现 PEP 249，接口一致：`connect()` → `cursor()` → `execute(sql, params)` → `fetchall()`。

这样写换来一件很实际的事：**测试可以跑真正的 SQL**。

```python
with SQLSandbox(sqlite_connector(tmp_path / "shop.db")) as box:
    ...
```

`sqlite3` 是标准库、无需服务器、行为与 pymysql 同构。于是 55 条测试验证的是**真的语句构建器 + 真的驱动 + 真的表**，而不是一个只会附和自己的 mock。

要验证的东西恰好是安全边界——**mock 在这里毫无价值**：它只能证明 mock 和实现意见一致。

代价是 MySQL 特有的 SQL（`information_schema`）在测试里走不到，只能靠手工对真实服务器验一次（见第 6 节）。

---

## 2. 威胁模型和文件工具完全不同

文件工具的危险是**坏路径**；SQL 工具的危险是**坏语句**：

| 攻击 | 后果 |
|---|---|
| `DROP TABLE` / `UPDATE` | 毁数据 |
| `UNION SELECT password FROM users` | 越权读 |
| 笛卡尔积 / `sleep(10)` | 拖垮数据库 |
| `' OR 1=1 --` | 绕过条件 |

四条规则覆盖：

### 规则 1：语句绝不拼接模型文本

表名、列名、运算符是**验证后**的标识符；所有**值**都走绑定参数。

```python
sql = f"SELECT {selected} FROM {quoted_name}{clause.sql} LIMIT {rows}"
return sql, clause.params        # ← 值永远在 params 里，不在 sql 里
```

`query_table` 的接口刻意**不接受 SQL**，只接受结构化条件：

```python
query_table(
    table="orders",
    columns=["customer_id", "amount"],
    where=["status = 'paid'", "amount > 50"],
    order_by="amount desc",
    limit=20,
)
```

模型描述**想要什么**，这个模块决定**怎么问**。

### 规则 2：标识符验证 + 加引号

```python
_IDENT = re.compile(r"^[^\W\d][\w$/-]{0,63}$", re.UNICODE)

def _quote(name: str, dialect: str) -> str:
    return f"`{name}`" if dialect == "mysql" else f'"{name}"'
```

两件事必须同时成立：

* **验**：正则只放行「字母/数字/下划线/`$`/`-`/`/`」，且首字符不能是数字。引号、反引号、空格、`;`、括号、运算符全部出局。
* **引**：验完还要加引号。MySQL 里 `daa05-cy2` 不加反引号是**语法错误**（`-` 是减号）。

**先验后引**，所以 `"` 和 `` ` `` 不可能出现在名字里 → 引号无法被闭合 → 无法逃逸。

> 为什么用 `[^\W\d]` 而不是 `[A-Za-z_]`：真实库里有 `长垣单井基础信息表`、`井间动态_cy1`、`分注井月数据表`，还有列名 `累积产液量/t`。ASCII 正则会把这些**全部拒绝**——这是我第一次连真实库时踩到的坑。

### 规则 3：只读在会话层也被强制

```python
def _enforce_read_only(conn) -> str:
    if dialect == "sqlite":
        cursor.execute("PRAGMA query_only = 1")
        return "sqlite query_only"
    cursor.execute("SET SESSION TRANSACTION READ ONLY")
    cursor.execute("SELECT @@transaction_read_only")
    return "READ ONLY" if int(cursor.fetchone()[0]) == 1 else "NOT read-only"
```

这是**双保险**：即便将来有人加了一条写入路径，或者某条语句绕过了构建器，**服务端自己会拒绝**。

测试里真的验了这件事：

```python
def test_the_server_refuses_a_write_even_if_one_somehow_got_through(box):
    box.connection()  # 应用 PRAGMA query_only
    with pytest.raises(sqlite3.OperationalError, match="readonly database"):
        box.execute("INSERT INTO users (name) VALUES ('mallory')", [])
```

真实 MySQL 上验证的结果：

```
read-only state      : READ ONLY
@@transaction_read_only: 1
```

### 规则 4：真正的边界是数据库账号

上面三条防的是**意外**和**提示注入**。防不住的是「有人刻意设计」。那个要靠权限：

```sql
CREATE USER 'agent'@'%' IDENTIFIED BY '...';
GRANT SELECT ON shop.* TO 'agent'@'%';
```

**用只读账号，而不是用 root。**

---

## 3. 条件语法：给模型一个能猜对的形状

`where` 是 `list[str]`，每项形如 `column op value`。解析在 `_FILTER`：

```python
"status = 'paid'"      → status = ?            参数 'paid'
"amount > 100"         → amount > ?            参数 100（int，不是 "100"）
"name like 'A%'"       → name LIKE ?           参数 'A%'
"id in (1, 2, 3)"      → id IN (?, ?, ?)       参数 1, 2, 3
"note is null"         → note IS NULL          无参数
```

支持：`= != <> < <= > >=`、`like` / `not like`、`in` / `not in`、`is null` / `is not null`。多个条件用 `AND` 连接。

值类型推断在 `_literal`：

```python
"100"      → 100        # 裸数字转数值，才能数值比较
"'100'"    → "100"      # 引号包住就是字符串
"null"     → None
"true"     → True
```

**为什么裸数字要转 int**：`amount > '100'` 在 MySQL 里是字符串比较（`'99' > '100'` 为真），结果会莫名多出一堆行。这是真实会踩的坑。

### 一个我一开始写错的测试

我曾把 `name = 'x' OR 1=1` 放进「应该被拒绝」的参数化列表里。**它不该被拒绝——它根本构不成注入。**

`_FILTER` 把它解析为：列 `name`、运算符 `=`、值 `'x' OR 1=1`。值是**绑定参数**，所以 `OR 1=1` 只是一段普通文本，匹配不到任何行：

```python
sql, params = box.build_select("users", None, ["name = 'x' OR 1=1"], "", 10)
# SELECT * FROM "users" WHERE "name" = ? LIMIT 10
# params = ["x' OR 1=1"]        ← 就在参数里
```

断言「它应该抛异常」等于断言了错的东西：**重点不是拒绝，而是无法执行**。现在的测试叫 `test_sql_lookalikes_inside_a_value_are_harmless`，验证的是「一个占位符 + 零行结果」。

### 静默比报错危险：两种「0 行」

真正的问题出在 0.5B 跑真实问题时。它给出的调用是：

```python
query_table({"table": "dba04_cy4", "columns": ["HS"],
             "where": ["HS > 25 AND HS < 28"], "order_by": "ASC"})
```

两个错误，一个显眼、一个隐蔽：

* `order_by="ASC"` —— 把**方向**当成了列名；
* `where=["HS > 25 AND HS < 28"]` —— 两个条件被塞进**一个**元素。

工具当时照单全收，生成：

```sql
SELECT "HS" FROM "dba04_cy4" WHERE "HS" > ? ORDER BY "ASC" ASC LIMIT 20
-- params = ['25 AND HS < 28']
```

sqlite 把 `"ASC"` 当成**字符串常量**来排序（等于不排序），把 `'25 AND HS < 28'` 当成普通文本去和数字比 → **0 行**。模型看到「0 行」，于是认真地回答「没有符合条件的记录」。

**这是最坏的一种错：0 行是一个合法答案。** 没有任何东西提示你去复核。

MySQL 在同样两个地方会直接报错，sqlite 不会。所以这项检查只能由我们自己承担——这就是 `_PACKED_CONDITION` 和 `_RESERVED_WORDS` 存在的理由。

#### 拒绝要窄，否则会拒掉合法查询

第一版正则写成 `\b(and|or|not|select|union|from|where|like|between|is|in)\b|[<>=!()]`，结果几乎一切值都被拒——包括上面那个合法的 `name = 'x' OR 1=1`，也就是把刚讲对的道理又推翻了一遍。

区分「打包了两个条件」和「值里恰好像 SQL」的关键是：**`and`/`or` 后面跟的是不是另一个列名**。

```python
_PACKED_CONDITION = re.compile(
    r"\b(?:and|or)\b\s+[^\W\d]\w*\s*(?:[<>=!]|\bnot\b|\bis\b|\blike\b|\bin\b|\bbetween\b)",
    re.IGNORECASE,
)
```

* `25 AND HS < 28` → `AND HS <` 命中，拒绝；
* `'x' OR 1=1` → `or` 后面是 `1`，不是标识符，放过（它本来就无害）。

另外 `_is_quoted` 放行整体被引号包住的值：`name = 'a AND b < c'` 是在**搜索这段文字**，不是在写条件，必须能过。

同时 `_RESERVED_WORDS` 让 `_check_table` / `_check_column` 直接拒绝 `asc`、`from` 这类词；`_order` 对纯方向词给一句专用提示（“order_by 'ASC' is a direction, not a column. Name the column first: 'NY desc'”），而不是含糊的「读不懂」。

#### 错误观察里要带一句「可以重试」

`ToolRegistry.execute` 的失败回执现在长这样：

```text
ERROR: SQLPolicyError: condition 'HS > 25 AND HS < 28' packs more than one condition into the value. ...
Nothing was changed. Call query_table again with corrected arguments.
```

第二行是给弱模型看的。只给错误，0.5B 会道歉然后收工；说了「什么都没改，可以重来」，它至少会再试一次。

#### 结果：0.5B 仍然不会自我纠正

实测（`--backend local`）三种状态：

| 版本 | 模型行为 | 你看到的结果 |
|---|---|---|
| 修复前 | 调用被接受 | 「0 行」——**自信的错误答案** |
| 加拒绝 | 一次失败 → 道歉收工 | `no answer`，循环退出码 1 |
| 再加提示 | 三次重复同一调用 → 被 `loop_detected` 停下 | `no answer: loop_detected, and no statement ever ran` |

`order_by` 在加了 docstring 提示后修对了，`where` 依然修不对——「把『大于 25 小于 28』拆成两个条件」超出一个 0.5B 模型的能力，不是提示词问题。

**护栏的价值不在于让弱模型变强，而在于不让它用一个看起来完全正常的答案骗过你。** 换成 7B 以上、或给一个 few-shot 例子，才谈得上自我纠正。

---

## 4. 有界的代价

三个配额，都在 `SQLPolicy`：

| 字段 | 默认 | 作用 |
|---|---|---|
| `max_rows` | 100 | `LIMIT` 上限（模型传 999 也会被压下来） |
| `max_columns` | 40 | 一次最多几列 |
| `max_cell_chars` | 120 | 单格截断（超长文本、BLOB） |
| `max_output_chars` | 6000 | 整段结果截断 |
| `allowed_tables` | `None` | `None` = 用户可见的全部；集合 = 白名单 |
| `denied_tables` | 空 | **优先于**白名单 |
| `allow_raw_select` | `False` | 手写 SQL 开关 |
| `force_read_only_session` | `True` | 会话层只读 |

`allow_raw_select=True` 时仍然有一道 `check_raw`：必须以 `SELECT`/`WITH` 开头、单语句、命中关键字黑名单（`insert/update/drop/...`、`information_schema`、`sleep(`、`load_file`）、没写 `LIMIT` 就自动补。

---

## 5. 输出长什么样

```
5 row(s) from 长垣单井基础信息表 (limit 5)
井号        | 厂名    | 矿名    | 区块名称      | 原始地层压力
----------+-------+-------+-----------+-------
北1-4-丙114 | 第一采油厂 | 第一作业区 | 西部过渡带北西老井 | 11.89
北1-4-丙115 | 第一采油厂 | 第一作业区 | 西部过渡带北西老井 | 11.68
```

第一行是「取了几行 + 从哪张表 + limit」，这样模型能判断结果是否被截断。`NULL` 显式打印（不是空串），二进制显示成 `<N bytes>`。

### 截断必须说出来：哨兵行技巧

「取了几行」写在这里还不够——模型看到 `1 row(s)` 时没法知道**真实有几行**。0.5B 实测就是这个结果：`--max-rows 1` 下它把 1 行当成全部答案报了出去，而库里其实匹配 2 行。

所以 `query_table` 会**多取一行**（`build_select(..., probe=True)` → `LIMIT capped + 1`），多出来的那行只用来判断「还有没有」，绝不返回给模型：

```python
truncated = len(rows) > capped
rows = rows[:capped]
```

一旦命中，回执第一行变成：

```text
1 row(s) from dba04_cy4 (limit 1; MORE ROWS EXIST -- this is a partial result,
call count_rows or narrow the filters)
```

审计日志里因此会看到 `LIMIT 2`（1 行展示 + 1 行哨兵），这不是 bug。

代价说明：**多取一行仍然要付磁盘读取的钱**，但比起「再发一条 `COUNT(*)`」便宜得多，也不需要第二条语句的往返。这是 SQL 分页里的标准做法（`LIMIT n+1`）。

> 实测补充：信号给出去了，0.5B **照样**把 1 行当成全部答案。工具能做的到此为止——它保证了「信息没被隐瞒」，但「读了信息要不要行动」是模型的事。

`box.queries` 是**审计日志**——本次运行的每一条语句：

```
SELECT `井号`, `厂名` FROM `长垣单井基础信息表` WHERE `厂名` = %s LIMIT 3
```

看到 `%s` 就知道值是绑定的，看到反引号就知道标识符被引住了。

---

## 6. 手工验真实 MySQL（测试覆盖不到的部分）

> 参考服务器（`data` 库 59 张表，`dba04_cy4` 有 217 万行）**只能从内网访问**。
> 不在公司网络时，`--mysql` 会在建立连接那一步就退出：
>
> ```
> cannot reach mysql root@100.81.78.48:3306/data: OperationalError: ...
> That server is only reachable from the intranet.  Drop --mysql to use the
> offline replica of dba04_cy4 (same 53 columns, fabricated rows)
> ```
>
> 离线副本（`agent_workspace/sqldemo.db`）就是为这个场景准备的：**列名和列类型与
> 真表完全一致（53 列）**，只有数据是编的（14 行）。`describe_table` 看到的形状、
> `SELECT *` 的宽度、`HS` 是 `REAL/DOUBLE` 而不是 text —— 全部和真库一样，所以
> sqlite 上验过的行为对 MySQL 也成立，只有 `information_schema` 与反引号这两条
> 路径需要上内网再验。

```powershell
uv pip install --python .venv\Scripts\python.exe pymysql

# 凭据放文件，不要写进命令行（会进 shell 历史）
python scripts\agent_sql_demo.py --mysql --config-json D:\secrets\mysql.json --explore
```

`--explore` 不走模型，直接倒 schema：

```
database : mysql root@10.0.0.5:3306/data
read-only: READ ONLY
59 table(s): daa01, daa05-cy2, ..., 长垣单井基础信息表
-- 长垣单井基础信息表 (33 columns)
井号  varchar(255)
...
```

**没装 pymysql 时**报错是可操作的，不是 `ModuleNotFoundError`：

```
RuntimeError: pymysql is not installed.
Run: uv pip install --python .venv\Scripts\python.exe pymysql
```

---

## 7. 实测记录：从真实库学到的三件事

1. **中文标识符必须支持**。第一次连上去时 59 张表里有一半被 ASCII 正则拒绝。
2. **带连字符的表名必须加引号**。`daa05-cy2` 是真实存在的表；`SELECT * FROM daa05-cy2` 语法错误。
3. **很多列是 `text`**。`不稳定试井解释成果`、`分注井月数据表` 的全部列都是 text——导入时没做类型转换。所以 `amount > 100` 这类条件在那些表上可能走字符串比较。**先 `describe_table` 再查**，这个工具顺序不是可选的。

---

## 用到的 Python / sqlite / pymysql 特性

| 特性 | 说明 |
|---|---|
| PEP 249 `paramstyle` | `qmark`（sqlite `?`）vs `pyformat`（pymysql `%s`） |
| `cursor.description` | 列名（`[d[0] for d in ...]`） |
| `PRAGMA query_only = 1` | sqlite 的会话只读 |
| `SET SESSION TRANSACTION READ ONLY` | MySQL 的会话只读 |
| `@@transaction_read_only` | 验证是否生效 |
| `sqlite_master` | sqlite 的表清单 |
| `information_schema.tables/columns` | MySQL 的表/列清单 |
| `re.UNICODE` + `\w` | 匹配中文标识符 |
| `pymysql.connect(..., autocommit=True)` | 每条语句自成一个事务，只读设置才生效 |

---

## 设计取舍

**为什么不给 `query_table` 加 `sql` 参数？**

那会把整个安全设计作废。要手写 SQL 就显式开 `allow_raw_select`，且只能走 `check_raw`。**默认路径必须是安全的那条。**

**为什么 `count_rows` 单独一个工具？**

「有多少行」和「取几行看看」是两种意图。合成一个工具，模型会为了计数拉回一堆数据；分开之后它可以先 `count_rows` 判断要不要继续查。

**为什么 `_enforce_read_only` 失败不抛异常？**

受限账号可能没有 `SET SESSION` 权限，但它本身就是只读的。这时中断反而让工具不可用。状态记在 `read_only_state` 里，需要严格的场景自己断言它等于 `"READ ONLY"`。

### 已知缺口：两个声明了却没接上的参数

`SQLSandbox.__init__` 里的 `reuse_connection` 和 `statement_timeout_s` **被保存下来，但从没被读过**：

| 参数 | 期望语义 | 实际行为 |
|---|---|---|
| `reuse_connection=True` | `False` 时每次查询新建连接 | `connection()` 只判断 `_conn is not None`，永远是复用 |
| `statement_timeout_s=None` | 超时中断查询 | 没有任何地方引用 |

传它们不会报错，也不会有任何效果——**静默无效**。这和本文件反复强调的原则正好相反：一个转了没反应的旋钮，比没有这个旋钮更糟，因为使用者会以为自己设了保护。

目前刻意保留（不删也不实现），当作一个反面素材：读代码时看到"参数存了但没人用"，就该把它当成 bug 而不是功能。真要补的话，`statement_timeout_s` 在 MySQL 上走 `SET SESSION MAX_EXECUTION_TIME=ms`，sqlite 上得用 `set_progress_handler` 主动中断，两边语义并不对称——这也是它当初被搁置的原因。

---

## 常见误解

* 「`allowed_tables=None` 等于不限制」——等于「用户能看见的都行」。真限制靠 MySQL 授权。
* 「加了引号就安全」——引号**加上**验证才安全。只加引号、不禁止名字里的引号，一样能被闭合。
* 「sqlite 测过就等于 MySQL 也对」——语句构建逻辑一样，但 `information_schema` 路径只能在真服务器上验。
* 「`text` 列也能数值比较」——MySQL 会隐式转换，但 `'99' > '100'` 为真。查之前先看类型。
* 「`box.queries` 是调试用的」——它同时是审计记录，出问题第一眼看这个。
* 「工具应该尽量宽容，能把模型的意思猜出来」——`where=["HS > 25 AND HS < 28"]` 猜不出来只能猜错。**猜错得到的 0 行比报错危险得多**：0 行是合法答案，你会当成结论。
* 「加了护栏，弱模型就能自救了」——不会。护栏只保证**你不会被假答案骗**，自纠正能力是模型本身的事。

---

## 动手练习

1. 把 `_IDENT` 改回 `[A-Za-z_][A-Za-z0-9_$]*`，跑 `test_identifiers_that_need_quoting_are_quoted`，看中文表名怎么被拒。
2. 去掉 `_quote`（直接拼名字），跑同一个测试，看 `daa05-cy2` 那类名字如何变成语法错误。
3. 把 `force_read_only_session` 设成 `False`，跑 `test_the_server_refuses_a_write_even_if_one_somehow_got_through`，确认它失败——**这就是「只有构建器保护」和「再加一层服务端保护」的区别**。
4. 给 `SQLPolicy` 加 `max_seconds`，用 MySQL 的 `SET SESSION MAX_EXECUTION_TIME` 实现查询超时，然后想清楚 sqlite 上该怎么模拟。
5. **把 `_PACKED_CONDITION` 删掉，跑 `--backend local`，看模型怎样用一个「0 行」的答案骗过你。** 再把它加回来对比。
6. 试着自己修 0.5B：加一条 few-shot 例子进 `DOMAIN_NOTES`，看它能不能一次写对。写不对就退一步想——真上生产，这个活该交给模型还是交给工具？
