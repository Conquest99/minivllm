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

`box.queries` 是**审计日志**——本次运行的每一条语句：

```
SELECT `井号`, `厂名` FROM `长垣单井基础信息表` WHERE `厂名` = %s LIMIT 3
```

看到 `%s` 就知道值是绑定的，看到反引号就知道标识符被引住了。

---

## 6. 手工验真实 MySQL（测试覆盖不到的部分）

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

---

## 常见误解

* 「`allowed_tables=None` 等于不限制」——等于「用户能看见的都行」。真限制靠 MySQL 授权。
* 「加了引号就安全」——引号**加上**验证才安全。只加引号、不禁止名字里的引号，一样能被闭合。
* 「sqlite 测过就等于 MySQL 也对」——语句构建逻辑一样，但 `information_schema` 路径只能在真服务器上验。
* 「`text` 列也能数值比较」——MySQL 会隐式转换，但 `'99' > '100'` 为真。查之前先看类型。
* 「`box.queries` 是调试用的」——它同时是审计记录，出问题第一眼看这个。

---

## 动手练习

1. 把 `_IDENT` 改回 `[A-Za-z_][A-Za-z0-9_$]*`，跑 `test_identifiers_that_need_quoting_are_quoted`，看中文表名怎么被拒。
2. 去掉 `_quote`（直接拼名字），跑同一个测试，看 `daa05-cy2` 那类名字如何变成语法错误。
3. 把 `force_read_only_session` 设成 `False`，跑 `test_the_server_refuses_a_write_even_if_one_somehow_got_through`，确认它失败——**这就是「只有构建器保护」和「再加一层服务端保护」的区别**。
4. 给 `SQLPolicy` 加 `max_seconds`，用 MySQL 的 `SET SESSION MAX_EXECUTION_TIME` 实现查询超时，然后想清楚 sqlite 上该怎么模拟。
