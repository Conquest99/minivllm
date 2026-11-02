"""只读 SQL 工具：看结构、查一张表。

按 **DB-API 2.0**（PEP 249）写，而不是绑死某个驱动，所以同一份代码能跑在 MySQL
（``pymysql``）和 ``sqlite3`` 上。这不是为了泛化而泛化：它意味着下面这些安全规则
是被**真实 SQL** 验证的，而不是被一个只能和自己意见一致的 mock 验证。

威胁模型和文件工具完全不同。文件工具的危险是一个坏*路径*；SQL 工具的危险是一条坏
*语句* —— ``DROP TABLE``、``UNION SELECT password FROM users``、把 CPU 钉死的大
笛卡尔积。四条规则覆盖它：

1. **绝不用模型文本拼语句。**  表名、列名、运算符都先过严格标识符正则再对白名单；
   每个*值*都以绑定参数出行。任何地方都没有用户输入的字符串拼接，所以引号技巧
   不是「防住了」，而是根本不会发生。
2. **结构上就是只读。**  本模块能发出的语句只有 ``SELECT`` / ``SHOW`` / ``PRAGMA``。
   手写 SQL 默认关闭；即便打开，也要过关键字黑名单和单语句检查。
3. **有界成本。**  每条查询都带 ``LIMIT`` 并夹到 ``max_rows``；结果集在到达模型
   之前还会按单元格宽度和总大小截断。
4. **真正的边界是数据库账号。**  给 agent 一个只对某个 schema 有 ``SELECT`` 的
   MySQL 账号，别的什么都不给。上面这些规则挡的是误伤和提示注入；能挡住处心积虑
   的，只有权限。
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from miniagent.tools import Tool, tool

_IDENT = re.compile(r"^[^\W\d][\w$/-]{0,63}$", re.UNICODE)
"""A bare identifier that may be quoted safely.

Deliberately Unicode-aware: real schemas contain ``长垣单井基础信息表``, ``daa05-cy2``
and even column names like ``累积产液量/t``.  What it excludes is what could escape
a quoted identifier -- quotes, backticks, whitespace, ``;``, parentheses, operators
-- so the name is always quoted *after* validation and can never break out.
"""

_QUOTE_FORBIDDEN = ("`", '"', "'", ";", "(", ")", " ", "\t", "\n", "\r")


def _quote(name: str, dialect: str) -> str:
    """把校验过的标识符包上该方言的引号字符。

    引号是必需的，不是装饰：MySQL 对 ``daa05-cy2`` 必须用反引号（``-`` 是减号运算符），
    sqlite 要用双引号。安全的前提是 :data:`_IDENT` 已经拒绝了所有引号字符。
    """
    return f"`{name}`" if dialect == "mysql" else f'"{name}"'


_OPERATORS: dict[str, str] = {
    "=": "=",
    "!=": "!=",
    "<>": "!=",
    "<": "<",
    "<=": "<=",
    ">": ">",
    ">=": ">=",
    "like": "LIKE",
    "not like": "NOT LIKE",
    "in": "IN",
    "not in": "NOT IN",
    "is null": "IS NULL",
    "is not null": "IS NOT NULL",
}
_NEEDS_VALUE = {k for k in _OPERATORS if k not in ("is null", "is not null")}
_NEEDS_LIST = {"in", "not in"}

_FILTER = re.compile(
    r"^\s*(?P<col>[^\W\d][\w$/-]{0,63})\s*"
    r"(?P<op><=|>=|!=|<>|=|<|>|is\s+not\s+null|is\s+null|not\s+like|not\s+in|like|in)"
    r"\s*(?P<val>.*?)\s*$",
    re.IGNORECASE | re.DOTALL | re.UNICODE,
)
_ORDER = re.compile(
    r"^\s*(?P<col>[^\W\d][\w$/-]{0,63})(?:\s+(?P<dir>asc|desc))?\s*$",
    re.IGNORECASE | re.UNICODE,
)
_DENIED_KEYWORDS = (
    "insert",
    "update",
    "delete",
    "drop",
    "alter",
    "create",
    "truncate",
    "grant",
    "revoke",
    "into outfile",
    "load_file",
    "information_schema",
    "sleep(",
    "benchmark(",
)

_RESERVED_WORDS = frozenset(
    """
    asc desc order group having limit offset select from where join on using
    and or not null true false union all distinct insert update delete drop create
    alter truncate table database index into values set case when then else end
    between like is in exists
    """.split()
)
"""Words that are never a column or table name.

Refusing them turns a *silent* dialect quirk into a readable error: sqlite treats an
unknown identifier in ``ORDER BY`` / ``WHERE`` as the string literal of that name, so
``ORDER BY "ASC"`` silently sorts by a constant and ``"HS" > "25 AND ..."`` compares
against text.  MySQL would raise here; sqlite will not, so the check has to be ours.
"""

_PACKED_CONDITION = re.compile(
    r"\b(?:and|or)\b\s+[^\W\d]\w*\s*(?:[<>=!]|\bnot\b|\bis\b|\blike\b|\bin\b|\bbetween\b)",
    re.IGNORECASE,
)
"""A *value* shaped like ``<another column> <operator>`` means two conditions got packed.

``where=["HS > 25 AND HS < 28"]`` is the exact mistake a small model makes: it folds
two conditions into one list element.  Bound as a parameter it silently matches
nothing, which is worse than an error -- "0 rows" looks like a real answer.

Deliberately narrow: ``name = 'x' OR 1=1`` is *not* flagged, because a value is only
ever bound, never concatenated, and searching for that literal text is legitimate.
The pattern requires a real identifier after ``and``/``or`` to fire.
"""


class SQLPolicyError(ValueError):
    """被策略拒绝的请求。消息是写给模型看、让它照着改的。"""


@dataclass(frozen=True)
class SQLPolicy:
    allowed_tables: frozenset[str] | None = None
    """``None`` = any table the database user can see. A set = an explicit allowlist."""
    denied_tables: frozenset[str] = frozenset()
    max_rows: int = 100
    max_columns: int = 40
    max_cell_chars: int = 120
    max_output_chars: int = 6000
    allow_raw_select: bool = False
    """Off by default: raw SQL is the one place injection can still happen."""
    force_read_only_session: bool = True
    """Ask the *server* to refuse writes for this session.

    MySQL: ``SET SESSION TRANSACTION READ ONLY``; sqlite: ``PRAGMA query_only = 1``.
    Belt and braces on top of "we only ever build SELECT": if a future edit grows a
    write path, or a statement slips through the builder, the server still says no.
    """

    def __post_init__(self) -> None:
        for name in ("max_rows", "max_columns", "max_cell_chars", "max_output_chars"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")


def _dialect(conn: Any) -> str:
    """``sqlite`` 或 ``mysql``，从连接的驱动模块推断出来。"""
    module = getattr(type(conn), "__module__", "").split(".")[0]
    return "sqlite" if module == "sqlite3" else "mysql"


def _paramstyle(conn: Any) -> str:
    """PEP 249 把 ``paramstyle`` 声明在*模块*上；读它，并带好兜底。"""
    module = getattr(type(conn), "__module__", "").split(".")[0]
    declared = getattr(sys.modules.get(module), "paramstyle", None)
    if declared in ("qmark", "pyformat", "format", "numeric", "named"):
        return declared
    return "qmark" if module == "sqlite3" else "pyformat"


def _placeholder(style: str, index: int) -> str:
    if style == "qmark":
        return "?"
    if style == "numeric":
        return f":{index}"
    return "%s"  # pyformat / format


def _is_quoted(raw: str) -> bool:
    """值被一对匹配的引号包住时为真。

    被引号包住的值可以包含任何东西 —— ``'x'' OR 1=1 --'`` 是一个合法的搜索目标，反正
    它是当参数绑定的。只有*没有*引号、里面却带 SQL 语法的文本，才是值得拒绝的错误。
    """
    text = raw.strip()
    return len(text) >= 2 and text[0] == text[-1] and text[0] in "'\""


def _literal(raw: str) -> Any:
    """把运算符后面的文本变成 Python 值。

    带引号的保持文本；裸数字变成数值，于是 ``amount > 100`` 是按数值比较而不是按字典序。
    """
    text = raw.strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in "'\"":
        return text[1:-1]
    lowered = text.lower()
    if lowered in ("null", "none"):
        return None
    if lowered in ("true", "false"):
        return lowered == "true"
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        return text


def _split_list(raw: str) -> list[Any]:
    """把 ``(a, 'b', 3)`` 解析成一组值，并尊重引号。"""
    text = raw.strip()
    if text.startswith("(") and text.endswith(")"):
        text = text[1:-1]
    if not text.strip():
        raise SQLPolicyError("IN needs at least one value, for example: id in (1, 2)")
    values, buf, quote = [], [], None
    for ch in text:
        if quote:
            if ch == quote:
                quote = None
            else:
                buf.append(ch)
        elif ch in "'\"":
            quote = ch
        elif ch == ",":
            values.append("".join(buf))
            buf = []
        else:
            buf.append(ch)
    values.append("".join(buf))
    return [_literal(v) for v in values]


@dataclass
class _Clause:
    sql: str
    params: list[Any]


class SQLSandbox:
    """持有连接、策略和语句构建器。

    ``connect`` 是一个无参工厂，好让调用方自己决定怎么到达数据库（也让测试能塞进来一个
    内存 sqlite 连接）。
    """

    def __init__(
        self,
        connect: Callable[[], Any],
        policy: SQLPolicy | None = None,
        *,
        # TODO(未接线): 两者都只被存下来，之后再也没有被读过 —— 传它们
        # 什么也不会发生。一个转了没反应的旋钮比没有旋钮更糟；
        # see docs/miniagent/sql_tools.md ("已知缺口").
        reuse_connection: bool = True,
        statement_timeout_s: float | None = None,
    ) -> None:
        self._connect = connect
        self.policy = policy or SQLPolicy()
        self.reuse_connection = reuse_connection
        self.statement_timeout_s = statement_timeout_s
        self._conn: Any = None
        self.queries: list[str] = []
        """Every statement actually executed -- the audit log for a run."""
        self.read_only_state: str = "not applied"
        """What the server said when asked to go read-only."""

    # ------------------------------------------------------------ 连接
    def connection(self) -> Any:
        if self._conn is not None:
            return self._conn
        self._conn = self._connect()
        if self.policy.force_read_only_session:
            self.read_only_state = self._enforce_read_only(self._conn)
        return self._conn

    @staticmethod
    def _enforce_read_only(conn: Any) -> str:
        """让会话自身拒绝写入，并如实报告结果。

        失败是刻意*不*致命的：受限账号可能没有设置它的权限，但它凭授权本来就是只读的。状态
        字符串会出现在统计里，需要严格保证的调用方自己去断言它。
        """
        dialect = _dialect(conn)
        cursor = conn.cursor()
        try:
            if dialect == "sqlite":
                cursor.execute("PRAGMA query_only = 1")
                return "sqlite query_only"
            cursor.execute("SET SESSION TRANSACTION READ ONLY")
            cursor.execute("SELECT @@transaction_read_only")
            row = cursor.fetchone()
            return "READ ONLY" if row and int(row[0]) == 1 else "NOT read-only"
        except Exception as exc:  # pragma: no cover - depends on server privileges
            return f"unavailable ({type(exc).__name__})"
        finally:
            cursor.close()

    def close(self) -> None:
        if self._conn is not None:
            try:
                self._conn.close()
            finally:
                self._conn = None

    def __enter__(self) -> SQLSandbox:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -------------------------------------------------------------- 语句构建
    def _check_table(self, table: str) -> str:
        name = table.strip()
        if not _IDENT.match(name):
            raise SQLPolicyError(
                f"{table!r} is not a valid table name. Use a bare name like 'users' or "
                "'长垣单井基础信息表' (letters, digits, underscore, hyphen, slash); "
                "no dots, quotes, spaces or semicolons."
            )
        if name.lower() in _RESERVED_WORDS:
            raise SQLPolicyError(
                f"{table!r} is a SQL keyword, not a table name. Pass the actual table, "
                "for example 'dba04_cy4'."
            )
        if name in self.policy.denied_tables:
            raise SQLPolicyError(f"table {name!r} is blocked by policy")
        allowed = self.policy.allowed_tables
        if allowed is not None and name not in allowed:
            raise SQLPolicyError(
                f"table {name!r} is not in the allowlist; allowed: {', '.join(sorted(allowed)) or '(none)'}"
            )
        return name

    def _check_column(self, column: str) -> str:
        name = column.strip()
        if not _IDENT.match(name):
            raise SQLPolicyError(f"{column!r} is not a valid column name")
        if name.lower() in _RESERVED_WORDS:
            raise SQLPolicyError(
                f"{column!r} is a SQL keyword, not a column name. "
                "Pass a real column name from describe_table."
            )
        return name

    def _clause(self, filters: list[str], style: str, dialect: str) -> _Clause:
        parts: list[str] = []
        params: list[Any] = []
        for item in filters:
            match = _FILTER.match(item)
            if not match:
                raise SQLPolicyError(
                    f"cannot read the condition {item!r}. Use 'column op value', for example "
                    '"status = \'active\'", "amount > 100", "name like \'A%\'", '
                    '"id in (1, 2, 3)" or "deleted_at is null".'
                )
            column = _quote(self._check_column(match.group("col")), dialect)
            op_key = " ".join(match.group("op").lower().split())
            sql_op = _OPERATORS[op_key]
            raw = match.group("val")

            if op_key not in _NEEDS_VALUE:
                parts.append(f"{column} {sql_op}")
                continue

            if op_key in _NEEDS_LIST:
                values = _split_list(raw)
                marks = ", ".join(_placeholder(style, len(params) + i + 1) for i in range(len(values)))
                parts.append(f"{column} {sql_op} ({marks})")
                params.extend(values)
                continue

            if not raw.strip():
                raise SQLPolicyError(
                    f"condition {item!r} has no value. Write the value after the operator, "
                    'for example "name = \'alice\'"; use "name is null" to match NULLs.'
                )
            if _PACKED_CONDITION.search(raw) and not _is_quoted(raw):
                raise SQLPolicyError(
                    f"condition {item!r} packs more than one condition into the value. "
                    "Each list element is ONE condition: pass ['HS > 25', 'HS < 28'], "
                    "not ['HS > 25 AND HS < 28']."
                )
            parts.append(f"{column} {sql_op} {_placeholder(style, len(params) + 1)}")
            params.append(_literal(raw))

        where = f" WHERE {' AND '.join(parts)}" if parts else ""
        return _Clause(where, params)

    def _order(self, order_by: str, dialect: str) -> str:
        text = (order_by or "").strip()
        if not text:
            return ""
        if text.lower() in ("asc", "desc", "ascending", "descending"):
            raise SQLPolicyError(
                f"order_by {order_by!r} is a direction, not a column. Name the column "
                "first: 'NY desc', or omit order_by entirely."
            )
        match = _ORDER.match(text)
        if not match:
            raise SQLPolicyError(f"cannot read order_by {order_by!r}. Use 'column' or 'column desc'.")
        column = _quote(self._check_column(match.group("col")), dialect)
        direction = (match.group("dir") or "asc").upper()
        return f" ORDER BY {column} {direction}"

    def build_select(
        self,
        table: str,
        columns: list[str] | None,
        filters: list[str],
        order_by: str,
        limit: int,
        probe: bool = False,
    ) -> tuple[str, list[Any]]:
        """返回 ``(sql, params)``。暴露出来是为了让测试能直接断言 SQL 本身。

        ``probe`` 会比调用方打算展示的行数*多取一行*。这多出来的一行是哨兵：它回来了，就说明
        真实行数比调用方看到的多。一个沉默的 ``LIMIT`` 正是「匹配 2 行，这里给你 1 行」变成
        自信错误答案的方式 —— 模型没法知道自己没被展示的那部分。
        """
        name = self._check_table(table)
        dialect = _dialect(self.connection())
        style = _paramstyle(self.connection())

        selected = "*"
        if columns:
            if len(columns) > self.policy.max_columns:
                raise SQLPolicyError(f"at most {self.policy.max_columns} columns per query")
            selected = ", ".join(_quote(self._check_column(c), dialect) for c in columns)

        clause = self._clause(list(filters), style, dialect)
        rows = max(1, min(int(limit), self.policy.max_rows))
        if probe:
            rows += 1
        sql = (
            f"SELECT {selected} FROM {_quote(name, dialect)}"
            f"{clause.sql}{self._order(order_by, dialect)} LIMIT {rows}"
        )
        return sql, clause.params

    # ------------------------------------------------------------- 执行
    def execute(self, sql: str, params: list[Any]) -> tuple[list[str], list[tuple]]:
        conn = self.connection()
        self.queries.append(sql)
        cursor = conn.cursor()
        try:
            cursor.execute(sql, params)
            columns = [d[0] for d in (cursor.description or ())]
            rows = cursor.fetchall()
        finally:
            cursor.close()
        return columns, rows

    def check_raw(self, sql: str) -> str:
        """校验一条手写的 SELECT。只有策略允许时才走得到这里。"""
        if not self.policy.allow_raw_select:
            raise SQLPolicyError(
                "raw SQL is disabled; use query_table(table, columns, where, order_by, limit)"
            )
        text = sql.strip().rstrip(";").strip()
        if ";" in text:
            raise SQLPolicyError("only one statement per call")
        lowered = text.lower()
        if not lowered.startswith(("select", "with")):
            raise SQLPolicyError("raw SQL must start with SELECT")
        for word in _DENIED_KEYWORDS:
            if word in lowered:
                raise SQLPolicyError(f"raw SQL may not contain {word!r}")
        if " limit " not in lowered:
            text = f"{text} LIMIT {self.policy.max_rows}"
        return text


def _format_table(columns: list[str], rows: list[tuple], policy: SQLPolicy) -> str:
    """把行渲染成定宽表格，裁剪到能塞进模型的上下文。"""
    if not columns:
        return "(no rows)"

    def cell(value: Any) -> str:
        if value is None:
            text = "NULL"
        elif isinstance(value, bytes):
            text = f"<{len(value)} bytes>"
        else:
            text = str(value)
        text = text.replace("\r", " ").replace("\n", " ").replace("\t", " ")
        limit = policy.max_cell_chars
        return text if len(text) <= limit else text[: limit - 1] + "…"

    header = [cell(c) for c in columns[: policy.max_columns]]
    body = [[cell(v) for v in row[: policy.max_columns]] for row in rows]
    widths = [len(h) for h in header]
    for row in body:
        for i, value in enumerate(row):
            widths[i] = max(widths[i], len(value))

    lines = [" | ".join(h.ljust(widths[i]) for i, h in enumerate(header)).rstrip()]
    lines.append("-+-".join("-" * w for w in widths))
    for row in body:
        lines.append(" | ".join(v.ljust(widths[i]) for i, v in enumerate(row)).rstrip())

    text = "\n".join(lines)
    if len(text) > policy.max_output_chars:
        text = text[: policy.max_output_chars] + "\n... [truncated]"
    return text


def make_sql_tools(
    sandbox: SQLSandbox | None = None,
    *,
    lead: str = "",
) -> list[Tool]:
    """构建绑定到某个沙箱的只读 SQL 工具集。

    四个工具，刻意很窄。``query_table`` 收的是结构化条件而不是 SQL：模型描述*它要什么*，
    本模块决定*怎么问*。
    """
    if sandbox is None:
        raise ValueError(
            "make_sql_tools needs a SQLSandbox; build one from a connection factory, "
            "for example SQLSandbox(lambda: pymysql.connect(**cfg))"
        )
    box = sandbox
    prefix = f"{lead} " if lead else ""

    @tool(name="list_tables")
    def list_tables() -> str:
        """List the tables available in the connected database.

        Use this first: it tells you the exact table names to query.
        """
        policy = box.policy
        if _dialect(box.connection()) == "sqlite":
            _cols, rows = box.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table' "
                "AND name NOT LIKE 'sqlite_%' ORDER BY name",
                [],
            )
        else:
            _cols, rows = box.execute(
                "SELECT table_name FROM information_schema.tables "
                "WHERE table_schema = DATABASE() ORDER BY table_name",
                [],
            )
        names = [str(r[0]) for r in rows]
        if policy.allowed_tables is not None:
            names = [n for n in names if n in policy.allowed_tables]
        names = [n for n in names if n not in policy.denied_tables]
        return f"{prefix}{len(names)} table(s): " + (", ".join(names) if names else "(none)")

    @tool(name="describe_table")
    def describe_table(table: str) -> str:
        """Show the columns and types of one table.

        Run this before querying so you use real column names and know what the
        values look like.

        Args:
            table: Bare table name, for example "users".
        """
        name = box._check_table(table)
        dialect = _dialect(box.connection())
        if dialect == "sqlite":
            _cols, rows = box.execute(f"PRAGMA table_info({_quote(name, dialect)})", [])
            lines = [f"{r[1]}  {r[2] or 'ANY'}{'  NOT NULL' if r[3] else ''}" for r in rows]
        else:
            _cols, rows = box.execute(
                "SELECT column_name, column_type, is_nullable FROM information_schema.columns "
                "WHERE table_schema = DATABASE() AND table_name = "
                + _placeholder(_paramstyle(box.connection()), 1)
                + " ORDER BY ordinal_position",
                [name],
            )
            lines = [f"{r[0]}  {r[1]}{'' if r[2] == 'YES' else '  NOT NULL'}" for r in rows]
        if not lines:
            raise FileNotFoundError(f"no such table: {name}")
        return f"{prefix}{name}:\n" + "\n".join(lines)

    @tool(name="query_table")
    def query_table(
        table: str,
        columns: list[str] | None = None,
        where: list[str] | None = None,
        order_by: str = "",
        limit: int = 20,
    ) -> str:
        """Read rows from one table with optional filters. Read-only.

        Prefer this over trying to write SQL: the filters are combined with AND and
        every value is sent as a bound parameter.

        Args:
            table: Bare table name, for example "orders".
            columns: Column names to return; omit for all columns.
            where: Conditions like "status = 'paid'", "amount > 100",
                "name like 'A%'", "id in (1, 2)", "note is null".
                One condition per element: use ["HS > 25", "HS < 28"],
                never ["HS > 25 AND HS < 28"].
            order_by: A column name, optionally followed by "desc": "NY desc".
                Never a direction on its own. Omit it if unsure.
            limit: Maximum rows to return; clamped to the server-side policy.
        """
        sql, params = box.build_select(table, columns, where or [], order_by, limit, probe=True)
        names, rows = box.execute(sql, params)
        capped = min(max(1, limit), box.policy.max_rows)
        truncated = len(rows) > capped
        rows = rows[:capped]
        total = f"{len(rows)} row(s) from {table} (limit {capped}"
        if truncated:
            total += "; MORE ROWS EXIST -- this is a partial result, call count_rows or narrow the filters"
        head = f"{prefix}{total})\n"
        return head + _format_table(names, rows, box.policy)

    @tool(name="count_rows")
    def count_rows(table: str, where: list[str] | None = None) -> str:
        """Count rows matching the filters, without returning them.

        Args:
            table: Bare table name.
            where: Conditions in the same form as query_table's, one per element.
        """
        name = box._check_table(table)
        dialect = _dialect(box.connection())
        clause = box._clause(list(where or []), _paramstyle(box.connection()), dialect)
        sql = f"SELECT COUNT(*) AS n FROM {_quote(name, dialect)}{clause.sql}"
        _cols, rows = box.execute(sql, clause.params)
        total = rows[0][0] if rows else 0
        suffix = " matching the filters" if clause.sql else ""
        return f"{prefix}{total} row(s) in {table}{suffix}"

    return [list_tables, describe_table, query_table, count_rows]


# --------------------------------------------------------------------- 连接


def mysql_connector(
    host: str = "127.0.0.1",
    port: int = 3306,
    user: str = "root",
    password: str = "",
    database: str = "",
    **kwargs: Any,
) -> Callable[[], Any]:
    """MySQL 的连接工厂，走 ``pymysql``。

    ``pymysql`` 是延迟导入的，所以导入本模块不需要装驱动。

    给这个账号只授 ``SELECT``::

        CREATE USER 'agent'@'%' IDENTIFIED BY '...';
        GRANT SELECT ON shop.* TO 'agent'@'%';
    """

    def connect() -> Any:
        try:
            import pymysql
        except ImportError as exc:  # pragma: no cover - depends on the environment
            raise RuntimeError(
                "pymysql is not installed. Run: uv pip install --python .venv\\Scripts\\python.exe pymysql"
            ) from exc
        # 用 dict 而不是逐个关键字参数：调用方自己传的 charset/autocommit
        # （配置来自 JSON 文件时很常见）应该覆盖默认值，而不是撞车报错。
        options: dict[str, Any] = {
            "host": host,
            "port": port,
            "user": user,
            "password": password,
            "database": database or None,
            "charset": "utf8mb4",
            "autocommit": True,
        }
        options.update(kwargs)
        return pymysql.connect(**options)

    return connect


def sqlite_connector(path: str | Path) -> Callable[[], Any]:
    """本地 sqlite 文件的连接工厂 —— 测试和 demo 用起来很顺手。"""
    import sqlite3

    target = str(path)

    def connect() -> Any:
        return sqlite3.connect(target, check_same_thread=False)

    return connect


__all__ = [
    "SQLPolicy",
    "SQLPolicyError",
    "SQLSandbox",
    "make_sql_tools",
    "mysql_connector",
    "sqlite_connector",
]
