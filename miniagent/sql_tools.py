"""Read-only SQL tools: inspect a schema and query one table.

Written against **DB-API 2.0** (PEP 249) rather than a specific driver, so the
same code runs on MySQL (``pymysql``) and on ``sqlite3``.  That is not just
generality for its own sake: it means the security rules below are exercised by
real SQL in the test suite instead of by a mock that can only agree with itself.

The threat model is different from the file tools.  A file tool's danger is a bad
*path*; a SQL tool's danger is a bad *statement* -- ``DROP TABLE``, ``UNION SELECT
password FROM users``, a cartesian join that pins the CPU.  Four rules cover it:

1. **Never build a statement from model text.**  Table, column and operator names
   are validated against a strict identifier regex and then a whitelist; every
   *value* travels as a bound parameter.  There is no string concatenation of user
   input anywhere, so quoting tricks are not a thing that can happen.
2. **Read-only by construction.**  The only statements this module can emit are
   ``SELECT`` / ``SHOW`` / ``PRAGMA``.  Raw SQL is off by default and, when enabled,
   still has to pass a keyword deny-list and a single-statement check.
3. **Bounded cost.**  Every query gets a ``LIMIT`` clamped to ``max_rows``, and the
   result set is truncated by cell width and total size before it reaches the
   model.
4. **The real boundary is the database user.**  Grant the agent a MySQL account
   with ``SELECT`` on one schema and nothing else.  The rules above stop accidental
   damage and prompt injection; only privileges stop a determined one.
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
    """Wrap a validated identifier in the dialect's quote character.

    Quoting is required, not cosmetic: MySQL needs backticks for ``daa05-cy2``
    (``-`` is the minus operator) and sqlite needs double quotes.  Safe because
    :data:`_IDENT` already rejected every quote character.
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
    """A request the policy refuses. Message is written for the model to act on."""


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
    """``sqlite`` or ``mysql``, inferred from the driver module of the connection."""
    module = getattr(type(conn), "__module__", "").split(".")[0]
    return "sqlite" if module == "sqlite3" else "mysql"


def _paramstyle(conn: Any) -> str:
    """PEP 249 declares ``paramstyle`` on the *module*; read it, with fallbacks."""
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
    """True when the value is wrapped in a matching pair of quotes.

    A quoted value may contain anything -- ``'x'' OR 1=1 --'`` is a legitimate string
    to search for, and it is bound as a parameter anyway.  Only *unquoted* text with
    SQL syntax in it is a mistake worth refusing.
    """
    text = raw.strip()
    return len(text) >= 2 and text[0] == text[-1] and text[0] in "'\""


def _literal(raw: str) -> Any:
    """Turn the text after an operator into a Python value.

    Quoted text stays text; bare digits become numbers, so ``amount > 100`` compares
    numerically instead of lexicographically.
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
    """Parse ``(a, 'b', 3)`` into values, respecting quotes."""
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
    """Holds one connection, the policy, and the statement builder.

    ``connect`` is a zero-argument factory so the caller decides how to reach the
    database (and so tests can hand over an in-memory sqlite connection).
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
        """Make the session itself refuse writes, and report what happened.

        Failure is deliberately *not* fatal: a restricted account may lack the
        privilege to set it, yet still be read-only by grant.  The state string is
        surfaced in stats so a caller can assert on it when it matters.
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
        """Return ``(sql, params)``. Exposed so tests can assert on the SQL itself.

        ``probe`` asks for one row *more* than the caller intends to show.  That extra
        row is a sentinel: if it comes back, more rows exist than the caller will see.
        A silent ``LIMIT`` is how "2 rows match, here is 1" becomes a confident wrong
        answer -- the model cannot know what it was not shown.
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
        """Validate a hand-written SELECT. Only reachable when policy allows it."""
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
    """Render rows as a fixed-width table, trimmed to fit the model's context."""
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
    """Build the read-only SQL tool set bound to one sandbox.

    Four tools, deliberately narrow.  ``query_table`` takes structured filters rather
    than SQL: the model describes *what it wants*, this module decides *how to ask*.
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
    """A connection factory for MySQL via ``pymysql``.

    ``pymysql`` is imported lazily so importing this module needs no driver installed.

    Grant this account ``SELECT`` only::

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
    """A connection factory for a local sqlite file -- handy for tests and demos."""
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
