"""让 agent 用只读 SQL 工具回答一个数据库问题。

    python scripts\agent_sql_demo.py                    # sqlite：自动建样例库，不用服务器
    python scripts\agent_sql_demo.py --backend local    # 同上，换成本地 0.5B 模型
    python scripts\agent_sql_demo.py --mysql            # 真 MySQL 服务器（需要 pymysql）

示例任务来自一个真实的中文油田月报库::

    查询四厂油井月报表中含水大于25小于28的行

只有知道三条约定才能回答，所以它们被塞进了 system prompt（见 ``DOMAIN_NOTES``）：

    油井月报 = dba04        四厂 = 后缀 _cy4  ->  dba04_cy4        含水 = HS

这正是这个 demo 的意义：**agent 不会比它的词汇表更聪明。**  有这份笔记，它能从 59
张表里挑对那一张；没有（``--no-domain-notes``），它就只能在 ``dba04``、``dba04_cy1``
和 ``dba04_cy4`` 之间猜，而猜错照样会返回一个看起来很合理的行数。

sqlite 模式会在 ``agent_workspace/sqldemo.db`` 建一份**离线副本**：同样是 53 列、
同样的列名和类型，只是数据是编的（14 行）。所以整个流程可以离线、可重复地跑，而且
``describe_table`` 看到的形状和内网服务器一致。工具也是这么测的：``sqlite3`` 和
``pymysql`` 都是 DB-API 2.0，同一份策略代码对两者都成立。

> 参考服务器（217 万行的那份）只能从内网访问。``--mysql`` 连不上时会直接报一句
> 「只能在公司网络用」并退出，而不是等到某个工具调用里冒出 pymysql 栈追踪。

这里什么都不可能写。沙箱把会话本身置为只读并验证它；每条语句都由校验过的标识符加
绑定值构建；结尾的 ``statements executed`` 就是审计记录。

MySQL 凭据来自命令行或环境变量::

    $env:MINIAGENT_MYSQL_HOST = "127.0.0.1"
    $env:MINIAGENT_MYSQL_USER = "agent"
    $env:MINIAGENT_MYSQL_PASSWORD = "..."
    $env:MINIAGENT_MYSQL_DB = "data"

给那个账号只授 SELECT。下面的策略挡的是误伤；挡处心积虑的提示注入靠的是权限。
"""

from __future__ import annotations

import argparse
import difflib
import json
import os
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WORKSPACE = ROOT / "agent_workspace"
sys.path.insert(0, str(ROOT))

from miniagent import DEFAULT_SYSTEM_PROMPT, Agent, SQLPolicy, SQLSandbox  # noqa: E402
from miniagent.backends import CallableBackend, ScriptedBackend  # noqa: E402
from miniagent.sql_tools import make_sql_tools, mysql_connector, sqlite_connector  # noqa: E402
from miniagent.types import AgentResult, Role, Step  # noqa: E402

QUESTION = "查询四厂油井月报表中含水大于25小于28的行"

DOMAIN_NOTES = """\
领域知识（中文油田月报库，必须使用这些约定）:
- 油井月报的主体表叫 `dba04`，按"厂"分表存储。
- "厂"用后缀表示：四厂 -> `_cy4`。所以"四厂油井月报表"就是 `dba04_cy4`。
  不要查裸的 `dba04`（那是全厂合并/其他用途），也不要查 `dba04_cy1`。
- 含水字段名是 `HS`，单位是百分数。
- 井号字段是 `JH`，年月字段是 `NY`（形如 198206）。
- `HS` 为 NULL 表示当月没有测量数据，范围条件会自动排除这些行。

Domain notes (Chinese oilfield monthly report database; these conventions are required):
- The oil-well monthly report lives in `dba04`, sharded by plant.
- The plant is a suffix: plant 4 -> `_cy4`. "四厂油井月报表" is therefore `dba04_cy4`,
  not plain `dba04` and not `dba04_cy1`.
- Water cut is column `HS`, expressed as a percentage.
- Well name is `JH`; year-month is `NY` (for example 198206).
- A NULL `HS` means no measurement that month; range filters drop those rows.

调用形状（照抄结构，不要照抄数值）:

    query_table(table="dba04_cy4", columns=["JH", "NY", "HS"],
                where=["HS > 25", "HS < 28"], order_by="NY desc", limit=20)

`where` 是列表，一个元素一个条件，不要写成 ["HS > 25 AND HS < 28"]；
`order_by` 是列名（可加 desc），不要只写 "ASC"。
`columns` 要写清要哪几列 —— 这张表有 53 列，不写就是 53 列全返回。

Call shape (copy the structure, not the values):

    query_table(table="dba04_cy4", columns=["JH", "NY", "HS"],
                where=["HS > 25", "HS < 28"], order_by="NY desc", limit=20)

`where` is a list with one condition per element; never ["HS > 25 AND HS < 28"].
`order_by` is a column name, optionally with desc; never just "ASC".
Always name the `columns` you need: this table has 53 of them.
"""

EXPECTED_MYSQL_COUNT = 7194
"""Rows matching HS > 25 AND HS < 28 in dba04_cy4 on the reference server (2,173,779 rows).

That server is only reachable from the intranet, so ``--mysql`` fails at home.  The
offline path below replicates the same 53 columns locally instead; use
``--expect-count 7194`` only when you are actually on the intranet."""

DBA04_CY4_COLUMNS: tuple[tuple[str, str], ...] = (
    ("JH", "TEXT"),
    ("NY", "TEXT"),
    ("SCTS", "REAL"),
    ("CYFS", "TEXT"),
    ("BJ", "REAL"),
    ("BS", "REAL"),
    ("PL", "REAL"),
    ("YZ", "REAL"),
    ("CC", "REAL"),
    ("CC1", "REAL"),
    ("YY", "REAL"),
    ("TY", "REAL"),
    ("LY", "REAL"),
    ("JY", "REAL"),
    ("DYM", "REAL"),
    ("JYM", "REAL"),
    ("RCYL", "REAL"),
    ("RCSL", "REAL"),
    ("HS", "REAL"),
    ("YCYL", "REAL"),
    ("YCSL", "REAL"),
    ("YCQL", "REAL"),
    ("NCYL", "REAL"),
    ("NCSL", "REAL"),
    ("NCQL", "REAL"),
    ("LJCYL", "REAL"),
    ("LJCSL", "REAL"),
    ("LJCQL", "REAL"),
    ("CCYND", "REAL"),
    ("HSL1", "REAL"),
    ("BZDM1", "TEXT"),
    ("BZDM2", "TEXT"),
    ("BZDM3", "TEXT"),
    ("BZDM4", "TEXT"),
    ("YZYL", "REAL"),
    ("NZYL", "REAL"),
    ("LJZYL", "REAL"),
    ("BZ", "TEXT"),
    ("CW", "TEXT"),
    ("BX", "REAL"),
    ("SCJDDS", "REAL"),
    ("SCJDDS1", "REAL"),
    ("HSYCYL", "REAL"),
    ("HSYCSL", "REAL"),
    ("HSYCQL", "REAL"),
    ("HSNCYL", "REAL"),
    ("HSNCSL", "REAL"),
    ("HSNCQL", "REAL"),
    ("HSLJCYL", "REAL"),
    ("HSLJCSL", "REAL"),
    ("HSLJCQL", "REAL"),
    ("RCBZ", "TEXT"),
    ("_NullFlags", "TEXT"),
)
"""The 53 columns of the reference server's ``dba04_cy4``, in table order.

Real column names and types, so ``describe_table`` shows the same shape the intranet
server does.  Kept as data rather than DDL text so the count cannot drift: a wrong
number here would make the offline demo teach the wrong schema."""

# Fabricated rows.  Only about a dozen columns are filled -- the reference table is
# mostly NULL too, and a full 53-column INSERT would bury the interesting values.
SAMPLE_ROWS: tuple[dict[str, object], ...] = (
    # 25 < HS < 28 -- these are the answer
    {"JH": "X7-4-46", "NY": "198206", "HS": 26.0, "CYFS": "1B2", "SCTS": 30.0, "PL": 0.0},
    {"JH": "X7-4-46", "NY": "198207", "HS": 27.5, "CYFS": "1B2", "SCTS": 30.0, "PL": 0.0},
    {"JH": "X7-4-48", "NY": "198207", "HS": 25.5, "CYFS": "1B2", "SCTS": 31.0, "PL": 0.0},
    {"JH": "N6-D3-P2126", "NY": "198206", "HS": 26.5, "CYFS": "2A1", "SCTS": 28.0, "PL": 1.2},
    {"JH": "N6-D3-P2126", "NY": "198208", "HS": 27.2, "CYFS": "2A1", "SCTS": 28.0, "PL": 1.2},
    {"JH": "N6-D3-P2127", "NY": "198208", "HS": 25.9, "CYFS": "2A1", "SCTS": 27.0, "PL": 0.8},
    # boundary: 28.0 and 25.0 are outside the open interval
    {"JH": "X7-4-46", "NY": "198208", "HS": 28.0, "CYFS": "1B2", "SCTS": 30.0, "PL": 0.0},
    {"JH": "X7-4-47", "NY": "198206", "HS": 25.0, "CYFS": "1B2", "SCTS": 32.0, "PL": 0.0},
    # outside the interval for other reasons
    {"JH": "X7-4-47", "NY": "198207", "HS": 30.0, "CYFS": "1B2", "SCTS": 32.0, "PL": 0.0},
    {"JH": "X7-4-48", "NY": "198206", "HS": 21.0, "CYFS": "1B2", "SCTS": 31.0, "PL": 0.0},
    {"JH": "N6-D3-P2126", "NY": "198207", "HS": 24.0, "CYFS": "2A1", "SCTS": 28.0, "PL": 1.2},
    {"JH": "X7-4-50", "NY": "198206", "HS": 12.5, "CYFS": "1B2", "SCTS": 29.0, "PL": 0.0},
    {"JH": "X7-4-50", "NY": "198207", "HS": 95.0, "CYFS": "1B2", "SCTS": 29.0, "PL": 0.0},
    # no measurement that month
    {"JH": "X7-4-49", "NY": "198206", "HS": None, "CYFS": "1B2", "SCTS": 33.0, "PL": 0.0},
)
"""Fabricated data. Counts inside are deliberately distinguishable:

* ``dba04_cy4`` -> 6 rows match, the right answer;
* ``dba04_cy1`` -> 2 rows match, so a wrong plant suffix shows up as 2;
* ``dba04``     -> 3 rows match, so ignoring the suffix rule shows up as 3.

A decoy that returns 0 would be the dangerous case: "no rows" also looks like a real
answer, and nothing would tell you the table was wrong."""

SAMPLE_SCHEMA_HEADER = """\
-- Offline replica of the reference server's dba04_cy4: same 53 columns, invented rows.
-- The 2,173,779-row table lives on the intranet; this file exists so the demo runs anywhere.
"""


def _ddl(table: str, columns: tuple[tuple[str, str], ...]) -> str:
    body = ",\n  ".join(f"{name} {kind}" for name, kind in columns)
    return f"CREATE TABLE {table} (\n  {body}\n);\n"


def _insert(table: str, rows: tuple[dict[str, object], ...]) -> str:
    names = list(rows[0])
    for row in rows:
        if list(row) != names:
            raise ValueError(f"{table}: every row must name the same columns, in order")
    cols = ", ".join(names)
    lines = []
    for row in rows:
        values = ", ".join(_sql_literal(row[name]) for name in names)
        lines.append(f"    ({values})")
    return f"INSERT INTO {table} ({cols}) VALUES\n" + ",\n".join(lines) + ";\n"


def _sql_literal(value: object) -> str:
    if value is None:
        return "NULL"
    if isinstance(value, str):
        return "'" + value.replace("'", "''") + "'"
    return repr(value)


def _sample_schema() -> str:
    parts = [
        SAMPLE_SCHEMA_HEADER,
        _ddl("dba04_cy4", DBA04_CY4_COLUMNS),
        _insert("dba04_cy4", SAMPLE_ROWS),
        "-- decoys: same shape, other plants / the un-sharded parent table\n",
        _ddl("dba04_cy1", DBA04_CY4_COLUMNS),
        _insert(
            "dba04_cy1",
            (
                {"JH": "X7-4-46", "NY": "198206", "HS": 26.0, "CYFS": "1B2"},
                {"JH": "X7-4-46", "NY": "198207", "HS": 27.0, "CYFS": "1B2"},
            ),
        ),
        _ddl("dba04", DBA04_CY4_COLUMNS),
        _insert(
            "dba04",
            (
                {"JH": "X7-4-46", "NY": "198206", "HS": 26.0, "CYFS": "1B2"},
                {"JH": "X7-4-46", "NY": "198207", "HS": 26.5, "CYFS": "1B2"},
                {"JH": "X7-4-47", "NY": "198206", "HS": 27.9, "CYFS": "1B2"},
            ),
        ),
    ]
    return "\n".join(parts)


SAMPLE_SCHEMA = _sample_schema()

SAMPLE_EXPECTED = 6
"""Rows matching HS > 25 AND HS < 28 in the offline replica above."""


def tool_call(tool: str, **arguments) -> str:
    return "<tool_call>" + json.dumps({"name": tool, "arguments": arguments}) + "</tool_call>"


def build_sample_db(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    conn = sqlite3.connect(path)
    conn.executescript(SAMPLE_SCHEMA)
    conn.commit()
    conn.close()
    return path


def scripted_backend() -> ScriptedBackend:
    """一个写死的模型，像谨慎的模型那样使用词汇表。

    有意思的是 step 0：它*不*相信猜来的名字。它先列出表，看到 `dba04`、`dba04_cy1` 和
    `dba04_cy4` 并排摆在眼前，然后才按后缀规则挑中那一张。先计数再取数是发现选错表的
    便宜办法：一个厂的四厂月报不可能只有 21 行。
    """
    return ScriptedBackend(
        [
            tool_call("list_tables"),
            tool_call("describe_table", table="dba04_cy4"),
            tool_call("count_rows", table="dba04_cy4", where=["HS > 25", "HS < 28"]),
            tool_call(
                "query_table",
                table="dba04_cy4",
                columns=["JH", "NY", "HS"],
                where=["HS > 25", "HS < 28"],
                order_by="NY",
                limit=20,
            ),
            "四厂油井月报表（dba04_cy4）中含水在 25 到 28 之间的记录如下，"
            "表里共匹配到若干行，上面列出了前 20 行。",
        ]
    )


def scripted_backend_attacking() -> CallableBackend:
    """一个写死的模型，专门尝试三件策略必须拒绝的事。

    这是有意思的那个 demo：每次尝试都以一条可读的观察返回，而循环继续跑，不会死掉。
    """

    def decide(messages):
        observations = [m.content for m in messages if m.role == Role.TOOL]
        attempts = [
            ("query_table", {"table": "dba04_cy4; DROP TABLE dba04_cy4"}),
            ("query_table", {"table": "dba04_cy4", "where": ["1=1"]}),
            ("query_table", {"table": "dba04_cy4", "columns": ["HS; DROP TABLE x"]}),
        ]
        if len(observations) < len(attempts):
            name, args = attempts[len(observations)]
            return tool_call(name, **args)
        return "Every attempt was refused; the policy held."

    return CallableBackend(decide)


def print_step(step: Step) -> None:
    print(f"\n--- step {step.index} ({step.duration_s * 1000:.0f} ms) ---")
    if step.assistant_text:
        print(f"model: {step.assistant_text.strip()}")
    for call, result in zip(step.tool_calls, step.tool_results, strict=False):
        arrow = "->" if result.ok else "!!"
        print(f"  call   {call.name}({json.dumps(call.arguments, ensure_ascii=False)})")
        body = result.content.replace("\n", "\n           ").rstrip()
        print(f"  {arrow} result {body}")
    for err in step.parse_errors:
        print(f"  parse error: {err}")
    for note in step.notes:
        print(f"  note: {note}")


def explore(sandbox: SQLSandbox, *, max_tables: int, columns: int, sample: int) -> None:
    """不牵扯模型，直接把 schema 倒出来。

    在陌生的大库上很有用：它一遍回答「这里头有什么」，而且因为每次调用都走同一个只读
    沙箱，它同样改不了任何东西。
    """
    tools = {t.name: t for t in make_sql_tools(sandbox)}
    print("\n" + "=" * 70)
    print(tools["list_tables"].func())

    names = []
    cursor = sandbox.connection().cursor()
    try:
        from miniagent.sql_tools import _dialect

        if _dialect(sandbox.connection()) == "sqlite":
            cursor.execute("SELECT name FROM sqlite_master WHERE type = 'table' ORDER BY name")
        else:
            cursor.execute(
                "SELECT table_name FROM information_schema.tables "
                "WHERE table_schema = DATABASE() ORDER BY table_name"
            )
        names = [r[0] for r in cursor.fetchall()]
    finally:
        cursor.close()

    policy = sandbox.policy
    if policy.allowed_tables is not None:
        names = [n for n in names if n in policy.allowed_tables]
    names = [n for n in names if n not in policy.denied_tables][:max_tables]

    for name in names:
        print("\n" + "=" * 70)
        try:
            described = tools["describe_table"].func(name).splitlines()
            print(f"-- {name} ({len(described) - 1} columns)")
            print("\n".join(described[1 : columns + 1]))
            if len(described) - 1 > columns:
                print(f"   ... ({len(described) - 1 - columns} more)")
        except Exception as exc:
            print(f"-- {name}: {type(exc).__name__}: {exc}")
            continue
        if sample > 0:
            try:
                rows = tools["query_table"].func(name, limit=sample)
                print("\n".join("   " + line for line in rows.splitlines()))
            except Exception as exc:
                print(f"   sample failed: {type(exc).__name__}: {exc}")


def _parse(parser: argparse.ArgumentParser) -> argparse.Namespace:
    """解析参数，但把不认识的开关变成一句「你是不是想打」，而不是干巴巴的拒绝。

    缩写是刻意关掉的（打错字不许静默生效），所以至少可以做到一件事：把调用方大概想打的
    那个开关名报出来。
    """
    args, unknown = parser.parse_known_args()
    if not unknown:
        return args
    known = sorted(parser._option_string_actions)  # noqa: SLF001 - argparse has no public list
    for token in unknown:
        name = token.split("=", 1)[0]
        if not name.startswith("--"):
            continue
        hit = difflib.get_close_matches(name, known, n=1, cutoff=0.7)
        if hit:
            parser.error(f"unrecognized argument {name!r}; did you mean {hit[0]!r}?")
    parser.error(f"unrecognized arguments: {' '.join(unknown)}")
    raise AssertionError("unreachable")  # parser.error 会退出


def _require_reachable(connect, label: str) -> None:
    """Fail early and clearly when the remote server cannot be reached.

    The reference MySQL lives on the intranet.  Without this check the first sign of
    trouble is a pymysql traceback from inside a tool call, several seconds into a model
    run -- which reads like a code bug rather than "you are not on the company network".
    """
    try:
        connect().close()
    except Exception as exc:
        print(f"cannot reach {label}: {type(exc).__name__}: {exc}", file=sys.stderr)
        print(
            "\nThat server is only reachable from the intranet.  Drop --mysql to use the"
            " offline replica of dba04_cy4 (same 53 columns, fabricated rows):"
            "\n    python scripts\\agent_sql_demo.py --backend local",
            file=sys.stderr,
        )
        raise SystemExit(2) from None


def main() -> None:
    # allow_abbrev=False：默认情况下 `--max-row 1` 会被静默缩写成
    # --max-rows，打错字照样生效。和这个仓库其它地方同一条规矩 ——
    # 名字写错就必须大声失败。
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument(
        "--backend",
        default="scripted",
        choices=["scripted", "scripted-attack", "local"],
        help="scripted = deterministic fake model (no torch)",
    )
    parser.add_argument("--mysql", action="store_true", help="use a real MySQL server")
    parser.add_argument(
        "--config-json",
        default="",
        help="path to a JSON file with host/port/user/password/database (keeps secrets out of argv)",
    )
    parser.add_argument("--host", default=os.environ.get("MINIAGENT_MYSQL_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("MINIAGENT_MYSQL_PORT", 3306)))
    parser.add_argument("--user", default=os.environ.get("MINIAGENT_MYSQL_USER", "root"))
    parser.add_argument("--password", default=os.environ.get("MINIAGENT_MYSQL_PASSWORD", ""))
    parser.add_argument("--database", default=os.environ.get("MINIAGENT_MYSQL_DB", ""))
    parser.add_argument(
        "--allow-tables",
        default="",
        help="comma-separated allowlist; empty means 'anything the user can see'",
    )
    parser.add_argument("--max-rows", type=int, default=20)
    parser.add_argument("--question", default=QUESTION)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--max-steps", type=int, default=8)
    parser.add_argument(
        "--explore",
        action="store_true",
        help="skip the agent: dump the schema and a few sample rows",
    )
    parser.add_argument("--max-tables", type=int, default=10, help="cap for --explore")
    parser.add_argument("--explore-columns", type=int, default=12, help="columns shown per table")
    parser.add_argument("--sample", type=int, default=2, help="sample rows per table (0 = none)")
    parser.add_argument(
        "--expect-count",
        type=int,
        default=-1,
        help="after the run, compare COUNT(*) of the target table against this number",
    )
    parser.add_argument(
        "--target-table",
        default="dba04_cy4",
        help="table used by --expect-count (and by the end-of-run summary)",
    )
    parser.add_argument(
        "--no-domain-notes",
        action="store_true",
        help="do not inject the domain glossary; useful to see the model fail without it",
    )
    args = _parse(parser)

    if args.config_json:
        import json as _json

        conf = _json.loads(Path(args.config_json).read_text(encoding="utf-8"))
        args.host = conf.get("host", args.host)
        args.port = int(conf.get("port", args.port))
        args.user = conf.get("user", args.user)
        args.password = conf.get("password", args.password)
        args.database = conf.get("database", args.database)

    if args.mysql:
        connect = mysql_connector(
            host=args.host,
            port=args.port,
            user=args.user,
            password=args.password,
            database=args.database,
        )
        label = f"mysql {args.user}@{args.host}:{args.port}/{args.database or '(none)'}"
        _require_reachable(connect, label)
    else:
        path = build_sample_db(DEFAULT_WORKSPACE / "sqldemo.db")
        connect = sqlite_connector(path)
        label = f"sqlite {path} (offline replica: 53 columns, {len(SAMPLE_ROWS)} rows)"

    allowed = frozenset(t.strip() for t in args.allow_tables.split(",") if t.strip())
    policy = SQLPolicy(
        allowed_tables=allowed or None,
        denied_tables=frozenset({"mysql", "information_schema", "performance_schema"}),
        max_rows=args.max_rows,
    )
    sandbox = SQLSandbox(connect, policy)

    # --explore 完全不需要模型，所以它放在构造后端之前：
    # 否则 `--explore --backend local` 会先加载 1 GB 权重，只为打印一份 schema。
    if args.explore:
        sandbox.connection()  # 报告只读状态之前，先让它真的生效
        print(f"database : {label}")
        print(f"read-only: {sandbox.read_only_state}")
        try:
            explore(sandbox, max_tables=args.max_tables, columns=args.explore_columns, sample=args.sample)
        finally:
            print(f"\nstatements executed ({len(sandbox.queries)})")
            sandbox.close()
        return

    if args.backend == "scripted":
        backend = scripted_backend()
    elif args.backend == "scripted-attack":
        backend = scripted_backend_attacking()
    else:
        from miniagent.backends import MiniVLLMBackend

        backend = MiniVLLMBackend(device=args.device, dtype=args.dtype)
        print(backend.describe())

    tools = make_sql_tools(sandbox)

    system_prompt = DEFAULT_SYSTEM_PROMPT + (
        " You can inspect a database with read-only tools. List the tables, read the"
        " columns, then use query_table with structured filters instead of writing SQL."
        " Never modify anything: every tool is read-only by design."
    )
    if not args.no_domain_notes:
        system_prompt += "\n\n" + DOMAIN_NOTES

    agent = Agent(
        backend,
        tools,
        system_prompt=system_prompt,
        max_steps=args.max_steps,
        on_step=print_step,
    )

    print(f"database : {label}")
    print(f"backend  : {args.backend}")
    print(f"policy   : max_rows={policy.max_rows} allowlist={sorted(allowed) or 'any'}")
    print(f"domain   : {'injected' if not args.no_domain_notes else 'NOT injected'}")
    print(f"tools    : {agent.tools.names}")
    if not args.mysql:
        print(
            f"replica  : {SAMPLE_EXPECTED} rows match HS in (25, 28);"
            f" pass --expect-count {SAMPLE_EXPECTED} to assert it"
        )
    print(f"question : {args.question}")

    result = agent.run(args.question)

    print("\n" + "=" * 70)
    print(f"answer     : {result.answer}")
    print(f"stopped    : {result.stopped_reason.value}")
    print(f"steps      : {len(result.steps)}")
    print(f"tool calls : {result.num_tool_calls} ({result.num_tool_errors} failed)")

    print(f"\nstatements executed ({len(sandbox.queries)}):")
    for sql in sandbox.queries:
        print(f"  {sql}")
    executed = len(sandbox.queries)

    if args.expect_count >= 0:
        counts = {t.name: t for t in tools}
        try:
            actual = counts["count_rows"].func(args.target_table, where=["HS > 25", "HS < 28"])
        except Exception as exc:
            print(f"\nreference count failed: {type(exc).__name__}: {exc}")
        else:
            hit = f"{args.expect_count}" in actual.replace(",", "")
            print(
                f"\nreference count: {actual}  (expected {args.expect_count})  -> "
                f"{'MATCH' if hit else 'MISMATCH'}"
            )
            print("  measured by this script with its own filter -- it says the database is")
            print("  what we think it is, not that the model got the right answer")

    print(f"\nverdict    : {verdict(result, executed)}")

    sandbox.close()
    if result.stopped_reason.value != "final_answer":
        sys.exit(1)


def verdict(result: AgentResult, executed: int) -> str:
    """模型真的从数据里答出来的，还是只是吐了几句话？"""
    if result.stopped_reason.value != "final_answer":
        if executed:
            return (
                f"no answer: {result.stopped_reason.value} after {executed} statement(s). "
                "The tools refused the calls rather than returning a plausible wrong result."
            )
        return f"no answer: {result.stopped_reason.value}, and no statement ever ran."
    if not executed:
        return "answered WITHOUT reading anything -- treat the answer as invented."
    return f"answered after {len(result.steps)} step(s) with data from {executed} statement(s)."


if __name__ == "__main__":
    main()
