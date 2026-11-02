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

sqlite 模式会在 ``agent_workspace/sqldemo.db`` 建一个迷你的 ``dba04_cy4`` 加两张诱饵
表，所以整个流程可以离线、可重复地跑。工具也是这么测的：``sqlite3`` 和 ``pymysql``
都是 DB-API 2.0，同一份策略代码对两者都成立。

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

    query_table(table="dba04_cy4", where=["HS > 25", "HS < 28"], order_by="NY desc", limit=20)

`where` 是列表，一个元素一个条件，不要写成 ["HS > 25 AND HS < 28"]；
`order_by` 是列名（可加 desc），不要只写 "ASC"。

Call shape (copy the structure, not the values):

    query_table(table="dba04_cy4", where=["HS > 25", "HS < 28"], order_by="NY desc", limit=20)

`where` is a list with one condition per element; never ["HS > 25 AND HS < 28"].
`order_by` is a column name, optionally with desc; never just "ASC".
"""

EXPECTED_MYSQL_COUNT = 7194
"""Rows matching HS > 25 AND HS < 28 in dba04_cy4 on the reference server (2,173,779 rows).
Handy as ``--expect-count 7194`` when you want the demo to check itself."""

SAMPLE_SCHEMA = """
-- mirrors the shape of the real dba04_cy4 (53 columns there; four are enough here)
CREATE TABLE dba04_cy4 (JH TEXT, NY TEXT, HS REAL, CYFS TEXT);
INSERT INTO dba04_cy4 (JH, NY, HS, CYFS) VALUES
    ('X7-4-46', '198206', 26.0, '1B2'),
    ('X7-4-46', '198207', 27.5, '1B2'),
    ('X7-4-46', '198208', 28.0, '1B2'),   -- boundary: excluded
    ('X7-4-47', '198206', 25.0, '1B2'),   -- boundary: excluded
    ('X7-4-47', '198207', 30.0, '1B2'),
    ('X7-4-48', '198206', 21.0, '1B2'),
    ('X7-4-48', '198207', 25.5, '1B2'),
    ('X7-4-49', '198206', NULL, '1B2'),   -- no measurement: excluded
    ('N6-D3-P2126', '198206', 26.5, '2A1'),
    ('N6-D3-P2126', '198207', 24.0, '2A1');
-- a decoy table: the "plant 1" shard and a bare dba04
CREATE TABLE dba04_cy1 (JH TEXT, NY TEXT, HS REAL);
INSERT INTO dba04_cy1 (JH, NY, HS) VALUES ('X7-4-46', '198206', 26.0);
CREATE TABLE dba04 (JH TEXT, NY TEXT, HS REAL);
INSERT INTO dba04 (JH, NY, HS) VALUES ('X7-4-46', '198206', 26.0);
"""

SAMPLE_EXPECTED = 4
"""Rows matching HS > 25 AND HS < 28 in the sqlite sample above."""


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
    """A canned model that uses the glossary the way a careful one would.

    The interesting part is step 0: it does *not* trust a guessed name.  It lists
    the tables, sees `dba04`, `dba04_cy1` and `dba04_cy4` side by side, and only then
    picks the one the suffix rule points at.  Counting before selecting is the cheap
    way to spot a wrong table: a plant-4 monthly report cannot have 21 rows.
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
    """A canned model that tries three things the policy must refuse.

    This is the interesting demo: each attempt comes back as a readable observation,
    and the loop keeps going instead of dying.
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
    """Dump the schema without involving a model.

    Useful on a large unfamiliar database: it answers "what is in here?" in one pass
    and, because every call goes through the same read-only sandbox, it cannot change
    anything either.
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
    """Parse, but turn an unknown flag into a "did you mean" instead of a bare refusal.

    Abbreviations are off on purpose (a typo must not silently take effect), so the
    least we can do is name the flag the caller probably wanted.
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
    else:
        path = build_sample_db(DEFAULT_WORKSPACE / "sqldemo.db")
        connect = sqlite_connector(path)
        label = f"sqlite {path}"

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
    """Did the model actually answer from data, or did it just emit sentences?"""
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
