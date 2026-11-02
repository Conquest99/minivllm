"""Let the agent explore a database with the read-only SQL tools.

    python scripts\\agent_sql_demo.py                  # sqlite: builds a sample DB, no server
    python scripts\\agent_sql_demo.py --backend local   # same, with the local 0.5B model
    python scripts\\agent_sql_demo.py --mysql           # a real MySQL server (needs pymysql)

The sqlite mode is the one to read first: it creates ``agent_workspace/sqldemo.db``
with a tiny shop schema, so the whole thing runs offline and repeatably.  It is also
how the SQL tools are tested -- ``sqlite3`` and ``pymysql`` are both DB-API 2.0, so
the same policy code runs against either.

MySQL credentials come from flags or the environment::

    $env:MINIAGENT_MYSQL_HOST = "127.0.0.1"
    $env:MINIAGENT_MYSQL_USER = "agent"
    $env:MINIAGENT_MYSQL_PASSWORD = "..."
    $env:MINIAGENT_MYSQL_DB = "shop"

Grant that account SELECT only.  The policy below stops accidents; privileges are
what stops a determined prompt-injection.
"""

from __future__ import annotations

import argparse
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
from miniagent.types import Role, Step  # noqa: E402

QUESTION = "Which customers have paid orders worth more than 50? Give me their names and the amounts."

SAMPLE_SCHEMA = """
CREATE TABLE customers (id INTEGER PRIMARY KEY, name TEXT, city TEXT, joined TEXT);
CREATE TABLE orders (id INTEGER PRIMARY KEY, customer_id INTEGER, amount REAL, status TEXT);
INSERT INTO customers (name, city, joined) VALUES
    ('alice', 'Berlin',  '2024-01-11'),
    ('bob',   'Lisbon',  '2024-02-03'),
    ('carol', 'Berlin',  '2024-03-19'),
    ('dave',  'Madrid',  '2024-04-02');
INSERT INTO orders (customer_id, amount, status) VALUES
    (1, 120.0, 'paid'),
    (1,  15.5, 'paid'),
    (2,  99.0, 'open'),
    (2,  75.0, 'paid'),
    (3,  40.0, 'paid'),
    (4, 210.0, 'refunded'),
    (4,  88.0, 'paid');
"""


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
    """A canned model that explores the schema the way a careful one would.

    It lists tables, describes the two it needs, counts before selecting (cheap
    sanity check), then queries with a join-free filter and reads nothing else.
    """
    return ScriptedBackend(
        [
            tool_call("list_tables"),
            tool_call("describe_table", table="customers"),
            tool_call("describe_table", table="orders"),
            tool_call(
                "query_table",
                table="orders",
                columns=["customer_id", "amount"],
                where=["status = 'paid'", "amount > 50"],
                order_by="amount desc",
            ),
            tool_call("query_table", table="customers", columns=["id", "name"], where=["id in (1, 2, 4)"]),
            "alice paid 120.0, bob paid 75.0 and dave paid 88.0 -- all above 50.",
        ]
    )


def scripted_backend_attacking() -> CallableBackend:
    """A canned model that tries three things the policy must refuse.

    This is the interesting demo: each attempt comes back as a readable observation,
    and the loop keeps going instead of dying.
    """

    def decide(messages):
        observations = [m.content for m in messages if m.role is Role.TOOL]
        attempts = [
            ("query_table", {"table": "customers; DROP TABLE customers"}),
            ("query_table", {"table": "customers", "where": ["1=1"]}),
            ("query_table", {"table": "customers", "columns": ["name; DROP TABLE x"]}),
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
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
    args = parser.parse_args()

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

    if args.backend == "scripted":
        backend = scripted_backend()
    elif args.backend == "scripted-attack":
        backend = scripted_backend_attacking()
    else:
        from miniagent.backends import MiniVLLMBackend

        backend = MiniVLLMBackend(device=args.device, dtype=args.dtype)
        print(backend.describe())

    tools = make_sql_tools(sandbox)

    if args.explore:
        sandbox.connection()  # apply the read-only guard before reporting it
        print(f"database : {label}")
        print(f"read-only: {sandbox.read_only_state}")
        try:
            explore(sandbox, max_tables=args.max_tables, columns=args.explore_columns, sample=args.sample)
        finally:
            print(f"\nstatements executed ({len(sandbox.queries)})")
            sandbox.close()
        return

    agent = Agent(
        backend,
        tools,
        system_prompt=(
            DEFAULT_SYSTEM_PROMPT + " You can inspect a database with read-only tools. List the tables, read"
            " the columns, then use query_table with filters instead of writing SQL."
        ),
        max_steps=args.max_steps,
        on_step=print_step,
    )

    print(f"database : {label}")
    print(f"backend  : {args.backend}")
    print(f"policy   : max_rows={policy.max_rows} allowlist={sorted(allowed) or 'any'}")
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

    sandbox.close()
    if result.stopped_reason.value != "final_answer":
        sys.exit(1)


if __name__ == "__main__":
    main()
