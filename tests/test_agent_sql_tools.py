"""SQL tools exercised against a real database.

``sqlite3`` is a DB-API 2.0 driver, exactly like ``pymysql``, so these tests run the
*real* statement builder and the *real* driver against real tables.  Nothing here is
mocked, which matters because the thing being tested is a security boundary: a mock
would only prove the mock agrees with itself.

The one thing sqlite cannot prove is MySQL-specific SQL (``information_schema``,
``SHOW``).  Those paths are covered by a small formatting test and must be checked
against a live server by hand -- see ``scripts/agent_sql_demo.py --mysql``.
"""

from __future__ import annotations

import sqlite3

import pytest

from miniagent.sql_tools import (
    SQLPolicy,
    SQLPolicyError,
    SQLSandbox,
    make_sql_tools,
    sqlite_connector,
)
from miniagent.tools import ToolRegistry
from miniagent.types import ToolCall


@pytest.fixture
def db(tmp_path):
    """A real sqlite file with two tables and a few rows."""
    path = tmp_path / "shop.db"
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT, status TEXT, note TEXT);
        CREATE TABLE orders (id INTEGER PRIMARY KEY, user_id INTEGER, amount REAL, state TEXT);
        INSERT INTO users (name, status, note) VALUES
            ('alice', 'active', NULL),
            ('bob',   'active', 'vip'),
            ('carol', 'paused', NULL),
            ('dave',  'active', NULL);
        INSERT INTO orders (user_id, amount, state) VALUES
            (1, 10.0, 'paid'),
            (1, 99.5, 'paid'),
            (2,  5.0, 'open'),
            (3, 42.0, 'open');
        """
    )
    conn.commit()
    conn.close()
    return path


@pytest.fixture
def box(db):
    with SQLSandbox(sqlite_connector(db)) as sandbox:
        yield sandbox


def tools_of(box) -> dict:
    return {t.name: t for t in make_sql_tools(box)}


# ------------------------------------------------------------------ the tools


def test_list_tables(box):
    out = tools_of(box)["list_tables"].func()
    assert "2 table(s)" in out
    assert "users" in out and "orders" in out


def test_describe_table(box):
    out = tools_of(box)["describe_table"].func("users")
    assert "id" in out and "INTEGER" in out
    assert "name" in out and "TEXT" in out


def test_query_all_rows(box):
    out = tools_of(box)["query_table"].func("users")
    assert "4 row(s) from users" in out
    assert "alice" in out and "carol" in out


def test_query_selected_columns(box):
    out = tools_of(box)["query_table"].func("users", columns=["name"])
    assert "name" in out
    assert "status" not in out.split("\n")[1]  # header line only has `name`


def test_query_with_equality_filter(box):
    out = tools_of(box)["query_table"].func("users", where=["status = 'active'"])
    assert "3 row(s)" in out
    assert "carol" not in out


def test_query_with_numeric_comparison(box):
    out = tools_of(box)["query_table"].func("orders", where=["amount > 40"])
    assert "2 row(s)" in out
    assert "10.0" not in out


def test_query_with_like(box):
    out = tools_of(box)["query_table"].func("users", where=["name like 'a%'"])
    assert "1 row(s)" in out and "alice" in out


def test_query_with_in_list(box):
    out = tools_of(box)["query_table"].func("users", where=["id in (1, 3)"])
    assert "2 row(s)" in out
    assert "alice" in out and "carol" in out and "bob" not in out


def test_query_with_is_null(box):
    out = tools_of(box)["query_table"].func("users", where=["note is null"])
    assert "3 row(s)" in out
    assert "vip" not in out


def test_query_with_is_not_null(box):
    out = tools_of(box)["query_table"].func("users", where=["note is not null"])
    assert "1 row(s)" in out and "vip" in out


def test_query_with_two_conditions_are_anded(box):
    out = tools_of(box)["query_table"].func("users", where=["status = 'active'", "note is not null"])
    assert "1 row(s)" in out and "bob" in out


def test_query_honours_order_by(box):
    out = tools_of(box)["query_table"].func("users", columns=["name"], order_by="name desc")
    body = out.splitlines()[3:]  # [0] summary, [1] header, [2] separator
    assert body[0].strip().startswith("dave")


def test_query_limit_is_clamped_to_the_policy(db):
    policy = SQLPolicy(max_rows=2)
    with SQLSandbox(sqlite_connector(db), policy) as box:
        out = make_sql_tools(box)[2].func("users", limit=999)
    assert "2 row(s)" in out


def test_count_rows(box):
    assert "4 row(s) in users" in tools_of(box)["count_rows"].func("users")
    assert "3 row(s)" in tools_of(box)["count_rows"].func("users", where=["status = 'active'"])


def test_null_is_rendered_explicitly(box):
    out = tools_of(box)["query_table"].func("users", columns=["name", "note"], where=["id = 1"])
    assert "NULL" in out


# ------------------------------------------------------------------- security


@pytest.mark.parametrize(
    "table",
    [
        "users; DROP TABLE users",
        "users WHERE 1=1",
        "../../etc/passwd",
        "`users`",
        "'users'",
        'users"',
        "",
    ],
)
def test_table_names_that_are_not_bare_identifiers_are_refused(box, table):
    with pytest.raises(SQLPolicyError):
        box.build_select(table, None, [], "", 10)


def test_a_comment_marker_inside_a_name_cannot_comment_anything_out(box):
    """``users--`` is a legal identifier once quoted, so it stays one identifier."""
    sql, _params = box.build_select("users--", None, [], "", 10)
    assert sql == 'SELECT * FROM "users--" LIMIT 10'
    assert " -- " not in sql


@pytest.mark.parametrize(
    "column",
    ["name; DROP TABLE users", "name = 'x' OR '1'='1", "*", "count(*)"],
)
def test_column_names_that_are_not_bare_identifiers_are_refused(box, column):
    with pytest.raises(SQLPolicyError):
        box.build_select("users", [column], [], "", 10)


@pytest.mark.parametrize(
    "condition",
    [
        "1=1",
        "name ~ 'x'",
        "id between 1 and 2",
        "name =",
    ],
)
def test_conditions_must_use_the_supported_syntax(box, condition):
    with pytest.raises(SQLPolicyError):
        box.build_select("users", None, [condition], "", 10)


@pytest.mark.parametrize(
    "condition",
    [
        "name = 'x' OR 1=1",
        "name = 'x'; DROP TABLE users",
        "name = 'x' UNION SELECT 1",
    ],
)
def test_sql_lookalikes_inside_a_value_are_harmless(box, condition):
    """These are not injection attempts that got through -- they are *values*.

    Everything after the operator is bound as a parameter, so ``OR 1=1`` and
    ``; DROP TABLE`` are compared as literal text and match nothing.  Asserting that
    they raise would be asserting the wrong thing: the point is that they cannot
    execute, not that they are rejected.
    """
    sql, params = box.build_select("users", None, [condition], "", 10)
    assert sql.count("?") == 1, "exactly one bound value, no concatenation"
    for fragment in ("OR 1=1", "DROP TABLE", "UNION SELECT"):
        if fragment in condition:
            assert fragment not in sql

    _cols, rows = box.execute(sql, params)
    assert rows == []


def test_a_quote_in_a_value_is_carried_as_a_parameter_not_as_sql(box):
    """The value is bound, so a quote inside it cannot end the literal."""
    sql, params = box.build_select("users", None, ["name = 'o''brien'"], "", 10)
    assert "o''brien" not in sql
    assert params == ["o''brien"]
    assert sql.count("?") == 1


def test_a_value_that_looks_like_sql_stays_a_value(box):
    sql, params = box.build_select("users", None, ["name = 'x'' OR 1=1 --'"], "", 10)
    assert "OR 1=1" not in sql
    assert params == ["x'' OR 1=1 --"]
    _cols, rows = box.execute(sql, params)
    assert rows == []


def test_order_by_rejects_anything_but_column_and_direction(box):
    with pytest.raises(SQLPolicyError):
        box.build_select("users", None, [], "name; DROP TABLE users", 10)
    with pytest.raises(SQLPolicyError):
        box.build_select("users", None, [], "name desc, (select 1)", 10)


def test_table_allowlist_is_enforced(db):
    with SQLSandbox(sqlite_connector(db), SQLPolicy(allowed_tables=frozenset({"users"}))) as box:
        assert "1 table(s)" in make_sql_tools(box)[0].func()
        with pytest.raises(SQLPolicyError, match="not in the allowlist"):
            box.build_select("orders", None, [], "", 10)


def test_denied_tables_win_over_the_allowlist(db):
    policy = SQLPolicy(allowed_tables=frozenset({"users"}), denied_tables=frozenset({"users"}))
    with SQLSandbox(sqlite_connector(db), policy) as box:
        with pytest.raises(SQLPolicyError, match="blocked by policy"):
            box.build_select("users", None, [], "", 10)


def test_max_columns_is_enforced(db):
    with SQLSandbox(sqlite_connector(db), SQLPolicy(max_columns=1)) as box:
        with pytest.raises(SQLPolicyError, match="at most 1 column"):
            box.build_select("users", ["id", "name"], [], "", 10)


def test_raw_sql_is_off_by_default(box):
    with pytest.raises(SQLPolicyError, match="raw SQL is disabled"):
        box.check_raw("SELECT * FROM users")


def test_raw_sql_allows_a_select_when_enabled(db):
    with SQLSandbox(sqlite_connector(db), SQLPolicy(allow_raw_select=True)) as box:
        assert box.check_raw("SELECT 1") == "SELECT 1 LIMIT 100"
        assert box.check_raw("select * from users limit 5").lower().endswith("limit 5")


@pytest.mark.parametrize(
    "sql",
    [
        "DROP TABLE users",
        "UPDATE users SET status = 'x'",
        "SELECT 1; DROP TABLE users",
        "SELECT load_file('/etc/passwd')",
        "SELECT * FROM information_schema.tables",
        "SELECT sleep(10)",
    ],
)
def test_raw_sql_refuses_mutating_or_dangerous_statements(db, sql):
    with SQLSandbox(sqlite_connector(db), SQLPolicy(allow_raw_select=True)) as box:
        with pytest.raises(SQLPolicyError):
            box.check_raw(sql)


def test_policy_rejects_a_nonsense_limit():
    with pytest.raises(ValueError, match="max_rows"):
        SQLPolicy(max_rows=0)


# ------------------------------------------------------ driver-agnostic details


def test_placeholder_style_follows_the_driver(db):
    with SQLSandbox(sqlite_connector(db)) as box:
        sql, _params = box.build_select("users", None, ["id = 1"], "", 5)
    assert "?" in sql and "%s" not in sql, "sqlite is qmark"


def test_the_generated_sql_is_exactly_what_was_intended(box):
    sql, params = box.build_select(
        "orders", ["id", "amount"], ["state = 'paid'", "amount > 10"], "amount desc", 5
    )
    assert sql == (
        'SELECT "id", "amount" FROM "orders" WHERE "state" = ? AND "amount" > ? '
        'ORDER BY "amount" DESC LIMIT 5'
    )
    assert params == ["paid", 10]


def test_identifiers_that_need_quoting_are_quoted(db):
    """Real schemas contain names like 长垣单井基础信息表, daa05-cy2 and 累积产液量/t."""
    conn = sqlite3.connect(db)
    conn.executescript(
        """
        CREATE TABLE "长垣单井基础信息表" ("井号" TEXT, "累积产液量/t" REAL);
        INSERT INTO "长垣单井基础信息表" VALUES ('well-1', 12.5);
        """
    )
    conn.commit()
    conn.close()

    with SQLSandbox(sqlite_connector(db)) as box:
        out = make_sql_tools(box)[2].func(
            "长垣单井基础信息表", columns=["井号", "累积产液量/t"], where=["井号 = 'well-1'"]
        )
    assert "well-1" in out and "12.5" in out


def test_a_name_containing_a_quote_is_still_refused(db):
    """Quoting is only safe because the name may not contain the quote character."""
    with SQLSandbox(sqlite_connector(db)) as box:
        for name in ['users"', "users`", "us'ers", "users x"]:
            with pytest.raises(SQLPolicyError):
                box.build_select(name, None, [], "", 10)


# -------------------------------------------------------- registry integration


def test_errors_from_sql_tools_reach_the_model_as_observations(box):
    """The registry boundary: a policy refusal is a ToolResult, not an exception."""
    registry = ToolRegistry(make_sql_tools(box))
    result = registry.execute(ToolCall(name="query_table", arguments={"table": "users; DROP TABLE x"}))
    assert result.ok is False
    assert "not a valid table name" in result.content


def test_the_queries_executed_are_recorded_as_an_audit_trail(box):
    tools_of(box)["query_table"].func("users", where=["status = 'active'"])
    assert any('FROM "users"' in q for q in box.queries)


def test_the_session_is_put_into_read_only_mode(box):
    names = tools_of(box)
    names["query_table"].func("users")  # forces the connection open
    assert box.read_only_state == "sqlite query_only"


def test_the_server_refuses_a_write_even_if_one_somehow_got_through(box):
    """The session guard, not the builder, is the last line of defence."""
    box.connection()  # applies PRAGMA query_only
    with pytest.raises(sqlite3.OperationalError, match="readonly database"):
        box.execute("INSERT INTO users (name) VALUES ('mallory')", [])
    _cols, rows = box.execute("SELECT COUNT(*) FROM users", [])
    assert rows[0][0] == 4


def test_read_only_enforcement_can_be_disabled(db):
    with SQLSandbox(sqlite_connector(db), SQLPolicy(force_read_only_session=False)) as box:
        assert box.read_only_state == "not applied"
