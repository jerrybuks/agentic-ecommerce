import asyncio
import inspect
import json
from pathlib import Path

from src.golden_cases_mcp import list_golden_cases, server
from src.golden_loader import FIELD_NAMES
from src.golden_loader import list_golden_cases as load_cases

FIXTURE = Path(__file__).parent.parent / "data" / "golden_data" / "test_cases.json"


def fixture_cases():
    return json.loads(FIXTURE.read_text(encoding="utf-8"))["test_cases"]


def test_preserves_all_fields_and_source_order():
    cases = load_cases()
    source = fixture_cases()
    assert [case["id"] for case in cases] == [case["id"] for case in source]
    assert [tuple(case) for case in cases] == [FIELD_NAMES] * len(source)
    assert cases == source


def test_exact_category_filter_and_unknown_category():
    assert load_cases("product_search") == [
        case for case in fixture_cases() if case["category"] == "product_search"
    ]
    assert load_cases("PRODUCT_SEARCH") == []
    assert load_cases("missing") == []


def test_tool_has_only_optional_category_argument():
    assert list(inspect.signature(list_golden_cases).parameters) == ["category"]
    assert inspect.signature(list_golden_cases).parameters["category"].default is None


def test_server_exposes_the_read_only_tool():
    tools = asyncio.run(server.list_tools())
    assert [tool.name for tool in tools] == ["list_golden_cases"]
