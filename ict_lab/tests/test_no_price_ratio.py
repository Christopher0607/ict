"""Project-wide rule (Phase 1 item 2): no calculation anywhere may divide by
close/open/high/low, since ratio adjustment is unsafe on a backadjusted
series -- only points, ticks, or R multiples. This scans the actual source
tree so a future PR can't quietly reintroduce a price ratio.
"""
from __future__ import annotations

import ast
from pathlib import Path

LAB_ROOT = Path(__file__).resolve().parents[1]
BANNED_TOKENS = ("close", "open", "high", "low")
SKIP_DIR_NAMES = {"tests"}


def _iter_source_files():
    for path in sorted(LAB_ROOT.rglob("*.py")):
        if any(part in SKIP_DIR_NAMES for part in path.relative_to(LAB_ROOT).parts):
            continue
        yield path


def _divisor_mentions_banned_token(node: ast.AST) -> str | None:
    text = ast.unparse(node).lower()
    for token in BANNED_TOKENS:
        if (
            text == token
            or text.endswith(f".{token}")
            or text.endswith(f"['{token}']")
            or text.endswith(f'["{token}"]')
        ):
            return token
    return None


class _DivisionVisitor(ast.NodeVisitor):
    def __init__(self, path: Path) -> None:
        self.path = path
        self.violations: list[str] = []

    def visit_BinOp(self, node: ast.BinOp) -> None:
        if isinstance(node.op, (ast.Div, ast.FloorDiv)):
            token = _divisor_mentions_banned_token(node.right)
            if token:
                self.violations.append(
                    f"{self.path}:{node.lineno}: divides by `{token}` "
                    f"({ast.unparse(node)}) — use points, ticks, or R multiples instead."
                )
        self.generic_visit(node)


def test_no_price_percentage_or_ratio_anywhere_in_source():
    violations: list[str] = []
    for path in _iter_source_files():
        tree = ast.parse(path.read_text(), filename=str(path))
        visitor = _DivisionVisitor(path)
        visitor.visit(tree)
        violations.extend(visitor.violations)

    assert not violations, "\n".join(violations)


def test_scanner_actually_catches_a_violation(tmp_path):
    bad_file = tmp_path / "bad.py"
    bad_file.write_text("pct = (row.close - entry) / row.close\n")
    tree = ast.parse(bad_file.read_text(), filename=str(bad_file))
    visitor = _DivisionVisitor(bad_file)
    visitor.visit(tree)
    assert visitor.violations, "scanner should have flagged division by .close"
