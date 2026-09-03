"""Deterministic EOF brace balancing for output-budget truncation.

In run3 the assembly response stopped mid-file with every present statement
complete, and the syntax gate bought a full LLM window repair that only
appended ``}`` lines. The balancer takes that shape and refuses the rest: a
tail cut mid-token lost content, so the LLM repair decides.
"""
from src.utils.sysml_text_utils import close_truncated_blocks, find_block_end


_TRUNCATED = """package Drone {
    part def Mechanism {
        attribute isLocked : Boolean = true;
        state def ReleaseBehavior {
            entry; then Locked;
            state Locked;"""


def test_boundary_tail_closed():
    balanced, closed = close_truncated_blocks(_TRUNCATED)
    assert closed == 3
    assert balanced.endswith("state Locked;\n}\n}\n}\n")
    # the result is scannable by the block tools every injector relies on
    # (the outermost close is the last character before the trailing newline)
    assert find_block_end(balanced, balanced.index("{")) == len(balanced) - 2


def test_mid_token_tail_refused():
    text = _TRUNCATED + "\n            state Releas"
    assert close_truncated_blocks(text) == (text, 0)


def test_comment_tail_refused():
    text = _TRUNCATED + "\n            /* releasing means the payload"
    assert close_truncated_blocks(text) == (text, 0)


def test_string_tail_refused():
    text = _TRUNCATED[:-1] + '\n            doc = "held'
    assert close_truncated_blocks(text) == (text, 0)


def test_line_comment_is_boundary():
    text = _TRUNCATED + "\n            // remaining states follow"
    balanced, closed = close_truncated_blocks(text)
    assert closed == 3
    assert balanced.endswith("}\n}\n}\n")


def test_balanced_text_untouched():
    text = _TRUNCATED + "\n}\n}\n}\n"
    assert close_truncated_blocks(text) == (text, 0)


def test_over_closed_not_padded():
    text = "package P {\n}\n}\n"
    assert close_truncated_blocks(text) == (text, 0)


def test_braces_in_comments_ignored():
    text = (
        "package P {\n"
        "    // a } in a comment\n"
        "    /* and { another } here */\n"
        '    doc /* "{" */\n'
        "    part def X;"
    )
    balanced, closed = close_truncated_blocks(text)
    assert closed == 1
    assert balanced.endswith("part def X;\n}\n")
