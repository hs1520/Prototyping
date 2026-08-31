"""EOF brace balancing for output-budget truncation — deterministic, not LLM.

Measured category (run3): the assembly response hits its output budget and
stops mid-file with every present statement complete; the syntax gate then
buys a full LLM window repair whose entire edit is appending ``}`` lines.
The balancer eats exactly that shape and refuses everything else — a tail
cut mid-token means content was lost and the LLM repair must decide.
"""
from src.utils.sysml_text_utils import close_truncated_blocks, find_block_end


_TRUNCATED = """package Drone {
    part def Mechanism {
        attribute isLocked : Boolean = true;
        state def ReleaseBehavior {
            entry; then Locked;
            state Locked;"""


def test_a_statement_boundary_tail_is_closed_and_counted():
    balanced, closed = close_truncated_blocks(_TRUNCATED)
    assert closed == 3
    assert balanced.endswith("state Locked;\n}\n}\n}\n")
    # the result is scannable by the block tools every injector relies on
    # (the outermost close is the last character before the trailing newline)
    assert find_block_end(balanced, balanced.index("{")) == len(balanced) - 2


def test_a_mid_token_tail_is_refused_for_the_llm_repair_path():
    text = _TRUNCATED + "\n            state Releas"
    assert close_truncated_blocks(text) == (text, 0)


def test_a_tail_inside_a_block_comment_is_refused():
    text = _TRUNCATED + "\n            /* releasing means the payload"
    assert close_truncated_blocks(text) == (text, 0)


def test_a_tail_inside_a_string_is_refused():
    text = _TRUNCATED[:-1] + '\n            doc = "held'
    assert close_truncated_blocks(text) == (text, 0)


def test_a_trailing_line_comment_is_a_boundary():
    text = _TRUNCATED + "\n            // remaining states follow"
    balanced, closed = close_truncated_blocks(text)
    assert closed == 3
    assert balanced.endswith("}\n}\n}\n")


def test_balanced_text_is_untouched():
    text = _TRUNCATED + "\n}\n}\n}\n"
    assert close_truncated_blocks(text) == (text, 0)


def test_over_closed_text_is_never_padded():
    text = "package P {\n}\n}\n"
    assert close_truncated_blocks(text) == (text, 0)


def test_braces_inside_comments_and_strings_do_not_count():
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
