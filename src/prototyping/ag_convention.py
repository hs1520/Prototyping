"""Single source of truth for what the bounded A/G checker obliges an author to do.

Every obligation the checker enforces is enumerated here exactly once and placed in
one of two categories:

``CONVENTION``
    A rule of the notation: a required shape, element name, or structural link. The
    deterministic emitter satisfies these by construction, so they were never
    written down — which is precisely the defect this module fixes. A convention
    the generator is never told is unsatisfiable by any author and, worse, makes the
    checker's diagnostic unactionable: three separate measured Vertex runs stalled
    on rules that existed only inside the emitter and the checker.

``SPEC_VALUED``
    The checker compares against a per-requirement answer it holds internally (for
    example REQ_SAFE_005's response-set members and precedence edges). The *form*
    may be published; the *value* must never be, because that value is what the
    LLM-authored arm is measuring. Publishing it would turn a generation-accuracy
    number into a transcription score.

The authoring rules given to a generator are rendered from the ``CONVENTION``
entries here, so a rule cannot be added to the checker and silently omitted from
the prompt: ``test_option2_ag_convention.py`` fails if any checker diagnostic code
or named priority obligation has no entry, and fails if any withheld value reaches
the rendered prompt.

This module holds no gold and imports nothing from the evaluator or the checker; it
is plain data so that both sides can be cross-checked against it independently.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

CONVENTION = "CONVENTION"
SPEC_VALUED = "SPEC_VALUED"


@dataclass(frozen=True)
class Obligation:
    """One thing the checker requires, and whether it may be published."""

    obligation_id: str
    category: str
    #: Published to the author. Empty only when nothing about the obligation can be
    #: stated without revealing the answer it is checked against.
    authoring_rule: str = ""
    #: What the checker holds internally that must never reach a prompt.
    withheld: str = ""

    def __post_init__(self) -> None:
        if self.category not in (CONVENTION, SPEC_VALUED):
            raise ValueError(f"unknown category {self.category!r}")
        if self.category == CONVENTION and not self.authoring_rule:
            raise ValueError(
                f"{self.obligation_id}: a CONVENTION must state its rule, or an "
                "author cannot satisfy it"
            )
        if self.category == SPEC_VALUED and not self.withheld:
            raise ValueError(
                f"{self.obligation_id}: a SPEC_VALUED obligation must record what "
                "is withheld, or the leak check cannot enforce it"
            )


#: Obligations keyed by checker diagnostic code.
DIAGNOSTIC_OBLIGATIONS: Tuple[Obligation, ...] = (
    Obligation(
        "SOURCE_PROVENANCE_MISSING", CONVENTION,
        "The system contract's first member must be its provenance line: "
        "`doc /* bounded A/G system contract for <REQ>; safety_pattern=<PATTERN>; "
        "timing_origin=<assumption attribute> */`, and <REQ> must be the "
        "requirement actually present in the committed model.",
    ),
    Obligation(
        "CONTRACT_INCOMPLETE", CONVENTION,
        "Every contract declares each concept it uses as `attribute <name> : "
        "Boolean;`, one `assume constraint <n> { <concept> }` per consumed input "
        "and one `require constraint <n> { <concept> }` per produced guarantee.",
    ),
    Obligation(
        "CONTRACT_UNSUPPORTED", CONVENTION,
        "Constraint bodies use only the bounded subset: a Boolean identifier, "
        "`not`, `and`, `or`, or a single `<var> == <ENUM>::<MEMBER>` / `!=` "
        "comparison. Anything richer cannot be represented and is rejected.",
    ),
    Obligation(
        "GUARANTEE_NO_OWNER", CONVENTION,
        "Each component contract is allocated to exactly one owning part via "
        "`part <usage> : <Def>;` and `satisfy requirement <u> : <Contract> by "
        "<usage>;`.",
    ),
    Obligation(
        "GUARANTEE_MULTIPLE_OWNERS", CONVENTION,
        "Ownership must be unique: never satisfy the same contract by more than "
        "one part.",
    ),
    Obligation(
        "DECOMPOSITION_MISSING", CONVENTION,
        "Each component contract has exactly one decomposition edge from the "
        "system contract: `dependency decompose<X> from <SystemContract> to "
        "<ComponentContract>;`.",
    ),
    Obligation(
        "DECOMPOSITION_INSUFFICIENT", CONVENTION,
        "The component guarantees together must entail the system contract's "
        "observed guarantee; some component must produce it.",
    ),
    Obligation(
        "ASSUMPTION_UNDISCHARGED", CONVENTION,
        "Every assumption that is not an environment input must be produced by "
        "some upstream component in the same package.",
    ),
    Obligation(
        "DISCHARGE_EDGE_MISSING", CONVENTION,
        "A matching upstream guarantee is not enough: state the discharge "
        "explicitly as `dependency discharge<X> from <ProducerContract> to "
        "<ConsumerContract>;`, using element names only, never dotted members.",
    ),
    Obligation(
        "CIRCULAR_ASSUMPTION", CONVENTION,
        "The discharge graph must be acyclic — two contracts may not discharge "
        "each other, directly or transitively.",
    ),
    Obligation(
        "UNIT_INCOMPATIBLE", CONVENTION,
        "Every timing budget carries an explicit SysML unit and they must all use "
        "the same one, e.g. `attribute maxLatency : DurationValue = <N> [s];`.",
    ),
    Obligation(
        "TIMING_BUDGET_MISSING", CONVENTION,
        "Under a timed pattern the system contract carries `attribute maxLatency "
        ": DurationValue = <N> [s];`, and every component contributing a segment "
        "carries `attribute latencyBudget : DurationValue = <N> [s];` with "
        "`attribute timingSegmentRequired : Boolean = true;`.",
    ),
    Obligation(
        "TIMING_BUDGET_EXCEEDED", CONVENTION,
        "Component budgets are additive: their sum must not exceed the system "
        "deadline. Apportion the deadline across the contributing components.",
    ),
    Obligation(
        "REALIZATION_MISSING", CONVENTION,
        "Every contract is realized by exactly one state machine, linked as "
        "`dependency realize<Contract> from <Contract> to <ItsStateDef>;`.",
    ),
    Obligation(
        "REALIZATION_TRIGGER_MISSING", CONVENTION,
        "A realizing state machine must accept a trigger compatible with the "
        "contract's assumptions. Declare each accepted event as its own "
        "`attribute def <Signal>;` before using it in "
        "`transition <n> first <s> accept <Signal> then <t>;`.",
    ),
    Obligation(
        "REALIZATION_UNREACHABLE", CONVENTION,
        "The state machine must have a reachable trigger-to-response path: an "
        "`entry; then <state>;` initial edge leading to the responding state.",
    ),
    Obligation(
        "REALIZATION_ACTION_MISSING", CONVENTION,
        "The responding state must perform the action that establishes the "
        "guarantee, as `state <t> { entry action set<GuaranteeConcept>; }` — the "
        "action name is the guarantee concept prefixed with `set`.",
    ),
    Obligation(
        "OBSERVATION_MISSING", CONVENTION,
        "The system contract is observed by exactly one verification element: "
        "`verification def <V> { objective <o> { verify requirement <r> : "
        "<SystemContract>; } }` plus `dependency observe<V> from <SystemContract> "
        "to <V>;`.",
    ),
    Obligation(
        "PATTERN_DECLARATION_INCONSISTENT", SPEC_VALUED,
        authoring_rule=(
            "Declare `safety_pattern=` on the provenance line as exactly one of "
            "TRIGGERED_TIMED_FAILSAFE_RESPONSE, STARTUP_INHIBIT, "
            "LOCKED_UNTIL_AUTHORISED_RELEASE, and classify the requirement "
            "yourself. A timed pattern must carry a timing budget; an invariant "
            "pattern must not."
        ),
        withheld=(
            "which pattern each source requirement instantiates "
            "(_SOURCE_PATTERN_PROFILE) — the classification is the author's to "
            "derive"
        ),
    ),
    Obligation(
        "PATTERN_TOPOLOGY_INCOMPLETE", CONVENTION,
        "The declared pattern's state/transition topology must be present and "
        "conform to the bounded profile: the states, the guarded transitions, and "
        "the initial state the pattern requires.",
    ),
    Obligation(
        "INVARIANT_SEMANTICS_MISSING", CONVENTION,
        "An invariant pattern must state its invariant as constraints in the "
        "committed model; an invariant absent from SysML does not exist.",
    ),
    Obligation(
        "INVARIANT_SEMANTICS_INVALID", CONVENTION,
        "Each invariant must bind a parseable Boolean AST, its provenance, and "
        "the model elements it constrains, consistently with each other.",
    ),
    Obligation(
        "PRIORITY_TOPOLOGY_MISSING", CONVENTION,
        "A TRIGGERED_TIMED_FAILSAFE_RESPONSE must model its arbitration "
        "explicitly, using these fixed element names: `enum def <RESPONSE_SET>`, "
        "`requirement def SafetyResponsePriorityContract` carrying "
        "`doc /* bounded A/G evaluator semantic auxiliary; "
        "response_set_id=<RESPONSE_SET> */`, `attribute selectedResponse : "
        "<RESPONSE_SET>;`, `assume constraint priorityTrigger { <TRIGGER> }`, "
        "`require constraint selectHighestPriority { selectedResponse == "
        "<RESPONSE_SET>::<WINNER> }`, and `state def SafetyResponseArbitration`.",
    ),
    Obligation(
        "PRIORITY_TOPOLOGY_INCOMPLETE", SPEC_VALUED,
        authoring_rule=(
            "The arbitration must be internally consistent: one "
            "`require constraint precedence_<WINNER>_over_<LOSER> { not <TRIGGER> "
            "or selectedResponse != <RESPONSE_SET>::<LOSER> }` for every response "
            "the winner outranks, a competing transition guarded by `not "
            "<TRIGGER>` for each of those losers, and a reachable selection "
            "transition whose action sets the winning response."
        ),
        withheld=(
            "REQ_SAFE_005's reviewed response-set members and precedence edges "
            "(_SAFE005_PRIORITY_MEMBERS / _SAFE005_PRIORITY_EDGES) and the "
            "literal winning response — the vocabulary and the ordering are what "
            "the LLM-authored arm measures"
        ),
    ),
)

#: Named obligations inside the PRIORITY_TOPOLOGY_INCOMPLETE check. These are
#: reported individually so a failure says which fact is wrong; each still needs a
#: category, because several compare against REQ_SAFE_005's reviewed answer.
PRIORITY_OBLIGATIONS: Tuple[Obligation, ...] = (
    Obligation(
        "response_set_members", SPEC_VALUED,
        withheld="the reviewed member names of the response set",
    ),
    Obligation(
        "precedence_edges", SPEC_VALUED,
        withheld="the reviewed precedence ordering between responses",
    ),
    Obligation(
        "single_highest_response", SPEC_VALUED,
        withheld="the literal identity of the highest-priority response",
    ),
    Obligation(
        "selected_response", SPEC_VALUED,
        withheld="the literal identity of the selected response",
    ),
    Obligation(
        "trigger_concept", SPEC_VALUED,
        withheld="the literal trigger concept for this requirement",
    ),
    Obligation(
        "trigger_matches_timing_origin", CONVENTION,
        "The arbitration trigger must be the same concept the provenance line "
        "declares as `timing_origin=`.",
    ),
    Obligation(
        "selection_guarded_by_trigger", CONVENTION,
        "The selection transition must be guarded by the trigger concept.",
    ),
    Obligation(
        "competing_transitions_guarded", CONVENTION,
        "Every response the winner outranks must have a competing transition "
        "guarded by `not <TRIGGER>`, so it cannot fire when the trigger holds.",
    ),
    Obligation(
        "selected_transition_reachable", CONVENTION,
        "The transition selecting the winning response must be reachable from the "
        "arbitration state machine's initial state.",
    ),
    Obligation(
        "selection_action_connected", CONVENTION,
        "The selecting transition's action must set the response selection "
        "guarantee, not merely enter a state.",
    ),
    Obligation(
        "arbiter_guarantees", CONVENTION,
        "The arbiter contract must produce both the command it issues and the "
        "response-selected guarantee, so the selection is observable downstream.",
    ),
    Obligation(
        "recovery_power_available_at_boundary", CONVENTION,
        "A component whose guarantee is available at the boundary (no timing "
        "segment) is realized by a single initial state named after that "
        "guarantee, with no transitions and an `entry action set<Concept>;`.",
    ),
    Obligation(
        "deployment_action_connected", CONVENTION,
        "The component realizing the system observation must perform the action "
        "that establishes it, named `set<ObservationConcept>`.",
    ),
    Obligation(
        "observation_connected", CONVENTION,
        "The verification observation link must itself be satisfied, not merely "
        "declared.",
    ),
)

#: Obligations enforced by the syntax gate rather than by an A/G diagnostic code.
#: They have no checker code, so they are excluded from the "stale entry" check —
#: but they are every bit as unsatisfiable when unstated. The missing imports below
#: cost every seed of one measured run its first iteration.
GATE_OBLIGATIONS: Tuple[Obligation, ...] = (
    Obligation(
        "stdlib_imports", CONVENTION,
        "The package must open with exactly these three imports, which supply "
        "Boolean, DurationValue and the SI units — without them those type names "
        "do not resolve: `private import ScalarValues::*;` "
        "`private import ISQ::*;` `private import SI::*;`",
    ),
    Obligation(
        "dependency_endpoints_are_names", CONVENTION,
        "`dependency` endpoints are element NAMES only, never dotted member "
        "references: `dependency d from <A> to <B>;`, not `<A>.constraint`.",
    ),
    Obligation(
        "declared_accept_signals", CONVENTION,
        "Declare every event accepted in a transition as its own `attribute def "
        "<Signal>;` inside the package before it is used.",
    ),
    Obligation(
        "bounded_construct_set", CONVENTION,
        "Use only the bounded construct set: `requirement def`, `attribute`, "
        "`assume`/`require constraint`, `part def`/`part`, `satisfy requirement "
        "... by ...`, `state def` with `entry; then`/`transition ... accept ... "
        "then`/`state { entry action }`, `enum def`, `verification def`, and "
        "`dependency`. Invent no new keywords.",
    ),
)

ALL_OBLIGATIONS: Tuple[Obligation, ...] = (
    *GATE_OBLIGATIONS, *DIAGNOSTIC_OBLIGATIONS, *PRIORITY_OBLIGATIONS,
)


def withheld_values() -> Tuple[str, ...]:
    """Everything the checker holds that must never reach a generation prompt."""
    return tuple(
        item.withheld for item in ALL_OBLIGATIONS
        if item.category == SPEC_VALUED and item.withheld
    )


def render_authoring_rules() -> str:
    """The checker's obligations as a numbered rule block for a generation prompt.

    Rendered from the entries above rather than hand-written, so a rule added to
    the checker cannot be silently omitted from what the author is told.
    """
    rules = [item.authoring_rule for item in ALL_OBLIGATIONS if item.authoring_rule]
    return "\n".join(f"{n}. {rule}" for n, rule in enumerate(rules, start=1))
