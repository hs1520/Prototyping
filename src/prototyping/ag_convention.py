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

from . import ag_profile as profile

CONVENTION = "CONVENTION"
SPEC_VALUED = "SPEC_VALUED"

#: Tiers exist because rule-set size is itself a variable. A leaner hand-written
#: prompt outscored the full rendered set on the same chain and checker, so
#: "state every rule" is not automatically the best policy and the split must be
#: measurable rather than assumed.
#: CORE — without it the package does not parse, or the A/G graph cannot be
#: extracted and composed at all.
#: REFINEMENT — what a particular safety pattern additionally requires once the
#: decomposition is already well formed.
CORE = "CORE"
REFINEMENT = "REFINEMENT"

#: Failure scope for the named obligations inside the aggregate priority check.
#: This belongs beside the obligation itself: the router must not maintain a
#: second list that can drift from what the checker and prompt call the rule.
PRIORITY_INPUT_OR_CONTRACT = "PRIORITY_INPUT_OR_CONTRACT"
PRIORITY_MODEL_WIRING = "PRIORITY_MODEL_WIRING"


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
    #: Whether the rule is needed for a well-formed decomposition at all (CORE) or
    #: refines an already-well-formed one (REFINEMENT). Lets rule-set size be
    #: ablated instead of assumed.
    tier: str = CORE
    #: Optional routing scope. Priority obligations use it to distinguish a
    #: missing/contradictory contract or response vocabulary (BLOCKED) from
    #: topology already authorised inside an existing behavior definition
    #: (dependency-closed surgical repair).
    failure_scope: str | None = None

    def __post_init__(self) -> None:
        if self.category not in (CONVENTION, SPEC_VALUED):
            raise ValueError(f"unknown category {self.category!r}")
        if self.tier not in (CORE, REFINEMENT):
            raise ValueError(f"unknown tier {self.tier!r}")
        if self.failure_scope not in (
            None, PRIORITY_INPUT_OR_CONTRACT, PRIORITY_MODEL_WIRING
        ):
            raise ValueError(
                f"{self.obligation_id}: unknown failure scope "
                f"{self.failure_scope!r}"
            )
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


@dataclass(frozen=True)
class PatternRoles:
    """The roles one safety pattern's invariants must fill, and how to say so.

    A pattern is *defined* by these roles: a locked-until-authorised-release model
    that never says where power loss leads has not stated the pattern, however it
    names its invariants. The checker therefore refuses an invariant set that
    leaves a role unfilled — which is CONVENTION, not gold, and must be published:
    stating "which roles" leaves "which concepts fill them" entirely to the author,
    exactly as a notation rule leaves the engineering to the engineer.

    Two measured chains failed on precisely this. Both stated well-formed
    invariants that simply did not cover the pattern's roles, because nothing in
    the prompt had ever said which roles the pattern has.
    """

    pattern: str
    #: Role names, identical to ``ag_contracts.PATTERN_INVARIANT_ROLES``; the two
    #: are pinned to each other in the tests.
    roles: Tuple[str, ...]
    #: The invariant shapes that fill them, in the decision format's terms.
    authoring_rule: str


#: What each invariant pattern's invariants must SAY. The shapes are the ones the
#: checker derives its roles from, read off the reference implementation rather
#: than assumed.
INVARIANT_ROLE_OBLIGATIONS: Tuple[PatternRoles, ...] = (
    PatternRoles(
        profile.LOCKED_UNTIL_RELEASE_PATTERN,
        profile.PATTERN_INVARIANT_ROLES[profile.LOCKED_UNTIL_RELEASE_PATTERN],
        "LOCKED_UNTIL_AUTHORISED_RELEASE is defined by five roles — locked, "
        "power_on, unlocked, authorisation, power — and needs one invariant per "
        "obligation, three in all:\n"
        "     (a) `<power-on concept> => <locked concept>`: the POWER_ON and "
        "LOCKED roles — the state the system defaults to when it powers on.\n"
        "     (b) `<unlocked concept> => <authorisation concept>`: the UNLOCKED "
        "and AUTHORISATION roles — being unlocked implies the authorisation was "
        "granted, so authorisation is the only way out of locked. The unlocked "
        "state may be a concept of its own or simply `not <locked concept>`; both "
        "state the same obligation and both are accepted.\n"
        "     (c) `not <power-available concept> => <locked concept>`: the POWER "
        "role — losing power returns to locked. Its antecedent is negated; no "
        "other antecedent in this pattern is.\n"
        "     The locked concept must be one the architecture actually produces: "
        "the component producing it is the mechanism whose behaviour realises the "
        "pattern, and that mechanism must be safe by default (see "
        "`lifecycle_events`).",
    ),
    PatternRoles(
        profile.STARTUP_INHIBIT_PATTERN,
        profile.PATTERN_INVARIANT_ROLES[profile.STARTUP_INHIBIT_PATTERN],
        "STARTUP_INHIBIT is defined by four roles — latch, reset, inhibited, "
        "forbidden — and needs one invariant per obligation, three in all:\n"
        "     (a) `<condition concepts> => not <forbidden> and not <forbidden>`: "
        "the FORBIDDEN role — name in this one invariant every state the inhibit "
        "must keep the system out of, each of them negated.\n"
        "     (b) `<latch concept> => <inhibited concept> [and <inhibited>]`: the "
        "LATCH and INHIBITED roles — what holding the latch inhibits. Nothing is "
        "negated here.\n"
        "     (c) `<reset concept> => not <latch concept>`: the RESET role, the "
        "event that clears the latch. This is the only invariant whose antecedent "
        "is one concept and whose consequent is one negated concept.\n"
        "     The latch concept must be one the architecture actually produces: "
        "the component producing it is the one whose behaviour the pattern is "
        "checked against.",
    ),
)


#: Obligations attached to individual decision fields. A decision prompt that
#: names a field without stating what the checker will do with it is the same
#: defect as an unstated convention: `timing_segment_required` was offered as a
#: bare `true/false`, a measured seed set it true for a component whose guarantee
#: is simply available at the boundary, and the run lost the timed chain to four
#: diagnostics at once (TIMING_BUDGET_EXCEEDED, REALIZATION_TRIGGER_MISSING,
#: REALIZATION_UNREACHABLE, PRIORITY_TOPOLOGY_INCOMPLETE/
#: recovery_power_available_at_boundary).
DECISION_FIELD_OBLIGATIONS: Tuple[Tuple[str, str], ...] = (
    (
        "timing_segment_required",
        "true only for a component that CONSUMES TIME on the path from the "
        "trigger to the response — one whose guarantee is produced in reaction to "
        "something upstream. A component whose guarantee is simply available at "
        "the boundary (a supply that is already on, not something that gets "
        "triggered) sets it false: it is realised as one steady state that "
        "establishes its guarantee, and a trigger-response machine cannot be "
        "built for it. This reading is the same under both triggered patterns — "
        "it says whether the component REACTS, not whether a deadline is being "
        "apportioned. So under THRESHOLD_TRIGGERED_RESPONSE a reacting component "
        "still sets it true, and pairs that with a null latency_budget_seconds.",
    ),
    (
        "lifecycle_events",
        "under an invariant pattern the component that guarantees the locked or "
        "latched concept must be safe by DEFAULT, which means it assumes NOTHING: "
        "list EVERY concept it consumes here, the authorisation included. A "
        "component that assumes the authorisation is not locked by default, it is "
        "locked while that authorisation happens to be absent — a different, "
        "weaker claim, and the topology check rejects it.",
    ),
    (
        "latency_budget_seconds",
        "under TRIGGERED_TIMED_FAILSAFE_RESPONSE, null exactly when "
        "timing_segment_required is false — a component that consumes no time on "
        "the path owns no part of the deadline. The budgets that are set must fit "
        "inside deadline_seconds once composed: they apportion the system "
        "deadline, they do not each restate it. Under THRESHOLD_TRIGGERED_"
        "RESPONSE it is ALWAYS null, for every component including the reacting "
        "ones: that pattern states no deadline, so there is nothing to apportion "
        "and a budget would be a number the requirement does not contain.",
    ),
    (
        "timing_segment_group",
        "optional, and only meaningful for segments that run CONCURRENTLY: give "
        "the same integer to segments that execute at the same time, and their "
        "group contributes its MAXIMUM instead of its sum. Omit it for a segment "
        "on the serial path — omitted means its own group, so the composition is "
        "the plain sum. Declare it when it is true: two 0.3 s responses running "
        "side by side occupy 0.3 s, and calling that 0.6 s would reject a design "
        "that meets its deadline.",
    ),
    (
        "timing_margin_seconds",
        "optional: deadline you deliberately do NOT apportion, checked as "
        "composed + margin <= deadline_seconds. How much reserve a safety "
        "response keeps is a design decision the requirement does not contain, so "
        "state it rather than leaving it implicit in an underspent budget.",
    ),
)


def render_decision_field_rules() -> str:
    """The per-field obligations, as prompt text for the decision schema."""
    return "\n".join(
        f"  * `{field}`: {rule}" for field, rule in DECISION_FIELD_OBLIGATIONS
    )


def render_invariant_role_rules() -> str:
    """The per-pattern role obligations, as prompt text.

    Rendered from the table above so a role the checker demands cannot be stated
    in one prompt and forgotten in another.
    """
    return "\n".join(
        f"  * {item.authoring_rule}" for item in INVARIANT_ROLE_OBLIGATIONS
    )


#: Obligations keyed by checker diagnostic code.
DIAGNOSTIC_OBLIGATIONS: Tuple[Obligation, ...] = (
    Obligation(
        profile.CODE_SOURCE_PROVENANCE_MISSING, CONVENTION,
        "The system contract's first member must be its provenance line: "
        "`doc /* bounded A/G system contract for <REQ>; safety_pattern=<PATTERN>; "
        "timing_origin=<assumption attribute> */`, and <REQ> must be the "
        "requirement actually present in the committed model.",
    ),
    Obligation(
        profile.CODE_CONTRACT_INCOMPLETE, CONVENTION,
        "Every contract declares each concept it uses as `attribute <name> : "
        "Boolean;`. Write one `assume constraint <n> { <concept> }` per CONTRACT "
        "assumption and one `require constraint <n> { <concept> }` per produced "
        "guarantee. A lifecycle/interface input used only to trigger behavior is "
        "not automatically a contract assumption.",
    ),
    Obligation(
        profile.CODE_SYSTEM_OBSERVATION_BINDING_MISSING, CONVENTION,
        "The SYSTEM's observed guarantee "
        "is a separate constraint named exactly `require constraint g_observed "
        "{ <observation> }`; invariant constraints do not replace it. The checker "
        "uses `g_observed` to decompose a compound observation into the positive "
        "concepts component guarantees must support.",
    ),
    Obligation(
        profile.CODE_COMPONENT_GUARANTEE_NONATOMIC, CONVENTION,
        "Every COMPONENT `require constraint` must be one atomic Boolean "
        "identifier, for example `require constraint g_locked { locked }`. Put "
        "each produced concept in its own constraint. Compound Boolean formulas "
        "belong on the system contract as observations or invariants, not in a "
        "component guarantee whose realizing action must establish one concept.",
    ),
    Obligation(
        profile.CODE_CONTRACT_UNSUPPORTED, CONVENTION,
        "Constraint bodies use only the bounded subset: a Boolean identifier, "
        "`not`, `and`, `or`, or a single `<var> == <ENUM>::<MEMBER>` / `!=` "
        "comparison. Anything richer cannot be represented and is rejected.",
    ),
    Obligation(
        profile.CODE_GUARANTEE_NO_OWNER, CONVENTION,
        "Each component contract is allocated to exactly one owning part, "
        "declaring the part definition before using it: `part def <Def>;` then "
        "`part <usage> : <Def>;` then `satisfy requirement <u> : <Contract> by "
        "<usage>;`.",
    ),
    Obligation(
        profile.CODE_GUARANTEE_MULTIPLE_OWNERS, CONVENTION,
        "Ownership must be unique: never satisfy the same contract by more than "
        "one part.",
    ),
    Obligation(
        profile.CODE_DECOMPOSITION_MISSING, CONVENTION,
        "Each component contract has exactly one decomposition edge from the "
        "system contract: `dependency decompose<X> from <SystemContract> to "
        "<ComponentContract>;`.",
    ),
    Obligation(
        profile.CODE_DECOMPOSITION_INSUFFICIENT, CONVENTION,
        "The component guarantees together must entail the system contract's "
        "observed guarantee; some component must produce it.",
    ),
    Obligation(
        profile.CODE_ASSUMPTION_UNDISCHARGED, CONVENTION,
        "Every assumption that is not an environment input must be produced by "
        "some upstream component in the same package.",
    ),
    Obligation(
        profile.CODE_DISCHARGE_EDGE_MISSING, CONVENTION,
        "A matching upstream guarantee is not enough: state the discharge "
        "explicitly as `dependency discharge<Concept>__to__<ConsumerContract> "
        "from <ProducerContract> to <ConsumerContract>;`, where `<Concept>` is "
        "the COMPLETE discharged assumption concept with only its first character "
        "capitalized. Use element names only, never dotted members.",
    ),
    Obligation(
        profile.CODE_CIRCULAR_ASSUMPTION, CONVENTION,
        "The discharge graph must be acyclic — two contracts may not discharge "
        "each other, directly or transitively.",
    ),
    Obligation(
        profile.CODE_UNIT_INCOMPATIBLE, CONVENTION,
        "Every timing budget carries an explicit SysML unit and they must all use "
        "the same one, e.g. `attribute maxLatency : DurationValue = <N> [s];`.",
           tier=REFINEMENT,
    ),
    Obligation(
        profile.CODE_TIMING_BUDGET_MISSING, CONVENTION,
        "Under a timed pattern the system contract carries `attribute maxLatency "
        ": DurationValue = <N> [s];`, and every component contributing a segment "
        "carries `attribute latencyBudget : DurationValue = <N> [s];` with "
        "`attribute timingSegmentRequired : Boolean = true;`.",
           tier=REFINEMENT,
    ),
    Obligation(
        profile.CODE_TIMING_BUDGET_EXCEEDED, CONVENTION,
        "Compose the component budgets and keep them inside the system deadline. "
        "Segments on the serial path ADD. Segments that run concurrently declare "
        "the same `attribute timingSegmentGroup : Integer = <n>;` and their group "
        "contributes its MAXIMUM, not its sum — omit the attribute for a serial "
        "segment. If the design keeps reserve, declare it on the system contract "
        "as `attribute timingMargin : DurationValue = <N> [s];` and the check "
        "becomes composed + margin <= deadline.",
           tier=REFINEMENT,
    ),
    Obligation(
        profile.CODE_REALIZATION_MISSING, CONVENTION,
        "Every contract is realized by exactly one state machine, linked as "
        "`dependency realize<Contract> from <Contract> to <ItsStateDef>;`.",
    ),
    Obligation(
        profile.CODE_REALIZATION_TRIGGER_MISSING, CONVENTION,
        "A realizing state machine must accept a trigger compatible with the "
        "contract's assumptions. Compatibility is lexical and uses the COMPLETE "
        "assumption concept: for assumption `<concept>`, name the event "
        "`<Concept>Signal` (capitalize only its first character), so assumption "
        "`sensorFailureReported` is accepted as "
        "`SensorFailureReportedSignal`, not `SensorFailureSignal`. Declare each "
        "accepted event as its own package-level `item def <Signal>;`. Keep "
        "response behaviors as separately named `action def` elements. Then write "
        "transitions in exactly this form — "
        "the guard clause is `if <boolean-expression>` and is written between "
        "`accept` and `then`, never as `guard`:\n"
        "   `transition <n> first <source> accept <Signal> then <target>;`\n"
        "   `transition <n> first <source> accept <Signal> if <expr> then "
        "<target>;`\n"
        "   Every concept a guard references must ALSO be declared inside that "
        "same `state def` as `attribute <name> : Boolean;` — a state machine "
        "cannot see the attributes of the contract it realizes, so an undeclared "
        "guard concept is an unresolved reference.",
    ),
    Obligation(
        profile.CODE_REALIZATION_UNREACHABLE, CONVENTION,
        "The state machine must have a reachable trigger-to-response path: an "
        "`entry; then <state>;` initial edge leading to the responding state.",
    ),
    Obligation(
        profile.CODE_REALIZATION_ACTION_MISSING, CONVENTION,
        "The responding state must perform the action that establishes the "
        "guarantee, as `state <t> { entry action set<GuaranteeConcept>; }` — the "
        "action name is the guarantee concept prefixed with `set`.",
    ),
    Obligation(
        profile.CODE_OBSERVATION_MISSING, CONVENTION,
        "The system contract is observed by exactly one verification element: "
        "`verification def <V> { objective <o> { verify requirement <r> : "
        "<SystemContract>; } }` plus `dependency observe<V> from <SystemContract> "
        "to <V>;`.",
    ),
    Obligation(
        profile.CODE_PATTERN_DECLARATION_INCONSISTENT, CONVENTION,
        "Declare `safety_pattern=` on the provenance line as exactly one of "
        "TRIGGERED_TIMED_FAILSAFE_RESPONSE, THRESHOLD_TRIGGERED_RESPONSE, "
        "STARTUP_INHIBIT, LOCKED_UNTIL_AUTHORISED_RELEASE, and classify the "
        "requirement yourself. Only TRIGGERED_TIMED_FAILSAFE_RESPONSE carries a "
        "timing budget; the other three must not. Choose "
        "THRESHOLD_TRIGGERED_RESPONSE when the requirement names a trigger, a "
        "response, and a precedence relation to other responses but states NO "
        "deadline — do not invent one to reach the timed pattern, and do not "
        "state a trigger-response obligation as an invariant. Whether your "
        "classification is the right one is scored by the evaluator, not "
        "enforced here.",
           tier=REFINEMENT,
    ),
    Obligation(
        profile.CODE_PATTERN_TOPOLOGY_INCOMPLETE, CONVENTION,
        "The declared pattern's state/transition topology must satisfy these "
        "gold-blind structural duties. STARTUP_INHIBIT: the component producing "
        "the latch has an initial decision state, exactly one non-initial state "
        "whose entry action is `set<Latch>`, at least one reset state whose entry "
        "action is `clear<Latch>`, latching and clearing are alternatives from a "
        "common source, reset uses a different trigger from latch, and no "
        "transition reaches a state the invariant forbids. "
        "LOCKED_UNTIL_AUTHORISED_RELEASE: the lock component has an initial state "
        "whose entry action establishes the locked concept, exactly one unlocked "
        "state is reached by `<AuthorisationConcept>Signal`, no other trigger "
        "reaches it, and a distinct power-loss event returns it to the initial "
        "locked state. The default-safe lock component has no assumptions and "
        "owns the locked, authorised-only, and de-energise-to-lock guarantees.",
           tier=REFINEMENT,
    ),
    Obligation(
        profile.CODE_INVARIANT_SEMANTICS_MISSING, CONVENTION,
        "An invariant pattern must state every invariant as a `require constraint` "
        "INSIDE the system contract. Its exact name is "
        "`inv__<invariant_id>__source__<source_id>__kind__<source_kind>`, where "
        "`source_kind` is `STAKEHOLDER` or "
        "`STUDENT_DERIVED_DESIGN_CONSTRAINT`. An invariant absent from SysML, "
        "placed in a separate requirement def, or lacking this provenance name "
        "does not exist.",
           tier=REFINEMENT,
    ),
    Obligation(
        profile.CODE_INVARIANT_SEMANTICS_INVALID, CONVENTION,
        "Each invariant must bind a parseable Boolean AST, its provenance, and "
        "the model elements it constrains, consistently with each other — and the "
        "invariants must together fill every role the declared pattern is defined "
        "by:\n" + render_invariant_role_rules(),
           tier=REFINEMENT,
    ),
    Obligation(
        profile.CODE_PRIORITY_TOPOLOGY_MISSING, CONVENTION,
        "A TRIGGERED_TIMED_FAILSAFE_RESPONSE must model its arbitration "
        "explicitly, using these fixed element names: `enum def <RESPONSE_SET> "
        "{ enum <MEMBER>; ... }` (each member uses the `enum` keyword), "
        "`requirement def SafetyResponsePriorityContract` carrying "
        "`doc /* bounded A/G evaluator semantic auxiliary; "
        "response_set_id=<RESPONSE_SET> */`, `attribute selectedResponse : "
        "<RESPONSE_SET>;`, `assume constraint priorityTrigger { <TRIGGER> }`, "
        "`require constraint selectHighestPriority { selectedResponse == "
        "<RESPONSE_SET>::<WINNER> }`, and `state def SafetyResponseArbitration`.",
           tier=REFINEMENT,
    ),
    Obligation(
        profile.CODE_PRIORITY_TOPOLOGY_INCOMPLETE, CONVENTION,
        "The arbitration must be internally consistent: one "
        "`require constraint precedence_<WINNER>_over_<LOSER> { not <TRIGGER> or "
        "selectedResponse != <RESPONSE_SET>::<LOSER> }` for every response the "
        "winner outranks; a selection transition guarded `if <TRIGGER>` reaching "
        "the winning state; and for each outranked response its own competing "
        "transition guarded `if not <TRIGGER>`, each accepting its own request "
        "signal, e.g. `transition select<LOSER> first <await> accept "
        "<LOSER>RequestSignal if not <TRIGGER> then <LOSER>;`.",
           tier=REFINEMENT,
    ),
)

#: Named obligations inside the PRIORITY_TOPOLOGY_INCOMPLETE check. These are
#: reported individually so a failure says which fact is wrong; each still needs a
#: category, because several compare against REQ_SAFE_005's reviewed answer.
PRIORITY_OBLIGATIONS: Tuple[Obligation, ...] = (
    Obligation(
        "response_member_provenance", CONVENTION,
        "Every response-set enum member must have a provenance doc in the "
        "priority contract: `response_member=<member>; "
        "source_kind=<EXISTING_MODEL_BEHAVIOR or approved/derived design kind>; "
        "source_id=<element-or-design-id>`. Interface signals and guarantees do "
        "not become selectable responses merely by being placed in the enum.",
           tier=REFINEMENT,
           failure_scope=PRIORITY_INPUT_OR_CONTRACT,
    ),
    Obligation(
        "response_set_members", CONVENTION,
        "The response set must be non-empty and must contain every response named "
        "by a precedence constraint — no edge may reference a member you did not "
        "declare in the `enum def`.",
           tier=REFINEMENT,
           failure_scope=PRIORITY_INPUT_OR_CONTRACT,
    ),
    Obligation(
        "precedence_edges", CONVENTION,
        "Precedence must be a single-winner ordering: state one precedence "
        "constraint for every other member of the response set, so the winner "
        "outranks all of them and none is left unordered.",
           tier=REFINEMENT,
           failure_scope=PRIORITY_INPUT_OR_CONTRACT,
    ),
    Obligation(
        "single_highest_response", CONVENTION,
        "Exactly one response may sit at the top of the ordering; two responses "
        "both outranking others is not a precedence the checker can interpret.",
           tier=REFINEMENT,
           failure_scope=PRIORITY_INPUT_OR_CONTRACT,
    ),
    Obligation(
        "selected_response", CONVENTION,
        "The response named by `selectHighestPriority` must be one of the members "
        "declared in the response-set `enum def`.",
           tier=REFINEMENT,
           failure_scope=PRIORITY_INPUT_OR_CONTRACT,
    ),
    Obligation(
        "trigger_concept", CONVENTION,
        "The arbitration trigger must be one of the system contract's own "
        "assumption concepts, not a concept introduced only in the arbitration.",
           tier=REFINEMENT,
           failure_scope=PRIORITY_INPUT_OR_CONTRACT,
    ),
    Obligation(
        "trigger_matches_timing_origin", CONVENTION,
        "The arbitration trigger must be the same concept the provenance line "
        "declares as `timing_origin=`.",
           tier=REFINEMENT,
           failure_scope=PRIORITY_INPUT_OR_CONTRACT,
    ),
    Obligation(
        "selection_guarded_by_trigger", CONVENTION,
        "The selection transition must be guarded by the trigger concept.",
           tier=REFINEMENT,
           failure_scope=PRIORITY_MODEL_WIRING,
    ),
    Obligation(
        "competing_transitions_guarded", CONVENTION,
        "Every response the winner outranks must have a competing transition "
        "carrying the guard `if not <TRIGGER>`, so it cannot fire while the "
        "trigger holds. A response with no such transition is unguarded.",
           tier=REFINEMENT,
           failure_scope=PRIORITY_MODEL_WIRING,
    ),
    Obligation(
        "selected_transition_reachable", CONVENTION,
        "The transition selecting the winning response must be reachable from the "
        "arbitration state machine's initial state. In the bounded A/G convention "
        "the initial edge is written `entry; then <awaiting-state>;`; preserve that "
        "form and make the selection transition's source reachable from it.",
           tier=REFINEMENT,
           failure_scope=PRIORITY_MODEL_WIRING,
    ),
    Obligation(
        "selection_action_connected", CONVENTION,
        "The selecting transition's action must set the response selection "
        "guarantee, not merely enter a state.",
           tier=REFINEMENT,
           failure_scope=PRIORITY_MODEL_WIRING,
    ),
    Obligation(
        "arbiter_guarantees", CONVENTION,
        "The arbiter contract must produce both the command it issues and the "
        "response-selected guarantee, so the selection is observable downstream.",
           tier=REFINEMENT,
           failure_scope=PRIORITY_INPUT_OR_CONTRACT,
    ),
    Obligation(
        "recovery_power_available_at_boundary", CONVENTION,
        "A component whose guarantee is available at the boundary (no timing "
        "segment) is realized by a single initial state named after that "
        "guarantee, with no transitions and an `entry action set<Concept>;`.",
           tier=REFINEMENT,
           failure_scope=PRIORITY_MODEL_WIRING,
    ),
    Obligation(
        "deployment_action_connected", CONVENTION,
        "The component realizing the system observation must perform the action "
        "that establishes it, named `set<ObservationConcept>`.",
           tier=REFINEMENT,
           failure_scope=PRIORITY_MODEL_WIRING,
    ),
    Obligation(
        "observation_connected", CONVENTION,
        "The verification observation link must itself be satisfied, not merely "
        "declared.",
           tier=REFINEMENT,
           failure_scope=PRIORITY_INPUT_OR_CONTRACT,
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
        "Declare every event accepted in a transition as exactly one package-level "
        "`item def <Signal>;` before it is used. Never declare an `action def`, "
        "`attribute def`, or other definition with the same package member name; "
        "executable responses use separately named `action def` elements.",
    ),
    Obligation(
        "bounded_construct_set", CONVENTION,
        "Use only the bounded construct set: `requirement def`, `attribute`, "
        "`assume`/`require constraint`, `part def`/`part`, `satisfy requirement "
        "... by ...`, `state def` containing `entry; then <state>;` / `state "
        "<s>;` / `state <s> { entry action <a>; }` / `transition <n> first <s> "
        "accept <Signal> [if <expr>] then <t>;`, `enum def`, `verification def`, "
        "`enum def <E> { enum <MEMBER>; }`, and `dependency`. Invent no new "
        "keywords — in particular there is no "
        "`guard` keyword: a transition guard is written `if <expr>`.",
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


def render_authoring_rules(tiers: Tuple[str, ...] = (CORE, REFINEMENT)) -> str:
    """The checker's obligations as a numbered rule block for a generation prompt.

    Rendered from the entries above rather than hand-written, so a rule added to
    the checker cannot be silently omitted from what the author is told.

    ``tiers`` selects how much to state. It defaults to everything, but rule-set
    size is a measured variable, not a settled one: a leaner prompt has scored
    better than the full set on the same chain and checker, so the ability to
    state only CORE exists to keep that comparable rather than anecdotal.
    """
    rules = [
        item.authoring_rule for item in ALL_OBLIGATIONS
        if item.authoring_rule and item.tier in tiers
    ]
    return "\n".join(f"{n}. {rule}" for n, rule in enumerate(rules, start=1))
